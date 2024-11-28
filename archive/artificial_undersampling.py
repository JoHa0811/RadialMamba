#########################################################################
#                                                                       #
#   Reconstruct fully sampled data, then artificially undersample them  #
#                                                                       #
#########################################################################

#%%
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import mrpro
import torch
from mrpro.operators import SensitivityOp
from mrpro.data import CsmData
from mrpro.data import KData
from mrpro.data import IData
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.operators.FourierOp import FourierOp

from mrpro.data import KData
from mrpro.data import SpatialDimension
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.data.KTrajectory import KTrajectory
from pathlib import Path

# load data & reconstruct it
folderpath = R"/echo/hammac01/CEST_DATA/2024-09-11_Volunteer_Brain/"
#%%
for file in Path(folderpath).rglob("*_traj.h5"):
    
    #check if reco allready exists
    #if not Path.exists(Path(f'{str(file).split(".h5")[0]}.nii')):
        
    print(f"Reconstructing: {file}")
    kdata = KData.from_file(file, KTrajectoryIsmrmrd())
    print(kdata.data.shape)
    nx, ny = (kdata.header.recon_matrix.x,kdata.header.recon_matrix.y)
    n_slices = kdata.data.shape[2]
    kdata.header.recon_matrix = SpatialDimension(n_slices,nx,ny)
    kdata.header.encoding_matrix = SpatialDimension(n_slices,nx,ny)
    print(kdata.data.shape)

    # exclude CEST Data
    if kdata.data.shape[0] > 1:
       continue

    # Calculate dcf using the trajectory
    dcf = DcfData.from_traj_voronoi(kdata.traj)

    # Reconstruct average image for coil map estimation
    fourier_op = FourierOp(
        recon_matrix=kdata.header.recon_matrix,
        encoding_matrix=kdata.header.encoding_matrix,
        traj=kdata.traj,
    )
    (img,) = fourier_op.adjoint(kdata.data * dcf.data)

    # Calculate and apply coil maps
    idata = IData.from_tensor_and_kheader(img, kdata.header)
    csm = CsmData.from_idata_walsh(idata)
    csm_op = SensitivityOp(csm)
    (img_from_ismrmrd_traj,) = csm_op.adjoint(img)
    first_img = img_from_ismrmrd_traj.cpu()[0, 0, :, :]  #  images, z, y, x
    plt.matshow(torch.abs(first_img[first_img.shape[0]//2,...]), cmap='gray')
    
    break

#%%
# create undersampling radial trajectory
def make_trajectory(n_spokes:int=240, n_k0:int=256, initial_angle:float=0):
    golden_angle=torch.pi * 0.618034
    radial = torch.arange(0, n_k0,dtype=torch.float32)# - center_sample
    spoke_idx=torch.arange(n_spokes)
    angle= (spoke_idx * golden_angle +initial_angle)[None,None,:,None]
    kx = radial * torch.cos(angle)
    ky = radial * torch.sin(angle)
    kz = torch.zeros(1, 1, 1, 1)
    return KTrajectory(kz, ky, kx)
#%%
undersampling_traj = make_trajectory(n_spokes=240, n_k0=256)
#%%
# undersample original data with new trajectory
undersampling_fourier_op=FourierOp(
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
    traj=undersampling_traj,
    )@csm.as_operator() # "resample" with new trajectory

undersampled_k_data,= undersampling_fourier_op(img_from_ismrmrd_traj)
(undersampled_img,) = fourier_op.adjoint(undersampled_k_data.data)
# Calculate and apply coil maps
idata = IData.from_tensor_and_kheader(undersampled_img, kdata.header)
csm = CsmData.from_idata_walsh(idata)
csm_op = SensitivityOp(csm)
(undersampled_img_from_ismrmrd_traj,) = csm_op.adjoint(img)
#%%
undersampled_first_img = undersampled_img_from_ismrmrd_traj.cpu()[0, 0, :, :]  #  images, z, y, x
plt.matshow(torch.abs(undersampled_first_img[undersampled_first_img.shape[0]//2,...]), cmap='gray')
# %%

# ToDo: save undersampled data