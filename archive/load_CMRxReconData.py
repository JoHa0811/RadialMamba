#%%
import h5py
import torch
import mrpro
import matplotlib.pyplot as plt
import numpy as np
import fastmri
from fastmri.data import transforms as T

from mrpro.data import CsmData
from mrpro.data import IData
from mrpro.data import SpatialDimension
from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.operators import SensitivityOp
from mrpro.operators.FourierOp import FourierOp

#%%
class CMRxReconData():
    def __init__(self):
         pass
    
    
    def radial_trajectory(self, n_spokes:int=240, n_k0:int=256, initial_angle:float=0):
            golden_angle=torch.pi * 0.618034
            center_sample = n_k0//2
            radial = torch.arange(0, n_k0,dtype=torch.float32) - center_sample
            spoke_idx=torch.arange(n_spokes)
            angle= (spoke_idx * golden_angle +initial_angle)[None,None,:,None]
            kx = radial * torch.cos(angle)
            ky = radial * torch.sin(angle)
            kz = torch.zeros(1, 1, 1, 1)
            return KTrajectory(kz, ky, kx)

    def loadCMRxReconData(self):
        file_name = "/home/global/mri_datasets/cmrxrecon24/MultiCoil/PreparedTrainingData/Cine/P001_cine_sax_512_246_12_11.h5"
        f = h5py.File(file_name, "r")
        k_data = torch.tensor(f["kspace"])
        k_data_ = k_data[:,:,0,...]
        
        ##### Reconstruct data using FFT
        coilwise = torch.fft.ifft2(k_data_)
        image = coilwise.abs().square().sum(1).sqrt()

        #image = image.squeeze(0).squeeze(1)
        for i in image:
            print(i.shape)
            plt.matshow(torch.abs(i))
            break

        #### Create Radial KTrajectory and Fourier Operator
        radial_ktraj = self.radial_trajectory()
        print(radial_ktraj.broadcasted_shape)
        self.fourier_op = FourierOp(recon_matrix=SpatialDimension(z=1, y=246, x=512),
                            encoding_matrix=SpatialDimension(z=1, y=246, x=512),
                            traj=radial_ktraj)
        
        us_radial_ktraj = self.radial_trajectory(n_spokes=1)
        self.undersampled_fourier_op = FourierOp(recon_matrix=SpatialDimension(z=1, y=246, x=512),
                            encoding_matrix=SpatialDimension(z=1, y=246, x=512),
                            traj=us_radial_ktraj)
        
        #### apply radial fourier_op to reconstructed data
        (self.us_rad_kdata,) = self.fourier_op.forward(coilwise.unsqueeze(0))
        (us_rad_image_data,) = self.fourier_op.H(self.us_rad_kdata)
        plt.matshow(us_rad_image_data.abs()[0,0,0,...])

        #self.one_d_multicoil_spoke = self.us_rad_kdata[:,:,1,:]
        #(one_d_multicoil_spoke_us_rad_image_data,) = self.fourier_op.H(self.one_d_multicoil_spoke.unsqueeze(0))
        #print(f"one_d_multicoil_spoke: {self.one_d_multicoil_spoke.shape}")
        #plt.matshow(one_d_multicoil_spoke_us_rad_image_data.abs()[0,0,0,...])
        #print(one_d_multicoil_spoke_us_rad_image_data.shape)
        one_d_multicoil_spoke_autoencoder = self.us_rad_kdata[0,0,:,0,:].unsqueeze(1)
        #print(f"one_d_multicoil_spoke_autoencoder: {one_d_multicoil_spoke_autoencoder.shape}")
        #(one_d_multicoil_spoke_us_rad_image_data,) = self.fourier_op.H(one_d_multicoil_spoke_autoencoder.unsqueeze(0).unsqueeze(0))
        return one_d_multicoil_spoke_autoencoder


#%%
def main():
    cmr = CMRxReconData()
    cmr.loadCMRxReconData()

if __name__ == "__main__":
    main()