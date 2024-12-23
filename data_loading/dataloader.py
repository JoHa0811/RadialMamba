# Schreibe dir ein pytorch dataset.
# Der dataset hat ein getitem, Len und init.
# In init, schau nach welche h5 Dateien es gibt mit Wieviel slices*dynamics jeweils. 
# Mache eine Liste mit Dateinamen und ein tensor mit den zahlen slices*dynamics in gleicher Reihenfolge.
# In dem def getitem(index),  mache auf CPU:
# Cumsum über die slices*dynamics Liste.
# Größe wert in der cumsum finden der kleiner ist als der index. Zugehörigen Datei Namen rausfinden.
# Die Datei öffnen und den richtigen slice und dynamic laden.
# Neue zufällige radial trajectory machen mit , sagen wir mal, 64 spokes bei komplett zufalligen winkeln.
# K raum sagen mittels Fourier op erzeugen.
# Zuruckgeben.
# Also datensatz[0] gibt dir 64 komplett zufallig gedrehte spokes aus slice 0 dynamic 0 der ersten datei zurück.
#%%
import os
import torch
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from mrpro.operators.FourierOp import FourierOp
from mrpro.data import SpatialDimension
from mrpro.data.DcfData import DcfData
from mrpro.data import KData
from mrpro.data import KTrajectory
from einops import rearrange
import h5py

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#%%
class CMRxReconDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir="/home/global/mri_datasets/cmrxrecon24/MultiCoil/PreparedTrainingData/Cine", transform=None):
        """
        Arguments:
            root_dir (string): Directory containing the h5 files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Normalize((0.5), (0.5))])
        self.h5_files, self.slcs_dnmcs = self._load_data()
        self.slcs_dnmcs_cumsum = torch.cumsum(self.slcs_dnmcs, dim=0)
        
    def _load_data(self):
        h5_files = [file for file in Path(self.root_dir).rglob("*sax*.h5")]
        slcs_dnmcs = torch.zeros((len(h5_files),2))
        
        counter = 0
        for file in h5_files:
            #slcs_dnmcs[h5_files.index(file)] = torch.tensor([torch.tensor(f["kspace"]).shape[0], torch.tensor(f["kspace"]).shape[2]])
            slcs_dnmcs[h5_files.index(file)] = torch.tensor([int(str(file).split(".h5")[0].split("_")[-1]), int(str(file).split(".h5")[0].split("_")[-2])])
            counter += 1

        print(f"Loaded {len(h5_files)} files.")
        slcs_dnmcs = slcs_dnmcs[slcs_dnmcs.sum(dim=1) != 0]
        return h5_files, slcs_dnmcs
    
    def _radial_trajectory(self, n_spokes:int=240, n_k0:int=256, initial_angle:float=0):
            golden_angle=torch.pi * 0.618034
            center_sample = n_k0//2
            radial = torch.arange(0, n_k0,dtype=torch.float32) - center_sample
            spoke_idx=torch.arange(n_spokes)
            angle= (spoke_idx * golden_angle +initial_angle)[None,None,:,None]
            kx = radial * torch.cos(angle)
            ky = radial * torch.sin(angle)
            kz = torch.zeros(1, 1, 1, 1)
            return KTrajectory(kz, ky, kx)
        
    def _create_dcf(self, n_spokes=128, n_k0=256, initial_angle=random.randint(0,360), batch_size=1):
        dcf = DcfData.from_traj_voronoi(self._radial_trajectory(n_spokes, n_k0, initial_angle))
        return dcf
        
    def _create_fourier_operator(self, x_dim, y_dim):
        #### Create Radial KTrajectory and Fourier Operator
        random_initial_angle = random.randint(0,360)
        radial_ktraj = self._radial_trajectory(n_spokes=128, initial_angle=random_initial_angle)
        fourier_op = FourierOp(recon_matrix=SpatialDimension(z=1, y=y_dim, x=x_dim),
                            encoding_matrix=SpatialDimension(z=1, y=y_dim, x=x_dim),
                            traj=radial_ktraj)
        return fourier_op
        
    def __len__(self):
        #return int(sum(self.slcs_dnmcs[:,0] * self.slcs_dnmcs[:,1]))
        return self.slcs_dnmcs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        slcs_dnmcs_product = self.slcs_dnmcs[:,0] * self.slcs_dnmcs[:,1]
        slcs_dnmcs_product_cumsum = torch.cumsum(slcs_dnmcs_product, dim=0)
        if idx > sum(slcs_dnmcs_product):
            raise IndexError(f"Index does not exist, maximum index: {int(sum(slcs_dnmcs_product))-1}")
        
        cumsum_index = (slcs_dnmcs_product_cumsum > idx).nonzero(as_tuple=True)[0]
        if cumsum_index.numel() > 0:
            file_index = cumsum_index[-1]
        else:
            file_index = 0
        self.slcs_dnmcs[file_index]
        h5_file_to_load = self.h5_files[file_index]
        f = h5py.File(h5_file_to_load, "r")
        
        
        available_indices = f["kspace"].shape[0]*f["kspace"].shape[2]
        slcs_dnmcs_product_cumsum_index = slcs_dnmcs_product_cumsum[0:file_index]
        if slcs_dnmcs_product_cumsum_index.numel() > 0:
            slcs_dnmcs_available = idx - slcs_dnmcs_product_cumsum[0:file_index][-1]
        else:
            slcs_dnmcs_available = idx
        
        data_slice_index = 0
        data_dynamics_index = 0
        if slcs_dnmcs_available > f["kspace"].shape[0]:
            data_dynamics_index = (slcs_dnmcs_available // f["kspace"].shape[0])-1
            if data_slice_index > f["kspace"].shape[0]:
                data_dynamics_index = data_dynamics_index-f["kspace"].shape[2]
                data_slice_index += 1
            #data_slice_index = slc_index - data_slice_index
        
        k_data = torch.tensor(f["kspace"][int(data_slice_index),:,int(data_dynamics_index),...])
        f.close()
        coilwise = torch.fft.ifft2(k_data)
        
        #old_threads=torch.get_num_threads()
        #torch.set_num_threads(1)
        self.fourier_op = self._create_fourier_operator(x_dim=coilwise.shape[-1],
                                                   y_dim=coilwise.shape[-2])
        (us_rad_kdata,) = self.fourier_op.forward(coilwise.unsqueeze(0).unsqueeze(2))
        
        us_rad_kdata = torch.fft.fftshift(us_rad_kdata,dim=-1)
        us_rad_kdata = torch.fft.ifft(us_rad_kdata, dim=-1)
        us_rad_kdata = torch.fft.fftshift(us_rad_kdata,dim=-1)
        
        #torch.set_num_threads(old_threads)
        
        #(us_rad_image_data,) = self.fourier_op.H(us_rad_kdata)
        us_rad_kdata = torch.view_as_real(us_rad_kdata)
        us_rad_kdata = rearrange(us_rad_kdata.squeeze(0).squeeze(1), "c s k0 r -> s (c r) k0")
        
        #us_rad_kdata -= us_rad_kdata.min(1, keepdim=True)[0]
        #us_rad_kdata /= us_rad_kdata.max(1, keepdim=True)[0]
        
        return us_rad_kdata / 2.0e-06#, coilwise
    
#%%
#def main():
dataset = CMRxReconDataset(root_dir="/echo/hammac01/RadialMamba/Data/Cine")
data = dataset[36]
# #%%
# for i in range(50):
#     print(i, dataset[i].shape)
#dataset[0]
# k_data = dataloader._load_data()
# %%
# dataloader = DataLoader(dataset, batch_size=5,
#                         shuffle=True, num_workers=0,
#                         collate_fn=lambda sampled: torch.concat(sampled, dim=0))
# # %%
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched.shape)
    
# %%
