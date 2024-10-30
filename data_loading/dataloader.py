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
from mrpro.data import KData
from mrpro.data import KTrajectory
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
        self.transform = transform
        self.h5_files, self.slcs_dnmcs = self._load_data()
        self.slcs_dnmcs_cumsum = torch.cumsum(self.slcs_dnmcs, dim=0)
        
    def _load_data(self):
        h5_files = [file for file in Path(self.root_dir).rglob("*.h5")]
        slcs_dnmcs = torch.zeros((len(h5_files),2))
        
        counter = 0
        for file in h5_files:
            f = h5py.File(file, "r")
            k_data = torch.tensor(f["kspace"])
            slcs_dnmcs[h5_files.index(file)] = torch.tensor([k_data.shape[0], k_data.shape[2]])
            counter += 1
            print(counter)
            if counter == 5:
                break
        
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
        
    def _create_fourier_operator(self, x_dim, y_dim):
        #### Create Radial KTrajectory and Fourier Operator
        random_initial_angle = random.randint(0,360)
        radial_ktraj = self._radial_trajectory(n_spokes=64, initial_angle=random_initial_angle)
        fourier_op = FourierOp(recon_matrix=SpatialDimension(z=1, y=y_dim, x=x_dim),
                            encoding_matrix=SpatialDimension(z=1, y=y_dim, x=x_dim),
                            traj=radial_ktraj)
        return fourier_op
        
    def __len__(self):
        return self.slcs_dnmcs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        slcs_dnmcs_product = self.slcs_dnmcs[:,0] * self.slcs_dnmcs[:,1]
        slcs_dnmcs_product_cumsum = torch.cumsum(slcs_dnmcs_product, dim=0)
        if idx > sum(slcs_dnmcs_product):
            raise IndexError(f"Index does not exist, maximum index: {int(sum(slcs_dnmcs_product))-1}")
        
        cumsum_index = (slcs_dnmcs_product_cumsum <= idx).nonzero(as_tuple=True)[0]
        if cumsum_index.numel() > 0:
            slc_index = cumsum_index[-1]
        else:
            slc_index = 0
        self.slcs_dnmcs[slc_index]
        h5_file_to_load = self.h5_files[slc_index]
        f = h5py.File(h5_file_to_load, "r")
        k_data = torch.tensor(f["kspace"])
        
        available_indices = k_data.shape[0]*k_data.shape[2]
        data_slice_index = 0
        data_dynamics_index = 0
        if slc_index > k_data.shape[0]:
            data_slice_index = slc_index // k_data.shape[0]
            data_dynamics_index = slc_index - data_slice_index
            
        
        #ToDo: indexing!
        coilwise = torch.fft.ifft2(k_data[data_slice_index,:,data_dynamics_index,...])
            
        fourier_op = self._create_fourier_operator(x_dim=coilwise.shape[-1],
                                                   y_dim=coilwise.shape[-2])
        (us_rad_kdata,) = fourier_op.forward(coilwise.unsqueeze(0).unsqueeze(2))
        (us_rad_image_data,) = fourier_op.H(us_rad_kdata)
        
        
        return us_rad_kdata, us_rad_image_data
    
#%%
#def main():
dataloader = CMRxReconDataset()
data, image_data = dataloader[34]
#k_data = dataloader._load_data()
# %%
