#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mrpro.data import CsmData
from mrpro.data import IData
from mrpro.data import SpatialDimension
from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.operators import SensitivityOp
from mrpro.operators.FourierOp import FourierOp
import matplotlib.pyplot as plt
#%%
class Encoder(nn.Module):
    def __init__(self, image_size, channels, embedding_dim):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = None
        # compute the flattened size after convolutions
        flattened_size = image_size
        # define fully connected layer to create embeddings
        print(flattened_size)
        print(embedding_dim)
        self.fc = nn.Linear(4096, out_features=embedding_dim)
    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        # store the shape before flattening
        self.shape_before_flattening = x.shape[1:]
        # flatten the tensor
        x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        print(x.shape)
        x = self.fc(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening, channels):
        super(Decoder, self).__init__()
        # define fully connected layer to unflatten the embeddings
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        # store the shape before flattening
        self.reshape_dim = shape_before_flattening
        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose1d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose1d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose1d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # define final convolutional layer to generate output image
        self.conv1 = nn.Conv1d(32, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # apply fully connected layer to unflatten the embeddings
        x = self.fc(x)
        # reshape the tensor to match shape before flattening
        print(x.shape)
        x = x.view(x.size(0), *self.reshape_dim)
        # apply ReLU activations after each transpose convolutional layer
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # apply sigmoid activation to the final convolutional layer to generate output image
        x = torch.sigmoid(self.conv1(x))
        return x
    
#%%
# check dimensions
import sys
sys.path.append('/echo/hammac01/RadialMamba/data_loading')
from load_CMRxReconData import CMRxReconData
cmrrecon = CMRxReconData()
# %%
spoke = cmrrecon.loadCMRxReconData()
# %%
encoder = Encoder(image_size=spoke.shape[2], channels=spoke.shape[0]*2, embedding_dim=16)
#%%
x = torch.view_as_real(torch.permute(spoke,(1,0,2))).permute(3,0,1,2)
x_rearrange = rearrange(x, "b s c k0 -> s (b c) k0")
encoded_x = encoder(x_rearrange)
#%%
decoder = Decoder(embedding_dim=16,shape_before_flattening=encoder.shape_before_flattening,
                  channels=spoke.shape[0]*2)
decoded_x = decoder(encoded_x)
#%%
decoded_x_rearrange = rearrange(decoded_x, "s (b c) k0 -> b s c k0",b=2,c=10).permute(1,2,3,0).contiguous()
#%%
decoded_x_complex = rearrange(torch.view_as_complex(decoded_x_rearrange),"s c k0 -> c s k0")
# %%
radial_ktraj = cmrrecon.radial_trajectory(n_spokes=1)
fourier_op = FourierOp(recon_matrix=SpatialDimension(z=1, y=246, x=512),
                        encoding_matrix=SpatialDimension(z=1, y=246, x=512),
                        traj=radial_ktraj)
(image,) = fourier_op.H(decoded_x_complex.unsqueeze(0).unsqueeze(0))
#image = cmrrecon.fourier_op.H(decoded_x.unsqueeze(0))
# %%
