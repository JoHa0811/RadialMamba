# encoder:
# conv1d(20,64,kernel_size=3,padding="same")
# relu
# conv1d(64,64,kernel_size=3,padding="same")
# relu
# maxpool(2)
# conv1d(64,64,kernel_size=3,padding="same")
# relu
# maxpool(2)
# conv1d(64,4,kernel_size=1,padding="same")
#
# decoder:
# conv1d(4,64,kernel_size=3,padding=same)
# relu
# upsamle(linear interpolation)
# conv1d(64,64,kernel_size=3,padding=same)
# relu
# upsamle(linear interpolation)
# conv1d(64,64,kernel_size=3,padding=same)
# relu
# conv1d(64,20,kernel_size=1,padding=same)

#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt


#%%
class Encoder(nn.Module):
    def __init__(self, image_size, channels, embedding_dim):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding="same")
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding="same")
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(64, 4, kernel_size=1)
        
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        #self.shape_before_flattening = None
        # compute the flattened size after convolutions
        #flattened_size = image_size
        # define fully connected layer to create embeddings
        #print(flattened_size)
        #print(embedding_dim)
        #self.fc = nn.Linear(4096, out_features=embedding_dim)
    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = self.conv4(x)
        # store the shape before flattening
        #self.shape_before_flattening = x.shape[1:]
        # flatten the tensor
        #x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        #print(x.shape)
        #x = self.fc(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # define fully connected layer to unflatten the embeddings
        #self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        # store the shape before flattening
        #self.reshape_dim = shape_before_flattening
        # define transpose convolutional layers
        self.deconv1 = nn.Conv1d(
            4, 64, kernel_size=3, padding="same"
        )
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.deconv2 = nn.Conv1d(
            64, 64, kernel_size=3, padding="same"
        )
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.deconv3 = nn.Conv1d(
            64, 64, kernel_size=3, padding="same"
        )
        self.deconv4 = nn.Conv1d(
            64, 20, kernel_size=3, padding="same"
        )
        # define final convolutional layer to generate output image
        #self.conv1 = nn.Conv1d(32, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # apply fully connected layer to unflatten the embeddings
        #x = self.fc(x)
        # reshape the tensor to match shape before flattening
        #print(x.shape)
        #x = x.view(x.size(0), *self.reshape_dim)
        # apply ReLU activations after each transpose convolutional layer
        x = F.relu(self.deconv1(x))
        print(x.shape)
        x = self.upsample1(x)
        x = F.relu(self.deconv2(x))
        print(x.shape)
        x = self.upsample2(x)
        x = F.relu(self.deconv3(x))
        print(x.shape)
        x = self.deconv4(x)

        # apply sigmoid activation to the final convolutional layer to generate output image
        #x = torch.sigmoid(self.conv1(x))
        return x
    
    
#%%
# check dimensions
import sys
sys.path.append('/echo/hammac01/RadialMamba/data_loading')
from dataloader import CMRxReconDataset
dataset = CMRxReconDataset()
data = dataset[34]
# %%
encoder = Encoder(image_size=data.shape[2], channels=data.shape[1], embedding_dim=16)
x_out = encoder(data)
#%%
decoder = Decoder()
x_out_rearrange = rearrange(x_out, "s c k0 -> c s k0")
decoded_x = decoder(x_out)

#%%
decoded_x = rearrange(decoded_x, "b (c r) k0 -> b c k0 r", c=10, r=2).contiguous()
decoded_x = torch.view_as_complex(decoded_x)
#%%
#us_rad_kdata.shape
#torch.Size([1, 10, 1, 64, 256])
decoded_x_rearranged = rearrange(decoded_x.unsqueeze(0).unsqueeze(2), "d b i c k0 -> d c i b k0")
(us_rad_image_data,) = dataset.fourier_op.H(decoded_x_rearranged)
# %%
plt.matshow(us_rad_image_data.abs().detach().numpy()[0,0,0,...])
# %%
