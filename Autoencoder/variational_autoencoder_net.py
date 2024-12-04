#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt


#%%
class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding="same")
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding="same")
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(64, 4, kernel_size=1)
        
    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = self.conv4(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample0 = nn.Upsample(scale_factor=4)
        self.deconv1 = nn.Conv1d(
            4, 64, kernel_size=3, padding="same"
        )
        self.upsample1 = nn.Upsample(scale_factor=4)
        self.deconv2 = nn.Conv1d(
            64, 64, kernel_size=3, padding="same"
        )
        self.upsample2 = nn.Upsample(scale_factor=8)
        self.deconv3 = nn.Conv1d(
            64, 64, kernel_size=3, padding="same"
        )
        self.upsample3 = nn.Upsample(scale_factor=8)
        self.deconv4 = nn.Conv1d(
            64, 20, kernel_size=3, padding="same"
        )

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = self.upsample1(x)
        x = F.relu(self.deconv2(x))
        x = self.upsample2(x)
        x = F.relu(self.deconv3(x))
        x = self.upsample3(x)
        x = self.deconv4(x)
        return x # Output shape should be [batch_size, 20, 256]

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         # Deconvolution layers to increase channels and dimensions
#         self.deconv1 = nn.ConvTranspose1d(4, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv2 = nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv3 = nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv4 = nn.ConvTranspose1d(64, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv5 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv6 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv7 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv8 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1)

#     def forward(self, x):
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = F.relu(self.deconv3(x))
#         x = F.relu(self.deconv4(x))
#         x = F.relu(self.deconv5(x))
#         x = F.relu(self.deconv6(x))
#         x = F.relu(self.deconv7(x))
#         x = self.deconv8(x)  # Output shape should be [batch_size, 20, 256]
#         return x

#%%
class VAE(nn.Module):
    def __init__(self, latent_dim=4, image_size=256, channels=20, embedding_dim=16, device="cpu"):
        super(VAE, self).__init__()

        self.device = device
        
        self.encoder = Encoder(channels)
        self.decoder = Decoder()        

        # latent mean and variance 
        self.mean_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        
    def encode(self, x):
        x = self.encoder(x)
        x_flattened = x.view(x.size(0), -1) # x.shape = torch.Size([512, 4, 64])
        mean, logvar = self.mean_layer(x_flattened), self.logvar_layer(x_flattened)
        return mean, logvar
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z
    
    def decode(self, z):
        # Find the correct shape to reshape z (make sure it's divisible by batch size)
        batch_size = z.size(0)
        z_unflattened = z.view(batch_size, 4, -1)  # Dynamically calculate the last dimension       
        x_hat = self.decoder(z_unflattened)  # Decoder expects [batch_size, 4, something]
        return x_hat
        #return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar