#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, repeat
import matplotlib.pyplot as plt

import sys
from tqdm import tqdm
import neptune
sys.path.append('/echo/hammac01/RadialMamba/data_loading')
sys.path.append('/echo/hammac01/RadialMamba/Autoencoder')
from dataloader import dataset
from variational_autoencoder_net import VAE

#%%
# Logging to Neptune
run = neptune.init_run(
    project="MRI/RadialMamba",
    source_files=["**/**/*.py"],
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmU5Y2Q4MC04N2MxLTQxMzAtYWJmYS00YWFlOTcxNDg4MWIifQ==",
)  # your credentials
#%%
# Initialize net
device = "cuda" if torch.cuda.is_available() else "cpu"
vae_net = VAE(image_size=256, channels=20, embedding_dim=16, device=device).to(device)

valds, trainingds = torch.utils.data.random_split(dataset, [0.2, 0.8], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(trainingds, batch_size=4,
                        shuffle=True, num_workers=0
                        ,collate_fn=lambda sampled: torch.concat(sampled, dim=0))
val_loader = DataLoader(valds, batch_size=4,
                        shuffle=True, num_workers=0,
                        collate_fn=lambda sampled: torch.concat(sampled, dim=0))
# Validation using MSE Loss function
#loss_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(
    vae_net.parameters(), lr=1e-3
)

#%%
# define KL divergence and MSE loss (for continuos latent space)
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x)
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
#%%
params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params
n_epochs = 100
#%%
# initialize the best validation loss as infinity
best_val_loss = float("inf")
# start training by looping over the number of epochs
for epoch in range(n_epochs):
    print(f"Epoch: {epoch + 1}/{n_epochs}")
    # set the encoder and decoder models to training mode
    vae_net.train()
    # initialize running loss as 0
    running_loss = 0.0
    # loop over the batches of the training dataset
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # move the data to the device (GPU or CPU)
        data = data.to(device)
        # reset the gradients of the optimizer
        optimizer.zero_grad()
        # forward pass: encode the data and decode the encoded representation
        x_hat, mean, log_var = vae_net(data)
        # compute the reconstruction loss between the decoded output and
        # the original data

        loss = loss_function(data, x_hat, mean, log_var)
        # backward pass: compute the gradients
        loss.backward()
        # update the model weights
        optimizer.step()
        # accumulate the loss for the current batch
        running_loss += loss.item()
        run["train/batch/loss"].append(loss.item())
        
        # compute the average training loss for the epoch
    train_loss = running_loss / len(trainingds)
    run["train/loss"].append(train_loss)
    # compute the validation loss

    running_val_loss = 0.0
    
    with torch.no_grad():
        
        print("Validation mode")
        
        vae_net.eval()
        
        for batch_idx, val_data in tqdm(enumerate(val_loader), total=len(val_loader)):
            
            val_data = val_data.to(device)
            
            val_x_hat, val_mean, val_log_var = vae_net(val_data)
            
            validation_loss = loss_function(val_data, val_x_hat, val_mean, val_log_var)
            
            running_val_loss += validation_loss.item()
            
            run["valid/batch/loss"].append(validation_loss.item())
            
            print(f"Validation Loss: {running_val_loss}")
    
    val_loss = running_val_loss / len(valds)
    run["valid/loss"].append(val_loss)
    # print training and validation loss for current epoch
    print(
        f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f}"
    )
    #save best model weights based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            vae_net.state_dict(),
            f"/echo/hammac01/RadialMamba/Autoencoder/logged_weights/epoch_{epoch}.pt",
        )
        run["model_checkpoints/my_model"].upload(f"/echo/hammac01/RadialMamba/Autoencoder/logged_weights/epoch_{epoch}.pt")
        print(f"Saved epoch {epoch} as new best model.")
        
        ## Reconstruction
        decoded_val = rearrange(val_x_hat, "b (c r) k0 -> b c k0 r", c=10, r=2).contiguous()
        decoded_val_real = torch.view_as_complex(decoded_val)

        decoded_val = torch.fft.fftshift(decoded_val_real,dim=-1)
        decoded_val = torch.fft.fft(decoded_val, dim=-1)
        decoded_val = torch.fft.fftshift(decoded_val,dim=-1)
        decoded_val = repeat(decoded_val, "(b k1) c k0->b c k2 k1 k0",k2=1, k1=128)
        
        # include dcf
        dcf = dataset._create_dcf()
        (val_image_data,) = dataset.fourier_op.H(decoded_val.to("cpu") * dcf.data.to("cpu"))
        
        
        # decoded_val = rearrange(val_x_hat, "b (c r) k0 -> b c k0 r", c=10, r=2).contiguous()
        # decoded_val = torch.view_as_complex(decoded_val)
        # #us_rad_kdata.shape
        # #torch.Size([1, 10, 1, 64, 256])
        
        # # in case of 1D transformation: undo 1D transformation first
        # decoded_val = torch.fft.fftshift(decoded_val,dim=-1)
        # decoded_val = torch.fft.fft(decoded_val, dim=-1)
        # decoded_val = torch.fft.fftshift(decoded_val,dim=-1)
        
        # decoded_val_rearranged = rearrange(decoded_val.unsqueeze(0).unsqueeze(2), "d b i c k0 -> d c i b k0")
        # (us_rad_image_data,) = dataset.fourier_op.H(decoded_val_rearranged.to("cpu"))
        
        
        ## Input data
        val_data_rearranged = rearrange(val_data, "b (c r) k0 -> b c k0 r", c=10, r=2).contiguous()
        val_data_rearranged_real = torch.view_as_complex(val_data_rearranged)

        val_data_rearranged = torch.fft.fftshift(val_data_rearranged_real,dim=-1)
        val_data_rearranged = torch.fft.fft(val_data_rearranged, dim=-1)
        val_data_rearranged = torch.fft.fftshift(val_data_rearranged,dim=-1)
        val_data_rearranged = repeat(val_data_rearranged, "(b k1) c k0->b c k2 k1 k0",k2=1, k1=128)
        (grdth_image_data,) = dataset.fourier_op.H(val_data_rearranged.to("cpu") * dcf.data.to("cpu"))
        
        #ToDo: Add density compensation!
        
        #ToDo: add plot for ground-truth image
        gt_matfig = plt.figure(figsize=(8,8))
        plt.matshow(grdth_image_data.abs().detach().numpy()[0,0,0,...], fignum=gt_matfig.number)
        gt_sinogram = plt.figure(figsize=(8,8))
        plt.matshow(val_data_rearranged_real.cpu().square().abs().sum(1).sqrt().detach().numpy(), fignum=gt_sinogram.number)
        
        matfig = plt.figure(figsize=(8,8))
        plt.matshow(val_image_data.abs().detach().numpy()[0,0,0,...], fignum=matfig.number)
        net_sinogram = plt.figure(figsize=(8,8))
        plt.matshow(decoded_val_real.cpu().square().abs().sum(1).sqrt().detach().numpy(), fignum=net_sinogram.number)
        
        run["recons"].append(gt_matfig)
        run["recons"].append(gt_sinogram)
        run["recons"].append(matfig)
        run["recons"].append(net_sinogram)
        

print("Training finished!")
run.stop()
# %%
