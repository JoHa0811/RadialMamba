#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import neptune
sys.path.append('/echo/hammac01/RadialMamba/data_loading')
sys.path.append('/echo/hammac01/RadialMamba/Autoencoder')
from dataloader import dataset
from autoencoder_net import Encoder, Decoder

#%%
# Logging to Neptune
run = neptune.init_run(
    project="MRI/RadialMamba",
    source_files=["**/*.py"],
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmU5Y2Q4MC04N2MxLTQxMzAtYWJmYS00YWFlOTcxNDg4MWIifQ==",
)  # your credentials
#%%
# Initialize net
encoder = Encoder(image_size=256, channels=20, embedding_dim=16)
decoder = Decoder()

valds, trainingds = torch.utils.data.random_split(dataset, [0.2, 0.8], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(trainingds, batch_size=4,
                        shuffle=True, num_workers=8)
                        #,collate_fn=lambda sampled: torch.concat(sampled, dim=0))
val_loader = DataLoader(valds, batch_size=4,
                        shuffle=True, num_workers=8)
                        #,collate_fn=lambda sampled: torch.concat(sampled, dim=0))
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
)

#%%
params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params
#%%
n_epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder.to(torch. device(device))
decoder.to(torch. device(device))
#%%
# initialize the best validation loss as infinity
best_val_loss = float("inf")
# start training by looping over the number of epochs
for epoch in range(n_epochs):
    print(f"Epoch: {epoch + 1}/{n_epochs}")
    # set the encoder and decoder models to training mode
    encoder.train()
    decoder.train()
    # initialize running loss as 0
    running_loss = 0.0
    # loop over the batches of the training dataset
    for batch_idx, data in tqdm(enumerate(trainingds), total=len(trainingds)):
        # move the data to the device (GPU or CPU)
        data = data.to(device)
        # reset the gradients of the optimizer
        optimizer.zero_grad()
        # forward pass: encode the data and decode the encoded representation
        encoded = encoder(data)
        decoded = decoder(encoded)
        # compute the reconstruction loss between the decoded output and
        # the original data
        loss = loss_function(decoded, data)
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
        
        encoder.eval()
        decoder.eval()
        
        for batch_idx, val_data in tqdm(enumerate(valds), total=len(valds)):
            
            val_data = val_data.to(device)
            
            print("Before net")
            
            encoded_val = encoder(val_data)
            decoded_val = decoder(encoded_val)
            
            print("After net")
            
            validation_loss = loss_function(decoded_val, val_data)
            
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
            {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
            f"/echo/hammac01/RadialMamba/Autoencoder/logged_weights/epoch_{epoch}.pt",
        )
        run["model_checkpoints/my_model"].upload(f"/echo/hammac01/RadialMamba/Autoencoder/logged_weights/epoch_{epoch}.pt")
        print(f"Saved epoch {epoch} as new best model.")
        
        decoded_val = rearrange(decoded_val, "b (c r) k0 -> b c k0 r", c=10, r=2).contiguous()
        decoded_val = torch.view_as_complex(decoded_val)
        #us_rad_kdata.shape
        #torch.Size([1, 10, 1, 64, 256])
        decoded_val_rearranged = rearrange(decoded_val.unsqueeze(0).unsqueeze(2), "d b i c k0 -> d c i b k0")
        (us_rad_image_data,) = dataset.fourier_op.H(decoded_val_rearranged.to("cpu"))
        
        val_data_rearranged = rearrange(val_data, "b (c r) k0 -> b c k0 r", c=10, r=2).contiguous()
        val_data_rearranged = torch.view_as_complex(val_data_rearranged)
        val_data_rearranged = rearrange(val_data_rearranged.unsqueeze(0).unsqueeze(2), "d b i c k0 -> d c i b k0")
        (val_image_data,) = dataset.fourier_op.H(val_data_rearranged.to("cpu"))
        
        #ToDo: add plot for ground-truth image
        gt_matfig = plt.figure(figsize=(8,8))
        plt.matshow(val_image_data.abs().detach().numpy()[0,0,0,...], fignum=gt_matfig.number)
        
        matfig = plt.figure(figsize=(8,8))
        plt.matshow(us_rad_image_data.abs().detach().numpy()[0,0,0,...], fignum=matfig.number)
        
        run["recons"].append(gt_matfig)
        run["recons"].append(matfig)
        

print("Training finished!")
run.stop()
# %%
