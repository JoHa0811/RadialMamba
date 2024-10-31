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
sys.path.append('/echo/hammac01/RadialMamba/data_loading')
sys.path.append('/echo/hammac01/RadialMamba/Autoencoder')
from dataloader import dataset
from autoencoder_net import Encoder, Decoder


#%%
# Initialize net
encoder = Encoder(image_size=256, channels=20, embedding_dim=16)
decoder = Decoder()

valds, trainingds = torch.utils.data.random_split(dataset, [0.2, 0.8], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(trainingds, batch_size=4,
                        shuffle=True, num_workers=0,
                        collate_fn=lambda sampled: torch.concat(sampled, dim=0))
val_loader = DataLoader(valds, batch_size=4,
                        shuffle=True, num_workers=0,
                        collate_fn=lambda sampled: torch.concat(sampled, dim=0))
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-1
)
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
        
        # compute the average training loss for the epoch
    train_loss = running_loss / len(train_loader)
    # compute the validation loss

    running_val_loss = 0.0
    
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        
        for batch_idx, val_data in tqdm(enumerate(val_loader), total=len(val_loader)):
            
            val_data = val_data.to(device)
            
            encoded_val = encoder(val_data)
            decoded_val = decoder(encoded_val)
            
            validation_loss = loss_function(decoded_val, val_data)
            
            running_val_loss += validation_loss.item()
    
    val_loss = running_val_loss / len(val_loader)
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
        print(f"Saved epoch {epoch} as new best model.")
    # # adjust learning rate based on the validation loss
    # scheduler.step(val_loss)
    # # save validation output reconstruction for the current epoch
    # utils.display_random_images(
    #     data_loader=test_loader,
    #     encoder=encoder,
    #     decoder=decoder,
    #     file_recon=os.path.join(
    #         config.training_progress_dir, f"epoch{epoch + 1}_test_recon.png"
    #     ),
    #     display_real=False,
    # )
print("Training finished!")
# %%
