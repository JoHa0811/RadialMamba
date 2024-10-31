#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
sys.path.append('/echo/hammac01/RadialMamba/data_loading')
sys.path.append('/echo/hammac01/RadialMamba/Autoencoder')
from dataloader import dataloader
from autoencoder_net import Encoder, Decoder


#%%
# Initialize net
encoder = Encoder(image_size=256, channels=20, embedding_dim=16)
decoder = Decoder()
train_loader = dataloader
 
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-1
)
#%%
n_epochs = 10
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
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
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
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
    #val_loss = utils.validate(encoder, decoder, test_loader, criterion)
    val_loss = 0
    # print training and validation loss for current epoch
    print(
        f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f}"
    )
    # save best model weights based on validation loss
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(
    #         {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
    #         config.MODEL_WEIGHTS_PATH,
    #     )
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
