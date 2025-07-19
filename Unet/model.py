import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import optim, nn
from torchvision.transforms import v2 as tf
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchmetrics.segmentation import DiceScore
import os
from PIL import Image
from tqdm.auto import tqdm

class EarlyStop:
    def __init__(self, patience: int, tolerance: float, save_path: str):
        self.patience = patience
        self.tolerance = tolerance
        self.save_path = os.path.join(save_path, 'early_stopped_model.pth')
        self.count = 0
        self.best_loss = float('inf')
        self.stop = False

    def __call__(self, loss: float | torch.Tensor, model: torch.nn.Module):
        # Move the loss to CPU and convert it to a float if it's a tensor
        loss = loss.cpu().item() if isinstance(loss, torch.Tensor) else loss
        
        if self.best_loss - self.tolerance > loss:
            self.count = 0
            self.best_loss = loss
            torch.save(model.state_dict(), self.save_path)
        else:
            self.count += 1
            if self.count == self.patience:
                print(f"Early stopping...\nModel has been saved at {self.save_path}")
                self.stop = True

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, X):
        return self.conv_block(X)

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, in_channels * 2, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        skip = self.conv_block(X)
        out = self.pool(skip)
        return out, skip

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, in_channels // 2, **kwargs)

    def forward(self, X, skip):
        out = self.up_conv(X)
        out = torch.cat((skip, out), dim = 1)
        return self.conv_block(out)

class UNet(nn.Module):

    def __init__(self, in_channels: int, encoder_in_channel_list: list[int], decoder_in_channel_list: list[int], out_channels: int, **kwargs):
        super().__init__()

        # First convolution layer that takes the input of 3 channels from the original image
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, encoder_in_channel_list[0], **kwargs),
            nn.BatchNorm2d(encoder_in_channel_list[0]),
            nn.ReLU(),
            nn.Conv2d(encoder_in_channel_list[0], encoder_in_channel_list[0], **kwargs),
            nn.BatchNorm2d(encoder_in_channel_list[0]),
            nn.ReLU()
        )
        self.pool_after_in_conv = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.encoder = nn.ModuleList()
        for n_channels in encoder_in_channel_list: # 64, 128, 256
            self.encoder.append(EncoderBlock(n_channels, **kwargs))

        # Bottleneck
        self.bottleneck = ConvBlock(
            encoder_in_channel_list[-1] * 2,
            decoder_in_channel_list[0],
            **kwargs
        )

        # Decoder
        self.decoder = nn.ModuleList()
        for n_channels in decoder_in_channel_list: # 1024, 512, 256, 128
            self.decoder.append(DecoderBlock(n_channels, **kwargs))

        # Last convolution layer that output the predicted mask consisting n channels where n is
        # the number of classes [+ 1 (for background), for multiclass segmentation]
        self.out_conv = nn.Conv2d(decoder_in_channel_list[-1] // 2, out_channels, kernel_size=1)

    def forward(self, X):
        skip = []
        out = self.in_conv(X)
        skip.append(out)
        out = self.pool_after_in_conv(out)
        for encoder_block in self.encoder:
            out, s = encoder_block(out)
            skip.append(s)
        out = self.bottleneck(out)
        for decoder_block in self.decoder:
            s = skip.pop()
            out = decoder_block(out, s)
        out = self.out_conv(out)
        return out
    
def train(
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        accuracy_fn,
        dataloader: torch.utils.data.DataLoader
):
    # Put model to training mode
    model.train()

    # Aggregate loss and accuracy
    agg_loss, agg_acc = 0, 0

    # Iterate over the batches using the dataloader
    for X, y in dataloader:

        # Send to proper device
        X, y = X.to(device), y.to(device)
        y = y.to(torch.long) # Cross Entropy Loss requires target in long dtype
        y = y.squeeze(dim = 1) # Remove channel dimension [N, C, H, W] --> [N, C, W]

        # Reset optimizer
        optimizer.zero_grad()

        # Forward pass
        pred = model(X) # pred has shape [N, C, H, W]

        # Compute loss
        loss = loss_fn(pred, y)

        # Backpropagate loss
        loss.backward()

        # Update parameters
        optimizer.step()

        # Aggregate loss
        agg_loss += loss.item()

        # Convert pred to indexed format
        pred = pred.argmax(dim = 1) # [N, C, H, W] --> [N, H, W]

        # Aggregate accuracy
        agg_acc += accuracy_fn(pred, y).item()

        # Garbage collection
        del pred, loss, X, y

    num_batches = len(dataloader)
    return agg_loss / num_batches, agg_acc / num_batches

def validate(
        model: torch.nn.Module,
        device: torch.device,
        loss_fn,
        accuracy_fn,
        dataloader
):
    # Put model to evaluation model
    # Turns off drop-out and uses running stat
    # for batch normalization
    model.eval()

    # Aggregate loss and accuracy
    agg_loss, agg_accuracy = 0, 0

    # Use inference context manager
    with torch.inference_mode():

        # Iterate over the batches
        for X, y in dataloader:
            
            # Send to designated device
            X, y =X.to(device), y.to(device)
            y = y.to(torch.long)
            y = y.squeeze(dim = 1)

            # Forward pass
            pred = model(X)

            # Compute and aggregate loss
            loss = loss_fn(pred, y)
            agg_loss += loss.item()

            # Convert pred to indexed format
            # [N, C, H, W] --> [N, H, W]
            pred = pred.argmax(dim = 1)

            # Compute and aggregate accuracy
            agg_accuracy += accuracy_fn(pred, y).item()
            
            # Garbage collection
            del pred, loss, X, y

    num_batches = len(dataloader)
    return agg_loss / num_batches, agg_accuracy / num_batches

def run(
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_fn: callable,
        accuracy_fn: callable,
        train_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        early_stop: EarlyStop,
        epochs: int,
        vernose_after_every_n_epoch: int
):
    # Aggregate loss and accuracy fo every epoch
    t_loss, t_accuracy = [], []
    v_loss, v_accuracy = [], []

    for epoch in tqdm(range(1, epochs + 1)):

        # Free CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Train
        loss, accuracy = train(
            model,
            device,
            optimizer,
            loss_fn,
            accuracy_fn,
            train_dataloader
        )
        t_loss.append(loss)
        t_accuracy.append(accuracy)

        # Validate
        loss, accuracy = validate(
            model,
            device,
            loss_fn,
            accuracy_fn,
            validation_dataloader
        )
        v_loss.append(loss)
        v_accuracy.append(accuracy)

        # Verbose
        if epoch % vernose_after_every_n_epoch == 0:
            print(f"Epoch {epoch}\n----------")
            print (f"Train Loss {t_loss[-1]:.3f}\tVal Loss {v_loss[-1]:.3f}")
            print (f"Train Accuracy {t_accuracy[-1]:.3f}\tVal Accuracy {v_accuracy[-1]:.3f}\n\n")

        # Early Stop
        early_stop(loss, model)
        if early_stop.stop:
            break
        
        # Create and return a dataframe of metrics
    metrics = pd.DataFrame(
        {   
        'epoch' : list(range(1, len(t_loss) + 1)),
        'train_loss' : t_loss,
        'validation_loss' : v_loss,
        'train_accuracy' : t_accuracy,
        'validation_accuracy' : v_accuracy
        }
)
    return metrics