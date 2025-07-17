# iEdge 2025 REU
# Autoencoder flow and structure based on Vice et al. (https://github.com/jackvice/lstm_explore)

import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

class Autoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim=128):
        super().__init__()
        self.encoder_output_size = 2
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.encoder_output_size * self.encoder_output_size, latent_dim) # Calculuate input size for linear layer based on env
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * self.encoder_output_size * self.encoder_output_size), # Same as above
            nn.ReLU(),
            nn.Unflatten(1, (64, self.encoder_output_size, self.encoder_output_size)), # Reshape to match encoder output
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Assuming input is normalized to [0, 1]
        )    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def compute_ssim(img1_np, img2_np, data_range=1.0):
    return ssim(img1_np, img2_np, data_range=data_range, channel_axis=0)

