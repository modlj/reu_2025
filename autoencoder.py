# John Modl, iEdge 2025 REU
# Autoencoder flow and structure based on Vice et al. (https://github.com/jackvice/lstm_explore)

import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

class Autoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim=128, input_height=7, input_width=7):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Dynamically calculate the output size of the encoder's convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_output_shape = self.encoder(dummy_input).shape
            self.conv_output_channels = conv_output_shape[1]
            self.encoder_output_height = conv_output_shape[2]
            self.encoder_output_width = conv_output_shape[3]
            self.flattened_size = self.conv_output_channels * self.encoder_output_height * self.encoder_output_width

        self.encoder_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, latent_dim)
        )

        

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_size),
            nn.ReLU(),
            nn.Unflatten(1, (self.conv_output_channels, self.encoder_output_height, self.encoder_output_width)),
            nn.ConvTranspose2d(self.conv_output_channels, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Output 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=0), # Output 7x7
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder_linear(self.encoder(x)) # Chain encoder and encoder_linear
        reconstructed = self.decoder(latent)
        return reconstructed

def compute_ssim(img1_np, img2_np, data_range=1.0):
    # Ensure both images have the same dimensions
    # skimage ssim expects float values
    img1_np = img1_np.astype(float)
    img2_np = img2_np.astype(float)
    return ssim(img1_np, img2_np, data_range=data_range, channel_axis=2)