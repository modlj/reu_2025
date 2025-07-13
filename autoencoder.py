import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

class Autoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * ?, latent_dim) # Calculuate input size for linear layer based on env
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * ?), # Same as above
            nn.ReLU(),
            nn.Unflatten(1, (64, ?, ?)), # Reshape to match encoder output
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
    return ssim(img1_np, img2_np, data_range=data_range, channel_axis=0 if img1_np.shape[0] == 3 else None)

