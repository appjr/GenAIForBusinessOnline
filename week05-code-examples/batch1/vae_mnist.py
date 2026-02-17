"""
VAE (Variational Autoencoder) for MNIST
From: week05-slides-batch1.md - Slide 2

Complete implementation of VAE with encoder, decoder, and reparameterization trick.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class VAE(nn.Module):
    """Variational Autoencoder for MNIST"""
    
    def __init__(self, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_logvar = nn.Linear(200, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to image"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = Reconstruction + KL divergence"""
    # Reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD


def train_vae(model, train_loader, epochs=10, lr=1e-3, device='cpu'):
    """Train VAE"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item()/len(data):.4f}')
        
        print(f'====> Epoch {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    
    return model


def visualize_results(model, test_loader, device='cpu'):
    """Visualize reconstructions and generations"""
    model.eval()
    with torch.no_grad():
        # Get test batch
        data, _ = next(iter(test_loader))
        data = data.to(device)
        
        # Reconstruct
        recon, _, _ = model(data)
        
        # Plot
        n = 8
        fig, axes = plt.subplots(2, n, figsize=(12, 3))
        for i in range(n):
            # Original
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Reconstruction
            axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('vae_reconstruction.png', dpi=150)
        print("✓ Saved reconstruction to vae_reconstruction.png")
        
        # Generate new samples
        fig, axes = plt.subplots(4, 8, figsize=(12, 6))
        z = torch.randn(32, 20).to(device)
        samples = model.decode(z)
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i].cpu().view(28, 28), cmap='gray')
            ax.axis('off')
        
        plt.suptitle('Generated Samples from VAE', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('vae_generated.png', dpi=150)
        print("✓ Saved generated samples to vae_generated.png")
        
        plt.show()


if __name__ == "__main__":
    print("="*70)
    print("VAE TRAINING ON MNIST")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = VAE(latent_dim=20).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model = train_vae(model, train_loader, epochs=10, device=device)
    
    # Visualize
    visualize_results(model, test_loader, device)
    
    print("\n✓ VAE training complete!")
    print("  - Check vae_reconstruction.png for reconstructions")
    print("  - Check vae_generated.png for generated samples")
