"""
GAN (Generative Adversarial Network) for MNIST
From: week05-slides-batch1.md - Slide 4

Complete GAN with Generator and Discriminator, adversarial training.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Generator(nn.Module):
    """Generator network: noise → image"""
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """Discriminator network: image → real/fake"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img.view(-1, 784))


def train_gan(generator, discriminator, dataloader, epochs=50, latent_dim=100, device='cpu'):
    """Train GAN with alternating updates"""
    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)  # Want D to think they're real
            
            g_loss.backward()
            g_optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')
        
        # Save samples
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                z = torch.randn(64, latent_dim).to(device)
                samples = generator(z).cpu()
                
                fig, axes = plt.subplots(8, 8, figsize=(10, 10))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(samples[i].squeeze(), cmap='gray')
                    ax.axis('off')
                plt.suptitle(f'GAN Generated Samples - Epoch {epoch}', fontsize=14)
                plt.tight_layout()
                plt.savefig(f'gan_epoch_{epoch}.png', dpi=150)
                plt.close()
            generator.train()
    
    return generator, discriminator


if __name__ == "__main__":
    print("="*70)
    print("GAN TRAINING ON MNIST")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Models
    generator = Generator(latent_dim=100).to(device)
    discriminator = Discriminator().to(device)
    
    print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Train
    generator, discriminator = train_gan(generator, discriminator, dataloader, epochs=50, device=device)
    
    print("\n✓ GAN training complete! Check gan_epoch_*.png files")
