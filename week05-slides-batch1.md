# Week 5: Image Generation, Audio, and Music - Slides Batch 1 (Slides 1-10)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Date:** February 17, 2026  
**Duration:** 2.5 hours

---

## Slide 1: Week 5 Title Slide

### Image Generation, Audio, and Music with GenAI

**Today's Focus:**
- Understanding generative models beyond text
- VAEs, GANs, and Diffusion Models
- Creating images from text prompts
- Audio and music generation
- Multimodal AI applications
- Business use cases in creative industries

**Prerequisites:**
- Week 4: Deep Learning and Transformers
- Understanding of neural networks
- Basic PyTorch/TensorFlow knowledge

---

## Slide 2: Today's Agenda

### Class Overview

1. **Image Generation Fundamentals** (30 min)
2. **VAEs and GANs** (30 min)
3. **Break** (10 min)
4. **Diffusion Models** (35 min)
5. **Audio and Music Generation** (25 min)
6. **Business Applications** (20 min)
7. **Hands-on Lab & Q&A** (20 min)

---

## Slide 3: Learning Objectives

### By the End of This Class, You Will:

✅ **Understand** how generative models create images and audio  
✅ **Explain** VAEs, GANs, and Diffusion Models  
✅ **Implement** basic image generation systems  
✅ **Recognize** audio generation techniques  
✅ **Apply** these technologies to business problems  
✅ **Evaluate** ROI of creative GenAI applications

---

## Slide 4: The Evolution of Image Generation

### From Rules to Neural Networks

**Historical Timeline:**

**1. Rule-Based Graphics (1960s-1990s)**
- Computer graphics with manual programming
- 3D rendering engines
- Procedural generation
- Limited creativity, deterministic

**2. Style Transfer (2015)**
- Neural Style Transfer using CNNs
- Combine content of one image with style of another
- First neural network "art"
- Example: Photo in Van Gogh's style

**3. GANs (2014-2020)**
- Generate realistic faces, objects, scenes
- Progressive improvements (StyleGAN)
- High-resolution synthesis
- Limited control over output

**4. Diffusion Models (2020-Present)**
- DALL-E, Stable Diffusion, Midjourney
- Text-to-image generation
- Precise control via prompts
- Revolutionary creative tool

**Business Impact:**
- Design automation
- Content creation at scale
- Personalization
- Cost reduction: $50-200 per image → $0.01

---

## Slide 5: Variational Autoencoders (VAEs)

### Learning Compressed Representations

**What is a VAE?**

A Variational Autoencoder learns to:
1. **Encode** images into a compressed latent space
2. **Sample** from that latent space
3. **Decode** samples back into images

**Architecture:**
```
Image → Encoder → Latent Space (μ, σ) → Decoder → Reconstructed Image
                      ↓
                   Sample z ~ N(μ, σ)
                      ↓
                   Decoder → New Image
```

**Key Insight:** The latent space is **continuous and smooth**, meaning:
- Similar images cluster together
- Interpolation between images is meaningful
- We can sample to generate new images

**Complete VAE Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class VAE(nn.Module):
    """
    Variational Autoencoder for image generation.
    
    Architecture:
        Encoder: Image → Latent distribution (μ, log_var)
        Decoder: Latent sample → Reconstructed image
    
    Loss:
        Reconstruction loss + KL divergence
    """
    
    def __init__(self, latent_dim=20, input_channels=1, hidden_dim=400):
        """
        Args:
            latent_dim: Dimension of latent space
            input_channels: 1 for grayscale, 3 for RGB
            hidden_dim: Hidden layer dimension
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: Image → Hidden → Latent parameters
        self.fc1 = nn.Linear(input_channels * 28 * 28, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Latent → Hidden → Image
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_channels * 28 * 28)
    
    def encode(self, x):
        """
        Encode image to latent distribution parameters.
        
        Args:
            x: Input images (batch, channels, height, width)
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Flatten image
        x = x.view(-1, 28 * 28)
        
        # Encoder forward pass
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
        
        This allows backpropagation through sampling.
        
        Args:
            mu: Mean
            logvar: Log variance
        
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vector
        
        Returns:
            Reconstructed image
        """
        h = F.relu(self.fc3(z))
        x_reconstructed = torch.sigmoid(self.fc4(h))
        return x_reconstructed.view(-1, 1, 28, 28)
    
    def forward(self, x):
        """Full forward pass: encode, sample, decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar


def vae_loss(x_reconstructed, x, mu, logvar):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Reconstruction loss: How well we reconstruct input
    KL divergence: How close latent distribution is to N(0,1)
    
    Args:
        x_reconstructed: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
    
    Returns:
        Total loss
    """
    # Reconstruction loss (binary cross-entropy)
    BCE = F.binary_cross_entropy(
        x_reconstructed.view(-1, 28*28), 
        x.view(-1, 28*28), 
        reduction='sum'
    )
    
    # KL divergence: KL(N(μ,σ²) || N(0,1))
    # = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD


def train_vae(model, train_loader, optimizer, epoch, device):
    """Train VAE for one epoch"""
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        x_reconstructed, mu, logvar = model(data)
        loss = vae_loss(x_reconstructed, data, mu, logvar)
        
        # Backward pass
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item() / len(data):.4f}')
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss


# Complete training example
if __name__ == "__main__":
    print("="*70)
    print("VARIATIONAL AUTOENCODER (VAE) TRAINING")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load MNIST
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Create model
    latent_dim = 20
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\nModel Configuration:")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    n_epochs = 10
    losses = []
    
    for epoch in range(1, n_epochs + 1):
        loss = train_vae(model, train_loader, optimizer, epoch, device)
        losses.append(loss)
    
    # Generate new images
    print("\n" + "="*70)
    print("GENERATING NEW IMAGES")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        # Sample from standard normal
        z = torch.randn(64, latent_dim).to(device)
        samples = model.decode(z).cpu()
        
        # Visualize
        fig, axes = plt.subplots(8, 8, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i].squeeze(), cmap='gray')
            ax.axis('off')
        
        plt.suptitle('Generated Images from VAE', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('vae_generated_images.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved generated images to 'vae_generated_images.png'")
        plt.show()
    
    # Visualize latent space
    print("\nInterpolating in latent space...")
    with torch.no_grad():
        # Two random points
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)
        
        # Interpolate
        fig, axes = plt.subplots(1, 10, figsize=(20, 2))
        for i, alpha in enumerate(np.linspace(0, 1, 10)):
            z = (1 - alpha) * z1 + alpha * z2
            img = model.decode(z).cpu()
            axes[i].imshow(img.squeeze(), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'{alpha:.1f}')
        
        plt.suptitle('Latent Space Interpolation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('vae_interpolation.png', dpi=300, bbox_inches='tight')
        print("✓ Saved interpolation to 'vae_interpolation.png'")
        plt.show()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final loss: {losses[-1]:.4f}")
    print("Model can now generate new images by sampling from latent space!")
```

**Key Advantages:**
- ✅ Smooth latent space (good for interpolation)
- ✅ Probabilistic framework (quantifies uncertainty)
- ✅ Easy to train (stable)

**Limitations:**
- ❌ Blurry outputs (due to reconstruction loss)
- ❌ Less realistic than GANs
- ❌ Limited diversity

**Business Applications:**
- Product design variations
- Data augmentation
- Anomaly detection
- Dimensionality reduction

---

## Slide 6: Generative Adversarial Networks (GANs)

### A Game Between Generator and Discriminator

**What is a GAN?**

Two neural networks compete in a game:

**Generator (G):**
- Creates fake images from random noise
- Goal: Fool the discriminator
- "Counterfeiter making fake money"

**Discriminator (D):**
- Distinguishes real from fake images
- Goal: Detect generator's fakes
- "Police detecting counterfeit money"

**Training Process:**
```
1. Generator creates fake images
2. Discriminator sees real + fake images
3. Discriminator learns to tell them apart
4. Generator learns to fool discriminator better
5. Repeat until generator produces realistic images
```

**Mathematical Framework:**
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]

Where:
- D(x): Discriminator's probability that x is real
- G(z): Generator's output from noise z
- Real images maximize D(x)
- Fake images minimize D(G(z))
```

**Complete GAN Implementation:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class Generator(nn.Module):
    """
    Generator network: Noise → Image
    
    Takes random noise and generates realistic images.
    Uses transposed convolutions to upsample.
    """
    
    def __init__(self, latent_dim=100, img_channels=1, feature_dim=64):
        """
        Args:
            latent_dim: Dimension of input noise vector
            img_channels: Number of image channels (1 for grayscale, 3 for RGB)
            feature_dim: Base feature map size
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Network: Noise (100) → 7x7x256 → 14x14x128 → 28x28x1
        self.model = nn.Sequential(
            # Input: (batch, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, feature_dim * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            # State: (batch, feature_dim*4, 7, 7)
            
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            # State: (batch, feature_dim*2, 14, 14)
            
            nn.ConvTranspose2d(feature_dim * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: (batch, img_channels, 28, 28), values in [-1, 1]
        )
    
    def forward(self, z):
        """Generate image from noise"""
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    """
    Discriminator network: Image → Real/Fake probability
    
    Classifies images as real or fake.
    Uses standard convolutions to downsample.
    """
    
    def __init__(self, img_channels=1, feature_dim=64):
        """
        Args:
            img_channels: Number of image channels
            feature_dim: Base feature map size
        """
        super(Discriminator, self).__init__()
        
        # Network: 28x28x1 → 14x14x64 → 7x7x128 → 1
        self.model = nn.Sequential(
            # Input: (batch, img_channels, 28, 28)
            nn.Conv2d(img_channels, feature_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch, feature_dim, 14, 14)
            
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch, feature_dim*2, 7, 7)
            
            nn.Conv2d(feature_dim * 2, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: (batch, 1, 1, 1) → probability
        )
    
    def forward(self, img):
        """Classify image as real or fake"""
        validity = self.model(img)
        return validity.view(-1, 1)


def train_gan(generator, discriminator, dataloader, n_epochs, latent_dim, device):
    """
    Train GAN using alternating optimization.
    
    Training loop:
        1. Train Discriminator on real + fake images
        2. Train Generator to fool Discriminator
    """
    
    # Loss function: Binary Cross-Entropy
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Training history
    d_losses = []
    g_losses = []
    
    print("\n" + "="*70)
    print("GAN TRAINING")
    print("="*70)
    
    for epoch in range(n_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Labels for real and fake
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real images
            real_validity = discriminator(real_imgs)
            d_real_loss = criterion(real_validity, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_validity, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate images and try to fool discriminator
            fake_validity = discriminator(fake_imgs)
            g_loss = criterion(fake_validity, real_labels)  # Want discriminator to think they're real
            
            g_loss.backward()
            optimizer_G.step()
            
            # Logging
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{n_epochs}] Batch [{i}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        
        # Generate samples every epoch
        if epoch % 5 == 0:
            with torch.no_grad():
                z = torch.randn(16, latent_dim, 1, 1).to(device)
                gen_imgs = generator(z).cpu()
                
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for idx, ax in enumerate(axes.flat):
                    ax.imshow(gen_imgs[idx].squeeze(), cmap='gray')
                    ax.axis('off')
                plt.suptitle(f'Generated Images - Epoch {epoch}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'gan_epoch_{epoch}.png', dpi=150)
                plt.close()
    
    return d_losses, g_losses


# Complete example
if __name__ == "__main__":
    print("="*70)
    print("GENERATIVE ADVERSARIAL NETWORK (GAN) TRAINING")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Hyperparameters
    latent_dim = 100
    batch_size = 128
    n_epochs = 50
    
    # Load data
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create models
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Train
    d_losses, g_losses = train_gan(
        generator, discriminator, dataloader, 
        n_epochs, latent_dim, device
    )
    
    # Visualize training
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gan_training_losses.png', dpi=300)
    plt.show()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("✓ Generator can now create realistic images from random noise!")
```

**Key Advantages:**
- ✅ Sharp, realistic images
- ✅ High quality outputs
- ✅ Versatile (images, video, audio)

**Limitations:**
- ❌ Training instability (mode collapse)
- ❌ Difficult to converge
- ❌ Limited control over outputs

**Famous GANs:**
- StyleGAN (faces)
- BigGAN (ImageNet)
- Pix2Pix (image-to-image)
- CycleGAN (unpaired translation)

---

## Slide 7: GAN Applications

### Real-World Business Use Cases

**1. Face Generation (StyleGAN)**
- Create realistic human faces
- Adjust age, gender, expression
- Use: Avatars, game characters, privacy

**2. Image-to-Image Translation**
- Pix2Pix: Sketch → Photo
- CycleGAN: Summer → Winter, Horse → Zebra
- Use: Design automation, video effects

**3. Super-Resolution**
- Enhance low-resolution images
- Restore old photos
- Use: Medical imaging, surveillance

**4. Deepfakes (Ethical Concerns)**
- Face swapping in videos
- Voice cloning
- Use: Entertainment, BUT major ethical issues

**Business ROI Example:**

```python
# Fashion Industry Use Case
traditional_photoshoot = {
    'model_fee': 2000,
    'photographer': 1500,
    'studio_rental': 1000,
    'post_processing': 500,
    'total_per_outfit': 5000,
    'time': '1 day'
}

gan_approach = {
    'model_3d_scan': 500,  # One-time
    'gan_generation': 10,  # Per outfit
    'retouching': 100,
    'total_per_outfit': 110,
    'time': '1 hour'
}

savings_per_outfit = traditional_photoshoot['total_per_outfit'] - gan_approach['total_per_outfit']
print(f"Savings: ${savings_per_outfit} per outfit (98% cost reduction)")
print(f"For 100 outfits: ${savings_per_outfit * 100:,} saved")
```

---

## Slide 8: Mode Collapse in GANs

### A Common Training Problem

**What is Mode Collapse?**

Generator learns to produce only a few types of outputs (modes) instead of the full diversity of the data distribution.

**Example:**
- Training on faces dataset with thousands of unique people
- Generator only produces 10-20 different faces
- High quality but low diversity

**Why It Happens:**
1. Generator finds a few outputs that fool discriminator
2. Keeps generating those "winning" outputs
3. Doesn't explore other possibilities
4. Gets stuck in local optimum

**Visual Example:**
```
Desired:  😀 😃 😄 😁 😆 😅 😂 🤣 😊 😇 ... (1000s of variations)
Reality:  😀 😀 😀 😃 😃 😃 😄 😄 😄  (only 3 variations)
```

**Solutions:**

**1. Minibatch Discrimination**
- Discriminator compares images within a batch
- Penalizes if all images are similar
- Encourages diversity

**2. Unrolled GANs**
- Generator looks ahead several discriminator update steps
- Prevents short-term exploitation

**3. Wasserstein GAN (WGAN)**
- Different loss function (Earth Mover's Distance)
- More stable training
- Meaningful loss curves

**4. Conditional GANs**
- Add class labels as input
- Force generator to produce specific types
- Better control and diversity

**Business Impact:**
- Mode collapse → Limited usefulness
- Stable training → Reliable deployment
- Diversity → Better creative applications

---

## Slide 9: Conditional GANs (cGANs)

### Adding Control to Generation

**Problem with Standard GANs:**
- Can't control what gets generated
- Random noise → random output
- No way to specify "generate a 7" or "make it blue"

**Solution: Conditional GANs**

Add additional information (labels, attributes) to both generator and discriminator:

**Architecture:**
```
Generator:  (Noise + Label) → Image
Discriminator: (Image + Label) → Real/Fake
```

**Implementation Example:**

```python
class ConditionalGenerator(nn.Module):
    """Generator that takes class label as additional input"""
    def __init__(self, latent_dim=100, n_classes=10, img_channels=1):
        super().__init__()
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(n_classes, latent_dim)
        
        # Generator (now takes latent_dim * 2)
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Concatenate noise and embedded labels
        label_embedding = self.label_emb(labels)
        combined_input = torch.cat([noise, label_embedding], dim=1)
        
        # Generate image
        img = self.model(combined_input)
        img = img.view(-1, 1, 28, 28)
        return img


# Usage: Generate specific digit
generator = ConditionalGenerator()
noise = torch.randn(1, 100)
label = torch.tensor([7])  # Generate a "7"
generated_seven = generator(noise, label)
```

**Applications:**

**1. Text-to-Image (like DALL-E)**
- Text prompt as condition
- Generate images matching description
- Revolutionary for creative industries

**2. Image-to-Image Translation**
- Pix2Pix: Sketch + "make realistic" → Photo
- Edges + "add colors" → Colored image

**3. Style Transfer**
- Content image + Style reference → Stylized image
- Preserve content, change artistic style

**4. Face Editing**
- Face + Attributes (add glasses, smile) → Modified face
- Precise control over generation

**Business Use Case:**
```
E-commerce Product Images:
- Input: Product sketch + "professional studio lighting"
- Output: High-quality product photo
- Saves: $100-500 per product photoshoot
- ROI: 95% cost reduction for 1000+ products
```

---

## Slide 10: Diffusion Models Introduction

### The New State-of-the-Art

**What Changed Everything: Diffusion Models (2020+)**

Diffusion models are now the leading approach for image generation, powering:
- DALL-E 2 & 3
- Stable Diffusion
- Midjourney
- Imagen (Google)

**The Big Idea:**

**Forward Process (Adding Noise):**
```
Clean Image → Add noise → Add more noise → ... → Pure noise
    ↓            ↓              ↓                      ↓
  Step 0      Step 1        Step 2              Step 1000
```

**Reverse Process (Denoising):**
```
Pure noise → Denoise → Denoise more → ... → Clean Image
    ↓          ↓            ↓                    ↓
 Step 1000  Step 999    Step 998            Step 0
```

**Key Insight:** If we learn to reverse the noise-adding process, we can generate images from pure noise!

**Why Diffusion Models Won:**

✅ **Better Quality:** More realistic than GANs
✅ **More Stable:** Easier to train than GANs
✅ **Better Control:** Text conditioning works excellently
✅ **Scalable:** Works well with massive models
✅ **Diverse:** No mode collapse issues

**The Numbers:**
- DALL-E 2: 3.5B parameters
- Stable Diffusion: 860M parameters
- Training cost: $50M+ for DALL-E 2
- Business impact: $10B+ market by 2028

**Simple Analogy:**

Think of it like restoring an old, damaged photo:
1. Start with extremely noisy image (like static on TV)
2. AI gradually removes noise, revealing structure
3. Each step makes image slightly clearer
4. After 1000 steps: Perfect, new image

**Coming up:** Full implementation in Slide 11-13!

---

**End of Batch 1 (Slides 1-10)**

*Continue to Batch 2 for Diffusion Model Implementation and Audio Generation*
