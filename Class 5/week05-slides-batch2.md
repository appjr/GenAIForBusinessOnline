# Week 5: Image Generation, Audio, and Music - Slides Batch 2 (Slides 11-15)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Continuation from Batch 1**

---

## Slide 11: Diffusion Models - The Math

### Understanding the Diffusion Process

**Forward Diffusion (Adding Noise):**

At each timestep t, we add a small amount of Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) * x_{t-1}, β_t * I)

Where:
- x_t: Image at timestep t
- β_t: Noise schedule (how much noise to add)
- Starts with clean image (t=0)
- Ends with pure noise (t=T, typically T=1000)
```

**Key Property:** After enough steps, image becomes pure Gaussian noise

**Reverse Diffusion (Denoising):**

Learn to reverse the process:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

Where:
- μ_θ: Predicted mean (what neural network learns)
- Σ_θ: Predicted variance
- θ: Neural network parameters
```

**Training Objective:**

Learn to predict the noise that was added:

```
L = E_{t, x_0, ε} [||ε - ε_θ(x_t, t)||²]

Where:
- ε: True noise added
- ε_θ: Predicted noise from network
- Simply predict what noise was added!
```

**Why This Works:**

1. **Simple objective:** Just predict noise
2. **Gradual refinement:** 1000 small steps vs 1 big step
3. **Stable training:** No adversarial dynamics like GANs
4. **High quality:** Time to refine details

**Intuitive Analogy:**

Imagine sculpting:
- Start with rough block (noisy image)
- Gradually refine with small touches (denoising steps)
- Each step reveals more detail
- Final result: Detailed sculpture (clean image)

---

## Slide 12: Implementing a Simple Diffusion Model

### DDPM (Denoising Diffusion Probabilistic Models)

**Complete Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleDiffusion:
    """
    Simple Diffusion Model for image generation.
    
    Implements DDPM (Denoising Diffusion Probabilistic Models).
    
    Reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    """
    
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        """
        Args:
            timesteps: Number of diffusion steps (T)
            beta_start: Initial noise level
            beta_end: Final noise level
            device: 'cpu' or 'cuda'
        """
        self.timesteps = timesteps
        self.device = device
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: Add noise to image.
        
        q(x_t | x_0) = N(x_t; √α̅_t * x_0, (1 - α̅_t) * I)
        
        Args:
            x_start: Original image
            t: Timestep
            noise: Optional pre-generated noise
        
        Returns:
            Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_start, t, noise=None):
        """
        Training loss: MSE between true noise and predicted noise.
        
        Args:
            denoise_model: Neural network that predicts noise
            x_start: Original image
            t: Timestep
            noise: Optional pre-generated noise
        
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise to image
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = denoise_model(x_noisy, t)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, denoise_model, x, t):
        """
        Reverse diffusion: Remove noise from image (single step).
        
        Args:
            denoise_model: Trained denoising model
            x: Noisy image at timestep t
            t: Current timestep
        
        Returns:
            Less noisy image at timestep t-1
        """
        # Predict noise
        predicted_noise = denoise_model(x, t)
        
        # Calculate x_{t-1}
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1, 1, 1)
        
        # Mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, denoise_model, shape):
        """
        Generate image by iteratively denoising.
        
        Start from pure noise, denoise for T steps.
        
        Args:
            denoise_model: Trained denoising model
            shape: Shape of image to generate (batch, channels, height, width)
        
        Returns:
            Generated images
        """
        device = next(denoise_model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # Progressively denoise
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(denoise_model, img, t)
        
        return img


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for denoising.
    
    Takes noisy image and timestep, predicts noise.
    """
    
    def __init__(self, channels=1, time_emb_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Downsampling
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        
        # Output
        self.conv_out = nn.Conv2d(64, channels, 3, padding=1)
        
    def forward(self, x, t):
        # Time embedding
        t = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_mlp(t)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        
        # Encoder
        x1 = F.relu(self.conv1(x))
        x2 = self.pool(x1)
        x2 = F.relu(self.conv2(x2))
        x3 = self.pool(x2)
        
        # Bottleneck
        x3 = F.relu(self.conv3(x3))
        
        # Decoder
        x = self.up(x3)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.conv4(x))
        
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.conv5(x))
        
        return self.conv_out(x)


def train_diffusion(model, diffusion, dataloader, epochs, device):
    """Train diffusion model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    
    print("="*70)
    print("TRAINING DIFFUSION MODEL")
    print("="*70)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # Calculate loss
            loss = diffusion.p_losses(model, images, t)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Generate samples every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                samples = diffusion.p_sample_loop(model, shape=(16, 1, 28, 28))
                samples = samples.cpu()
                
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(samples[i].squeeze(), cmap='gray')
                    ax.axis('off')
                plt.suptitle(f'Generated Images - Epoch {epoch}')
                plt.savefig(f'diffusion_epoch_{epoch}.png', dpi=150)
                plt.close()
            model.train()
    
    return model


# Usage example
if __name__ == "__main__":
    print("="*70)
    print("DIFFUSION MODEL TRAINING ON MNIST")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Create model and diffusion
    model = SimpleUNet().to(device)
    diffusion = SimpleDiffusion(timesteps=1000, device=device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Diffusion timesteps: {diffusion.timesteps}")
    
    # Train
    model = train_diffusion(model, diffusion, dataloader, epochs=10, device=device)
    
    # Generate new images
    print("\n" + "="*70)
    print("GENERATING NEW IMAGES")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        samples = diffusion.p_sample_loop(model, shape=(64, 1, 28, 28))
        samples = samples.cpu()
        
        fig, axes = plt.subplots(8, 8, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i].squeeze(), cmap='gray')
            ax.axis('off')
        plt.suptitle('Final Generated Images from Diffusion Model', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('diffusion_final_samples.png', dpi=300)
        plt.show()
    
    print("\n✓ Training complete! Model can generate images from pure noise.")
```

**Key Points:**

1. **Forward process:** Gradually add noise (deterministic)
2. **Reverse process:** Learn to remove noise (neural network)
3. **Training:** Predict the noise that was added
4. **Generation:** Start with noise, denoise iteratively

**Advantages over GANs:**
- ✅ Stable training (no adversarial dynamics)
- ✅ Better quality outputs
- ✅ No mode collapse
- ✅ Meaningful training curves

---

## Slide 13: Text-to-Image with Stable Diffusion

### Adding Text Conditioning

**The Revolution:** Combine diffusion with text embeddings!

**Architecture:**

```
Text Prompt → Text Encoder (CLIP) → Text Embeddings
                                           ↓
Pure Noise → U-Net (conditioned on text) → Denoised Image
                    ↑                          ↓
                Timestep                   Repeat 50x
```

**Key Components:**

**1. Text Encoder (CLIP)**
- Converts text to embeddings
- Pre-trained on 400M image-text pairs
- Captures semantic meaning

**2. Latent Diffusion**
- Work in compressed latent space (8x8x4 instead of 512x512x3)
- 64x faster than pixel-space diffusion
- Same quality, much more efficient

**3. Cross-Attention**
- U-Net attends to text embeddings
- Guides generation based on prompt
- Different layers attend to different concepts

**Stable Diffusion Pipeline:**

```python
from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt, negative_prompt="", num_steps=50):
    """
    Generate image from text using Stable Diffusion.
    
    Args:
        prompt: Text description
        negative_prompt: What to avoid
        num_steps: Number of denoising steps
    
    Returns:
        Generated PIL image
    """
    # Load model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    # Generate
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=7.5  # How strictly to follow prompt
    ).images[0]
    
    return image


# Example usage
prompt = "A beautiful sunset over mountains, oil painting style, highly detailed"
negative_prompt = "blurry, low quality, distorted"

image = generate_image(prompt, negative_prompt)
image.save("generated_sunset.png")
print("✓ Image generated successfully!")
```

**Prompt Engineering Tips:**

**Good Prompts:**
- "A photorealistic portrait of a cat wearing a top hat, studio lighting, 4k"
- "Cyberpunk city at night, neon lights, raining, cinematic, highly detailed"
- "Watercolor painting of a peaceful garden, soft colors, artistic"

**Bad Prompts:**
- "cat" (too vague)
- "nice picture" (not descriptive)
- "something cool" (no specifics)

**Modifiers that work:**
- Style: photorealistic, oil painting, watercolor, digital art
- Quality: highly detailed, 4k, masterpiece, trending on artstation
- Lighting: studio lighting, golden hour, dramatic lighting
- Camera: wide angle, close-up, aerial view

---

## Slide 14: Advanced Diffusion Techniques

### Making Diffusion Faster and Better

**Problem:** Standard diffusion requires ~1000 steps (slow!)

**Solutions:**

**1. DDIM (Denoising Diffusion Implicit Models)**
- Deterministic sampling (same noise → same image)
- Fewer steps: 50 instead of 1000 (20x faster!)
- Nearly same quality

```python
# DDIM sampling
def ddim_sample(model, x_t, t, t_prev):
    """Single DDIM step (deterministic)"""
    # Predict noise
    eps = model(x_t, t)
    
    # Predict x_0
    alpha_t = alphas_cumprod[t]
    x_0_pred = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
    
    # Direction pointing to x_t
    alpha_t_prev = alphas_cumprod[t_prev]
    direction = sqrt(1 - alpha_t_prev) * eps
    
    # x_{t-1}
    x_t_prev = sqrt(alpha_t_prev) * x_0_pred + direction
    
    return x_t_prev

# Use fewer steps: [0, 20, 40, 60, ..., 980, 1000]
timesteps = torch.linspace(1000, 0, 50).long()
```

**2. Latent Diffusion (Stable Diffusion)**
- Compress image to latent space first (using VAE)
- Run diffusion in latent space (much smaller)
- Decode back to image
- **64x faster** than pixel-space

```
High-res Image (512x512x3) → VAE Encode → Latent (64x64x4)
                                                ↓
                                          Diffusion here!
                                                ↓
                                 VAE Decode ← Denoised Latent
```

**3. Classifier-Free Guidance**
- Stronger control over text conditioning
- Mix conditional and unconditional predictions

```python
# Guidance scale = 7.5 (typical)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

**Speed Comparisons:**

| Method | Steps | Time | Quality |
|--------|-------|------|---------|
| DDPM | 1000 | 60s | ⭐⭐⭐⭐⭐ |
| DDIM | 50 | 3s | ⭐⭐⭐⭐⭐ |
| Latent Diffusion | 50 | 2s | ⭐⭐⭐⭐⭐ |
| Latent + Distillation | 4 | 0.5s | ⭐⭐⭐⭐ |

**Business Impact:**
- Faster = More images per dollar
- Real-time applications become possible
- Reduced infrastructure costs

---

## Slide 15: Image Editing with Diffusion

### Beyond Generation: Editing Existing Images

**Powerful Applications:**

**1. Inpainting (Fill in Missing Parts)**

```python
from diffusers import StableDiffusionInpaintPipeline

def inpaint_image(image, mask, prompt):
    """
    Fill in masked region based on prompt.
    
    Args:
        image: Original PIL image
        mask: Binary mask (white = fill in)
        prompt: What to generate in masked area
    
    Returns:
        Edited image
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting"
    ).to("cuda")
    
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask
    ).images[0]
    
    return result

# Example: Remove person from photo
prompt = "empty park bench, natural lighting"
edited = inpaint_image(photo, person_mask, prompt)
```

**Use Cases:**
- Remove unwanted objects
- Fill in missing parts
- Extend images (outpainting)
- Product placement

**2. Image-to-Image Translation**

```python
from diffusers import StableDiffusionImg2ImgPipeline

def style_transfer(image, prompt, strength=0.75):
    """
    Transform image based on prompt.
    
    Args:
        image: Input image
        prompt: Target style/content
        strength: How much to change (0=no change, 1=complete change)
    
    Returns:
        Styled image
    """
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    ).to("cuda")
    
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=7.5
    ).images[0]
    
    return result

# Example: Make photo look like oil painting
prompt = "oil painting, impressionist style, vibrant colors"
artistic = style_transfer(photo, prompt, strength=0.8)
```

**Use Cases:**
- Artistic style transfer
- Photo enhancement
- Sketch to photo
- Season transfer (summer → winter)

**3. ControlNet (Precise Control)**

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import HEDdetector

def controlled_generation(image, prompt, control_type="canny"):
    """
    Generate image with structure preserved.
    
    Args:
        image: Control image (edges, pose, depth, etc.)
        prompt: What to generate
        control_type: Type of control signal
    
    Returns:
        Generated image following structure
    """
    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained(
        f"lllyasviel/sd-controlnet-{control_type}"
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet
    ).to("cuda")
    
    # Generate
    result = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=50
    ).images[0]
    
    return result

# Example: Generate image matching sketch edges
prompt = "beautiful landscape, photorealistic, 4k"
generated = controlled_generation(sketch_edges, prompt, control_type="canny")
```

**ControlNet Types:**
- **Canny edges:** Preserve edges/structure
- **Depth map:** Preserve 3D geometry
- **Pose:** Preserve human poses
- **Segmentation:** Preserve object layout

**Business Applications:**

**Architecture/Interior Design:**
```python
# Input: Rough floor plan sketch
# Output: Photorealistic rendering
prompt = "modern interior design, natural lighting, minimalist"
design = controlled_generation(floor_plan, prompt, control_type="seg")

# ROI: $2000 per rendering → $10 per generation
# Savings: 99.5% cost reduction
```

**Fashion:**
```python
# Input: Model pose + clothing sketch
# Output: Realistic product photos
prompt = "professional fashion photography, studio lighting"
product_shot = controlled_generation(pose_skeleton, prompt, control_type="pose")

# ROI: $5000 photoshoot → $50 AI generation
# Savings: 99% cost reduction
```

**Marketing:**
```python
# Input: Product image + background description
# Output: Product in various settings
prompt = "product on beach at sunset, advertising photography"
ad_image = inpaint_image(product, background_mask, prompt)

# ROI: Create 100 variations in 1 hour vs 1 week
# Time savings: 98%
```

---

**End of Batch 2 (Slides 11-15)**

*Continue to Batch 3 for Audio & Music Generation (Slides 16-20)*
