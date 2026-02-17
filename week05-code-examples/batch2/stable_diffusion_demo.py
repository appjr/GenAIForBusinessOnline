"""
Stable Diffusion Demo - Text-to-Image Generation
From: week05-slides-batch2.md - Slide 13

Simple demo using Hugging Face Diffusers library.
"""

from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt, negative_prompt="", num_steps=50, guidance_scale=7.5):
    """
    Generate image from text using Stable Diffusion.
    
    Args:
        prompt: Text description of desired image
        negative_prompt: What to avoid in the image
        num_steps: Number of denoising steps (more = better quality but slower)
        guidance_scale: How strictly to follow the prompt (7-8 is good)
    
    Returns:
        PIL Image
    """
    # Load model (will download on first run, ~4GB)
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    print(f"Model loaded on {device}")
    
    # Generate
    print(f"Generating image for: '{prompt}'")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale
    ).images[0]
    
    return image


if __name__ == "__main__":
    print("="*70)
    print("STABLE DIFFUSION TEXT-TO-IMAGE DEMO")
    print("="*70)
    
    # Example prompts
    prompts = [
        {
            "prompt": "A beautiful sunset over mountains, oil painting style, highly detailed",
            "negative": "blurry, low quality, distorted",
            "filename": "sunset.png"
        },
        {
            "prompt": "A cute cat wearing a wizard hat, photorealistic, studio lighting, 4k",
            "negative": "cartoon, anime, low quality",
            "filename": "wizard_cat.png"
        },
        {
            "prompt": "Cyberpunk city at night, neon lights, raining, cinematic, highly detailed",
            "negative": "daytime, bright, low quality",
            "filename": "cyberpunk.png"
        }
    ]
    
    for example in prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: {example['prompt']}")
        print(f"Negative: {example['negative']}")
        
        image = generate_image(
            prompt=example['prompt'],
            negative_prompt=example['negative'],
            num_steps=30  # Reduced for faster generation
        )
        
        image.save(example['filename'])
        print(f"✓ Saved to {example['filename']}")
    
    print("\n" + "="*70)
    print("✓ Demo complete! Check the generated PNG files.")
    print("\nTips for better prompts:")
    print("  - Be specific about style (photorealistic, oil painting, etc.)")
    print("  - Add quality modifiers (highly detailed, 4k, masterpiece)")
    print("  - Specify lighting (studio lighting, golden hour, etc.)")
    print("  - Use negative prompts to avoid unwanted elements")
