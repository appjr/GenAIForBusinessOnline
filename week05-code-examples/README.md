# Week 5 Code Examples: Image, Audio, and Music Generation

Complete, runnable code examples extracted from Week 5 slides.

## 📁 Directory Structure

```
week05-code-examples/
├── README.md (this file)
├── batch1/  # Image Generation Basics
│   ├── vae_mnist.py
│   ├── gan_mnist.py
│   └── simple_diffusion.py
├── batch2/  # Advanced Image Generation
│   ├── stable_diffusion_demo.py
│   └── image_editing.py
├── batch3/  # Audio Generation
│   ├── audio_processor.py
│   ├── wavenet_simple.py
│   └── tts_demo.py
└── batch4/  # Music & Applications
    ├── music_lstm.py
    ├── music_transformer.py
    └── business_roi.py
```

## 🎯 Learning Objectives

By working through these examples, you will:
- ✅ Understand VAEs, GANs, and Diffusion Models
- ✅ Generate images using Stable Diffusion
- ✅ Build TTS and voice cloning systems
- ✅ Generate music using transformers
- ✅ Calculate business ROI for GenAI applications

## 📚 Batch Descriptions

### Batch 1: Image Generation Basics
- **vae_mnist.py**: Complete VAE implementation on MNIST
- **gan_mnist.py**: GAN with Generator and Discriminator
- **simple_diffusion.py**: DDPM (Denoising Diffusion Probabilistic Model)

### Batch 2: Advanced Image Generation
- **stable_diffusion_demo.py**: Text-to-image generation
- **image_editing.py**: Inpainting, img2img, ControlNet

### Batch 3: Audio Generation
- **audio_processor.py**: Audio utilities (load, save, spectrogram)
- **wavenet_simple.py**: Simplified WaveNet implementation
- **tts_demo.py**: Text-to-speech with Tacotron 2

### Batch 4: Music & Business Applications
- **music_lstm.py**: Melody generation with LSTM
- **music_transformer.py**: Polyphonic music generation
- **business_roi.py**: ROI calculators for various applications

## 🚀 Getting Started

### Installation

```bash
# Core packages
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install librosa soundfile

# For music
pip install pretty_midi music21

# Optional: For GPU acceleration
pip install xformers  # Faster attention
```

### Running Examples

```bash
# Batch 1: Image Generation
python batch1/vae_mnist.py
python batch1/gan_mnist.py
python batch1/simple_diffusion.py

# Batch 2: Stable Diffusion
python batch2/stable_diffusion_demo.py

# Batch 3: Audio
python batch3/audio_processor.py
python batch3/wavenet_simple.py

# Batch 4: Music
python batch4/music_lstm.py
python batch4/music_transformer.py
```

## 💡 Tips

1. **GPU Recommended**: Most models run much faster on GPU
2. **Memory**: Some models need 8GB+ VRAM
3. **Start Small**: Try smaller models/datasets first
4. **Check Slides**: Reference full slides for theory and context

## 📖 Related Materials

- **Slides**: week05-slides-batch*.md files
- **Week 4**: Neural network foundations
- **Assignment**: See week05-slides-batch7.md (Slide 38)

## 🔗 Resources

- Hugging Face Diffusers: https://huggingface.co/docs/diffusers
- PyTorch Tutorials: https://pytorch.org/tutorials/
- Magenta (Music): https://magenta.tensorflow.org/

## 📝 Assignment

See **Slide 38** in week05-slides-batch7.md for the Week 5 project assignment.

Choose one:
1. Marketing Content Generator
2. Audio Application
3. Multimodal System

## 🆘 Troubleshooting

**Out of Memory?**
- Reduce batch size
- Use smaller resolution
- Enable gradient checkpointing
- Use CPU (slower but works)

**Slow Generation?**
- Use GPU
- Try distilled models (SDXL Turbo)
- Reduce inference steps

**Import Errors?**
- Check all packages installed
- Try: `pip install -r requirements.txt`
- Verify Python 3.8+

## 📧 Support

- Office Hours: TBD
- Discussion Forum: Canvas
- GitHub Issues: Course repo

---

**Course**: BUAN 6v99.SW2 - Generative AI for Business  
**Week**: 5 - Image, Audio, and Music Generation  
**Spring 2026**
