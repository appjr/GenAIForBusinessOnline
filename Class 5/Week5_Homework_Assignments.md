# Week 5 Homework Assignments
## Image, Audio, and Music Generation with AI

**Course:** BUAN 6V99.SW2 - Generative AI for Business  
**Due Date:** [To be announced]  
**Total Points:** 100

---

## Assignment Overview

Choose **ONE** of the following three tracks. Each track focuses on a different aspect of generative AI covered in Week 5.

### Track Options:
1. **Image Generation Track** (VAE, GAN, or Diffusion Models)
2. **Audio Generation Track** (Speech/Sound Synthesis)
3. **Music Generation Track** (AI-Composed Music)

---

## 📊 TRACK 1: Image Generation (100 points)

### Option A: Fashion Design with VAE (100 points)

**Objective:** Build a VAE to generate new fashion item designs from the Fashion MNIST dataset.

**Requirements:**

#### Part 1: Implementation (60 points)
1. **Data Loading** (10 points)
   - Load Fashion MNIST dataset
   - Preprocess and normalize images
   - Split into train/validation sets

2. **VAE Architecture** (20 points)
   - Implement encoder with appropriate layers
   - Implement decoder architecture
   - Include reparameterization trick
   - Latent dimension: 10-20

3. **Training** (20 points)
   - Train for at least 20 epochs
   - Monitor reconstruction loss
   - Monitor KL divergence
   - Save model checkpoints

4. **Generation** (10 points)
   - Generate 50 new fashion items
   - Create interpolations between items
   - Visualize latent space

#### Part 2: Analysis (30 points)
1. **Quality Evaluation** (10 points)
   - Analyze reconstruction quality
   - Compare original vs reconstructed images
   - Calculate metrics (MSE, SSIM if possible)

2. **Business Application** (15 points)
   - Describe how this could be used in fashion retail
   - Estimate cost savings vs manual design
   - Identify 3 specific use cases

3. **Limitations** (5 points)
   - Discuss model limitations
   - Suggest improvements

#### Part 3: Documentation (10 points)
- Well-commented code
- README with setup instructions
- Results visualization

**Deliverables:**
- Jupyter notebook with all code
- Generated images (grid of 50 samples)
- 2-page business analysis report (PDF)

---

### Option B: Product Image Generation with GAN (100 points)

**Objective:** Create a GAN to generate realistic product images.

**Requirements:**

#### Part 1: Implementation (60 points)
1. **Dataset Selection** (10 points)
   - Choose dataset (CIFAR-10, CelebA, or custom)
   - Justify choice for business application
   - Load and preprocess data

2. **GAN Architecture** (25 points)
   - Implement Generator network
   - Implement Discriminator network
   - Use appropriate activation functions
   - Include batch normalization

3. **Training** (20 points)
   - Implement alternating training
   - Monitor G and D losses
   - Save generated samples every 10 epochs
   - Train for 50+ epochs

4. **Evaluation** (5 points)
   - Generate final samples
   - Assess quality progression

#### Part 2: Business Analysis (30 points)
1. **ROI Calculation** (10 points)
   - Cost of traditional product photography
   - Cost of AI-generated images
   - Savings estimation

2. **Use Cases** (15 points)
   - Identify 3 business applications
   - Describe implementation strategy
   - Address ethical considerations

3. **Quality Assessment** (5 points)
   - Compare to real images
   - Discuss when AI-generated is sufficient

#### Part 3: Documentation (10 points)
- Clean, documented code
- Training progression visualization
- Final report with recommendations

**Deliverables:**
- Jupyter notebook with implementation
- GIF showing training progression
- Business case report (3 pages, PDF)

---

### Option C: Text-to-Image with Stable Diffusion (100 points)

**Objective:** Use Stable Diffusion to generate marketing images from text prompts.

**Requirements:**

#### Part 1: Implementation (50 points)
1. **Setup** (10 points)
   - Install required libraries
   - Load Stable Diffusion model
   - Configure for optimal performance

2. **Prompt Engineering** (25 points)
   - Create 20 business-relevant prompts
   - Test different parameter settings
   - Document what works best
   - Categories: product, marketing, branding, lifestyle

3. **Generation** (15 points)
   - Generate images for each prompt
   - Vary parameters (steps, guidance_scale)
   - Use negative prompts effectively

#### Part 2: Business Application (35 points)
1. **Marketing Campaign** (20 points)
   - Design complete campaign (5 images)
   - Write prompts and rationale
   - Compare to stock photo costs

2. **Best Practices Guide** (10 points)
   - Document prompt patterns that work
   - Provide tips for business users
   - Include do's and don'ts

3. **Limitations** (5 points)
   - When NOT to use AI images
   - Ethical considerations
   - Copyright/licensing issues

#### Part 3: Presentation (15 points)
- Professional slide deck (10 slides)
- Clear examples and comparisons
- ROI analysis

**Deliverables:**
- Collection of 20+ generated images
- Prompt engineering guide (2 pages)
- Marketing campaign presentation (PDF/PPTX)
- Cost-benefit analysis

---

## 🎵 TRACK 2: Audio Generation (100 points)

### Assignment: Voice Assistant for Business

**Objective:** Create a text-to-speech system for business applications.

**Requirements:**

#### Part 1: Implementation (50 points)
1. **TTS System** (30 points)
   - Implement or use existing TTS library
   - Generate speech from text
   - Test with business scripts
   - Support multiple voices/styles

2. **Audio Processing** (20 points)
   - Load and analyze audio files
   - Create spectrograms
   - Measure audio quality metrics
   - Apply audio effects if needed

#### Part 2: Business Application (35 points)
1. **Use Case Development** (20 points)
   - Choose specific business scenario:
     * Customer service IVR
     * E-learning narration
     * Accessibility features
     * Voice notifications
   - Create 10 sample scripts
   - Generate audio for each

2. **Quality Analysis** (10 points)
   - Compare to human voice
   - Assess naturalness
   - Test comprehension

3. **Cost Analysis** (5 points)
   - Voice actor costs vs AI
   - Implementation expenses
   - ROI calculation

#### Part 3: Documentation (15 points)
- Implementation guide
- Audio samples library
- Business case report (3 pages)

**Deliverables:**
- Code/notebook with implementation
- 10 audio files (WAV/MP3)
- Business case analysis (PDF)
- Demo video (2-3 minutes)

---

## 🎼 TRACK 3: Music Generation (100 points)

### Assignment: AI Music for Business Applications

**Objective:** Generate background music for business use (ads, videos, retail).

**Requirements:**

#### Part 1: Implementation (50 points)
1. **Music Generation** (35 points)
   - Use MusicLM, MuseNet, or similar
   - Generate 5 different styles:
     * Upbeat/energetic
     * Calm/professional
     * Exciting/dramatic
     * Warm/friendly
     * Modern/tech
   - Each 30-60 seconds

2. **Customization** (15 points)
   - Adjust tempo/mood
   - Test different prompts
   - Create variations

#### Part 2: Business Application (35 points)
1. **Use Case Portfolio** (20 points)
   - Match music to business scenarios:
     * Retail store ambiance
     * Product video background
     * Podcast intro/outro
     * On-hold music
     * Presentation background
   - Justify each choice

2. **Cost Analysis** (10 points)
   - Licensing costs for stock music
   - AI generation costs
   - Calculate savings for 1-year

3. **Quality Comparison** (5 points)
   - Compare to licensed music
   - Survey feedback (5+ people)

#### Part 3: Portfolio Presentation (15 points)
- Music sample library
- Usage guide
- Business proposal

**Deliverables:**
- 5+ generated music tracks
- Matching guide (which track for what)
- Business proposal (3 pages, PDF)
- Demo video showcasing tracks in context

---

## 📋 Grading Rubric (All Tracks)

### Technical Implementation (50-60 points)
- **Excellent (90-100%):** Code runs perfectly, well-documented, creative solutions
- **Good (80-89%):** Code works with minor issues, adequately documented
- **Satisfactory (70-79%):** Code mostly works, basic documentation
- **Needs Improvement (<70%):** Major issues, poor documentation

### Business Analysis (30-35 points)
- **Excellent (90-100%):** Comprehensive analysis, realistic costs, strong ROI case
- **Good (80-89%):** Solid analysis, reasonable estimates, clear benefits
- **Satisfactory (70-79%):** Basic analysis, general estimates
- **Needs Improvement (<70%):** Weak analysis, missing components

### Documentation & Presentation (10-15 points)
- **Excellent (90-100%):** Professional, clear, complete, visually appealing
- **Good (80-89%):** Clear and organized, minor improvements needed
- **Satisfactory (70-79%):** Basic documentation, meets requirements
- **Needs Improvement (<70%):** Incomplete or unclear

---

## 💡 Submission Guidelines

### File Naming Convention:
```
LastName_FirstName_Week5_TrackNumber.zip
```

### Required Files:
1. Jupyter notebook (.ipynb) or Python script (.py)
2. Report (PDF)
3. Generated outputs (images/audio/music)
4. README.txt with setup instructions
5. requirements.txt (if applicable)

### Submission:
- Upload to Canvas/Learning Management System
- Ensure all files are included
- Test that code runs before submitting

---

## 🎯 Tips for Success

### General:
- Start early - generative models can be computationally intensive
- Use Google Colab if you don't have GPU access
- Document your process as you go
- Test code incrementally

### For Image Generation:
- Start with small images (28x28 or 64x64)
- Monitor training losses carefully
- Save checkpoints regularly
- Use pre-trained models when possible

### For Audio/Music:
- Use existing APIs/libraries (faster development)
- Focus on business application over implementation
- Gather user feedback
- Consider quality vs cost tradeoff

### For All Tracks:
- **Business focus:** This is a business analytics course
- **Practical application:** Real-world use cases matter
- **ROI analysis:** Show the financial benefit
- **Ethics:** Address potential issues

---

## 📚 Resources

### Code Examples:
- Week 5 code examples: `Class 5/week05-code-examples/`
- Week 5 notebooks: `Class 5/week05-notebooks/`

### Libraries:
- **Image:** PyTorch, TensorFlow, diffusers, transformers
- **Audio:** librosa, pyttsx3, gTTS, TorToiSe
- **Music:** music21, magenta, MusicLM API

### Datasets:
- Fashion MNIST: `torchvision.datasets`
- CIFAR-10: `torchvision.datasets`
- Audio: Free Music Archive, LibriSpeech

### Documentation:
- Hugging Face Diffusers: https://huggingface.co/docs/diffusers
- PyTorch Tutorials: https://pytorch.org/tutorials/
- Librosa Docs: https://librosa.org/doc/latest/

---

## ❓ Frequently Asked Questions

**Q: Can I use pre-trained models?**  
A: Yes! Focus on application and business analysis.

**Q: Do I need a GPU?**  
A: Recommended but not required. Use Google Colab free tier.

**Q: Can I work in a group?**  
A: Check with instructor. If yes, max 2 people, both submit.

**Q: What if my model doesn't generate good results?**  
A: Document challenges and discuss in limitations section.

**Q: How long should training take?**  
A: Varies by model. Budget 2-4 hours for training time.

---

## 🏆 Bonus Opportunities (Optional, +10 points max)

1. **Compare multiple models** (+5 points)
   - Implement 2 different approaches
   - Compare results quantitatively

2. **Real business pilot** (+10 points)
   - Partner with actual business
   - Deploy your solution
   - Document results

3. **Interactive demo** (+5 points)
   - Create Gradio/Streamlit interface
   - Deploy online
   - Share link

---

## 📞 Support

- **Office Hours:** [Times]
- **Discussion Forum:** [Link]
- **Email:** [Instructor email]
- **Technical Issues:** Post in course forum

---

**Good luck! Focus on creating practical business value with generative AI!** 🚀
