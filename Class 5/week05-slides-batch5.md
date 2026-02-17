# Week 5: Image Generation, Audio, and Music - Slides Batch 5 (Slides 26-30)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Continuation from Batch 4**

---

## Slide 26: Multimodal Generation - Combining Vision and Audio

### When Images Meet Sound

**Multimodal AI:** Systems that work with multiple types of data simultaneously.

**Key Applications:**
- Video generation (images + audio)
- Image-to-audio (generate sounds from images)
- Audio-to-image (visualize audio)
- Cross-modal retrieval

**Video Generation Pipeline:**

```
Text Prompt → Video Frames (Diffusion) + Audio (MusicLM) → Synchronized Video
```

**Implementation:**

```python
class MultimodalGenerator:
    """
    Generate synchronized video and audio content.
    
    Combines image generation, audio generation, and synchronization.
    """
    
    def __init__(self):
        from diffusers import StableDiffusionPipeline
        
        # Image generator
        self.image_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        )
        
        # Audio generator (placeholder for real API)
        self.audio_generator = AudioGeneratorAPI()
        
    def generate_video(self, text_prompt, duration=10, fps=24):
        """
        Generate video with matching audio from text.
        
        Args:
            text_prompt: Description of desired video
            duration: Video length in seconds
            fps: Frames per second
        
        Returns:
            video_path: Path to generated video file
        """
        total_frames = duration * fps
        
        print(f"Generating {total_frames} frames...")
        
        # Generate keyframes
        keyframes = []
        for i in range(0, total_frames, fps):  # 1 keyframe per second
            # Add temporal variation to prompt
            frame_prompt = f"{text_prompt}, frame {i/fps:.1f}s"
            
            # Generate image
            image = self.image_pipeline(frame_prompt).images[0]
            keyframes.append(image)
        
        # Interpolate between keyframes
        all_frames = self.interpolate_frames(keyframes, fps)
        
        # Generate matching audio
        print("Generating audio...")
        audio_prompt = self.extract_audio_description(text_prompt)
        audio = self.audio_generator.generate(audio_prompt, duration=duration)
        
        # Combine into video
        video_path = self.create_video(all_frames, audio, fps)
        
        return video_path
    
    def interpolate_frames(self, keyframes, target_fps):
        """Smooth interpolation between keyframes"""
        from PIL import Image
        import numpy as np
        
        all_frames = []
        for i in range(len(keyframes) - 1):
            frame1 = np.array(keyframes[i])
            frame2 = np.array(keyframes[i + 1])
            
            # Linear interpolation
            for alpha in np.linspace(0, 1, target_fps):
                interpolated = (1 - alpha) * frame1 + alpha * frame2
                all_frames.append(Image.fromarray(interpolated.astype('uint8')))
        
        return all_frames
    
    def extract_audio_description(self, video_prompt):
        """Convert video description to audio description"""
        # Map visual to audio
        mapping = {
            'ocean': 'sound of waves, seagulls, gentle wind',
            'city': 'urban sounds, traffic, people talking',
            'forest': 'birds chirping, rustling leaves, wind',
            'cafe': 'background chatter, coffee machine, soft music'
        }
        
        # Simple keyword matching
        for keyword, audio_desc in mapping.items():
            if keyword in video_prompt.lower():
                return audio_desc
        
        return 'ambient background music'
    
    def create_video(self, frames, audio, fps):
        """Combine frames and audio into video file"""
        import moviepy.editor as mp
        
        # Create video clip from frames
        video = mp.ImageSequenceClip([np.array(f) for f in frames], fps=fps)
        
        # Add audio
        audio_clip = mp.AudioFileClip(audio)
        video = video.set_audio(audio_clip)
        
        # Save
        output_path = 'generated_video.mp4'
        video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        return output_path


# Business application: Automated video ads
class VideoAdGenerator:
    """
    Generate video advertisements automatically.
    
    Creates product videos with matching music and voiceover.
    """
    
    def __init__(self):
        self.multimodal = MultimodalGenerator()
        self.tts = TTSSystem()
    
    def create_product_ad(self, product_info, style='professional'):
        """
        Create complete video ad for product.
        
        Args:
            product_info: Dict with product details
            style: Ad style (professional, energetic, calm)
        
        Returns:
            video_path: Generated video ad
        """
        # Generate script
        script = self.generate_script(product_info, style)
        
        # Generate visuals
        visual_prompt = f"{product_info['name']}, {style} product photography"
        video_frames = self.multimodal.generate_video(visual_prompt, duration=15)
        
        # Generate voiceover
        voiceover = self.tts.synthesize(script)
        
        # Generate background music
        music_style = {
            'professional': 'corporate uplifting',
            'energetic': 'upbeat electronic',
            'calm': 'soft ambient'
        }[style]
        music = generate_music(music_style, duration=15)
        
        # Mix audio
        final_audio = self.mix_audio(voiceover, music)
        
        # Combine
        final_video = self.combine_video_audio(video_frames, final_audio)
        
        return final_video
    
    def generate_script(self, product_info, style):
        """Generate ad script"""
        templates = {
            'professional': f"Introducing {product_info['name']}. {product_info['description']}. Available now.",
            'energetic': f"Get ready for {product_info['name']}! {product_info['description']}! Get yours today!",
            'calm': f"Experience {product_info['name']}. {product_info['description']}. Discover more."
        }
        return templates[style]


# ROI calculation
def video_ad_roi():
    """Calculate ROI for automated video ads"""
    traditional = {
        'videographer': 2000,
        'voice_actor': 500,
        'music_licensing': 300,
        'editing': 1000,
        'total': 3800,
        'time': '1 week'
    }
    
    ai_approach = {
        'api_costs': 100,
        'review_editing': 200,
        'total': 300,
        'time': '1 hour'
    }
    
    savings = traditional['total'] - ai_approach['total']
    
    print(f"Savings per ad: ${savings} ({savings/traditional['total']*100:.1f}% reduction)")
    print(f"For 50 product ads: ${savings * 50:,} saved")
    print(f"Time savings: {traditional['time']} → {ai_approach['time']}")


if __name__ == "__main__":
    video_ad_roi()
```

**Real-World Multimodal Systems:**

**1. Video Diffusion Models**
- Runway Gen-2
- Pika Labs
- Stability AI Video

**2. Audio-Visual Learning**
- CLIP for audio-visual alignment
- AudioCLIP
- ImageBind (Meta)

---

## Slide 27: Real-Time Generation Considerations

### From Offline to Interactive

**The Real-Time Challenge:**

Traditional generation is slow:
- Diffusion: 2-5 seconds for one image
- Audio: 1-2 seconds for 1 second of audio
- Video: Minutes for a few seconds

**Business Need:** Real-time interactive applications.

**Solutions:**

**1. Model Distillation**

```python
class FastDiffusion:
    """
    Distilled diffusion model for real-time generation.
    
    Student model learns to generate in fewer steps.
    """
    
    def __init__(self, teacher_model, num_steps=4):
        """
        Args:
            teacher_model: Original slow diffusion model
            num_steps: Target number of steps (vs 50-1000)
        """
        self.teacher = teacher_model
        self.student = self.create_student_model()
        self.num_steps = num_steps
    
    def distill(self, training_data):
        """
        Train student to match teacher's output.
        
        Teacher generates with 50 steps.
        Student learns to match with only 4 steps.
        """
        for data in training_data:
            # Teacher prediction (slow, high quality)
            with torch.no_grad():
                teacher_output = self.teacher(data, num_steps=50)
            
            # Student prediction (fast)
            student_output = self.student(data, num_steps=self.num_steps)
            
            # Match outputs
            loss = F.mse_loss(student_output, teacher_output)
            loss.backward()
            optimizer.step()
    
    def generate_realtime(self, prompt):
        """Generate in <1 second"""
        return self.student(prompt, num_steps=self.num_steps)


# Example: SDXL Turbo (Stability AI)
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipeline.to("cuda")

# Single-step generation!
image = pipeline(
    prompt="A beautiful sunset",
    num_inference_steps=1,  # Just 1 step!
    guidance_scale=0.0  # No guidance needed
).images[0]

# Result: ~0.1 seconds per image
```

**2. Latent Caching**

```python
class CachedGenerator:
    """
    Cache intermediate results for faster generation.
    
    Useful for interactive editing.
    """
    
    def __init__(self):
        self.cache = {}
    
    def generate_with_cache(self, base_prompt, modifications):
        """
        Generate variations quickly by caching base.
        
        Args:
            base_prompt: Base description
            modifications: List of edits
        
        Returns:
            images: Generated variations
        """
        # Generate base once
        cache_key = base_prompt
        if cache_key not in self.cache:
            self.cache[cache_key] = self.generate_base(base_prompt)
        
        base_latent = self.cache[cache_key]
        
        # Apply modifications quickly
        results = []
        for mod in modifications:
            # Start from cached latent
            modified = self.apply_modification(base_latent, mod)
            results.append(modified)
        
        return results
```

**3. Progressive Generation**

```python
def progressive_generation(prompt):
    """
    Generate low-res quickly, then refine.
    
    Show user something immediately, improve over time.
    """
    # Quick low-res preview (0.1s)
    preview = generate_image(prompt, resolution=128, steps=4)
    display(preview)
    
    # Medium quality (0.5s)
    medium = generate_image(prompt, resolution=256, steps=8)
    display(medium)
    
    # Final quality (2s)
    final = generate_image(prompt, resolution=512, steps=20)
    display(final)
```

**Real-Time Applications:**

```python
# Interactive image editing
class RealTimeEditor:
    """
    Edit images in real-time as user types.
    """
    
    def __init__(self):
        self.model = FastDiffusion()
        self.current_image = None
    
    def on_prompt_change(self, new_prompt):
        """Called when user updates prompt"""
        # Generate immediately
        self.current_image = self.model.generate_realtime(new_prompt)
        self.display(self.current_image)
    
    def on_brush_stroke(self, mask, fill_prompt):
        """Called when user paints on image"""
        # Instant inpainting
        self.current_image = self.model.inpaint_realtime(
            self.current_image, mask, fill_prompt
        )
        self.display(self.current_image)


# Gaming: Generate assets on-the-fly
class GameAssetGenerator:
    """
    Generate game assets in real-time during gameplay.
    """
    
    def generate_npc_dialogue(self, context):
        """Generate NPC responses instantly"""
        # Must be <100ms for smooth gameplay
        response = self.fast_llm.generate(context, max_tokens=50)
        return response
    
    def generate_background_music(self, scene_mood):
        """Adapt music to scene"""
        # Transition smoothly (real-time mixing)
        new_music = self.music_generator.generate_snippet(scene_mood)
        self.audio_mixer.crossfade(new_music, duration=2.0)
```

**Performance Targets:**

| Application | Latency Target | Solution |
|-------------|---------------|----------|
| Image Editor | <100ms | Distilled models + caching |
| Video Chat Effects | <33ms (30 FPS) | Lightweight models |
| Gaming | <16ms (60 FPS) | Pre-generated + blending |
| Music Apps | <10ms | Streaming generation |

---

## Slide 28: Quality Metrics and Evaluation

### How Do We Know If It's Good?

**The Challenge:** Evaluating generative models is subjective.

**Evaluation Approaches:**

**1. Objective Metrics**

```python
class GenerativeMetrics:
    """
    Compute objective quality metrics for generated content.
    """
    
    def __init__(self):
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
        
        self.fid = FrechetInceptionDistance(feature=2048)
        self.inception_score = InceptionScore()
    
    def compute_image_quality(self, generated_images, real_images):
        """
        Compute FID and Inception Score.
        
        Args:
            generated_images: Generated images
            real_images: Real reference images
        
        Returns:
            metrics: Dict of quality scores
        """
        # Fréchet Inception Distance (FID)
        # Lower is better, measures distribution similarity
        fid_score = self.fid(generated_images, real_images)
        
        # Inception Score (IS)
        # Higher is better, measures quality and diversity
        is_score, _ = self.inception_score(generated_images)
        
        return {
            'fid': fid_score.item(),
            'inception_score': is_score.item()
        }
    
    def compute_audio_quality(self, generated_audio, reference_audio):
        """
        Audio quality metrics.
        """
        import librosa
        
        # Mel Cepstral Distortion (MCD)
        # Lower is better
        mcd = self.compute_mcd(generated_audio, reference_audio)
        
        # Signal-to-Noise Ratio
        snr = self.compute_snr(generated_audio, reference_audio)
        
        # Perceptual Evaluation of Speech Quality (PESQ)
        from pesq import pesq
        pesq_score = pesq(16000, reference_audio, generated_audio, 'wb')
        
        return {
            'mcd': mcd,
            'snr': snr,
            'pesq': pesq_score
        }
    
    def compute_music_quality(self, generated_music):
        """Music-specific metrics"""
        # Pitch accuracy
        pitch_acc = self.evaluate_pitch(generated_music)
        
        # Rhythm consistency
        rhythm_score = self.evaluate_rhythm(generated_music)
        
        # Harmonic coherence
        harmony_score = self.evaluate_harmony(generated_music)
        
        return {
            'pitch_accuracy': pitch_acc,
            'rhythm_consistency': rhythm_score,
            'harmonic_coherence': harmony_score
        }


# Human evaluation
class HumanEvaluation:
    """
    Conduct human evaluation studies.
    """
    
    def run_ab_test(self, generated_samples, real_samples):
        """
        A/B test: Can humans tell the difference?
        
        Returns:
            fooling_rate: % of time AI fools humans
        """
        correct = 0
        total = 0
        
        for gen, real in zip(generated_samples, real_samples):
            # Show pair to human, ask which is real
            if random.random() < 0.5:
                shown = [(gen, 'A'), (real, 'B')]
            else:
                shown = [(real, 'A'), (gen, 'B')]
            
            answer = get_human_judgment(shown)
            if answer == 'B' if shown[1][1] == 'real' else 'A':
                correct += 1
            total += 1
        
        fooling_rate = 1 - (correct / total)
        return fooling_rate
    
    def collect_ratings(self, samples, criteria=['quality', 'creativity', 'relevance']):
        """
        Collect Likert scale ratings.
        
        Args:
            samples: Generated samples
            criteria: What to rate
        
        Returns:
            ratings: Mean ratings per criterion
        """
        ratings = {c: [] for c in criteria}
        
        for sample in samples:
            for criterion in criteria:
                rating = get_human_rating(sample, criterion, scale=1-5)
                ratings[criterion].append(rating)
        
        # Compute means
        mean_ratings = {c: np.mean(ratings[c]) for c in criteria}
        
        return mean_ratings
```

**2. Task-Specific Metrics**

```python
# Image generation
def evaluate_image_generation(model, test_prompts):
    """Comprehensive image evaluation"""
    results = {
        'fid': compute_fid(model, test_prompts),
        'clip_score': compute_clip_alignment(model, test_prompts),
        'diversity': compute_diversity(model, test_prompts),
        'aesthetic_score': compute_aesthetic_quality(model, test_prompts)
    }
    return results


# Audio generation
def evaluate_tts_system(tts_model, test_sentences):
    """Evaluate text-to-speech"""
    results = {
        'intelligibility': run_transcription_test(tts_model, test_sentences),
        'naturalness': collect_mos_scores(tts_model, test_sentences),  # MOS = Mean Opinion Score
        'speaker_similarity': compute_speaker_embedding_distance(tts_model),
        'prosody': evaluate_prosody(tts_model, test_sentences)
    }
    return results


# Music generation
def evaluate_music_generator(music_model):
    """Evaluate music quality"""
    results = {
        'musicality': expert_evaluation(music_model),
        'originality': check_plagiarism(music_model),
        'genre_accuracy': classify_generated_music(music_model),
        'user_engagement': measure_listening_duration(music_model)
    }
    return results
```

---

## Slide 29: Compression and Efficiency

### Making GenAI Practical at Scale

**The Cost Problem:**

Large models are expensive:
- Stable Diffusion: 860M parameters
- DALL-E 2: 3.5B parameters
- MusicLM: Multiple large models

**Solutions:**

**1. Model Quantization**

```python
import torch

def quantize_model(model, bits=8):
    """
    Reduce model precision.
    
    FP32 (32-bit) → INT8 (8-bit)
    4x smaller, 2-4x faster
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    return quantized_model


# Example: INT8 quantization
model_fp32 = load_model()  # 3.4 GB
model_int8 = quantize_model(model_fp32)  # 850 MB

# Minimal quality loss, major speedup
```

**2. Pruning**

```python
def prune_model(model, sparsity=0.5):
    """
    Remove unimportant weights.
    
    Args:
        model: Neural network
        sparsity: Fraction of weights to remove
    """
    import torch.nn.utils.prune as prune
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')
    
    return model


# 50% sparsity → 50% fewer parameters → 2x faster
```

**3. Knowledge Distillation**

```python
def distill_model(large_model, small_model, data):
    """
    Train small model to mimic large model.
    
    Student learns from teacher's outputs.
    """
    temperature = 3.0
    
    for batch in data:
        # Teacher predictions (soft targets)
        with torch.no_grad():
            teacher_logits = large_model(batch)
            soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        
        # Student predictions
        student_logits = small_model(batch)
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # Distillation loss
        loss = F.kl_div(student_probs, soft_targets, reduction='batchmean')
        loss = loss * (temperature ** 2)
        
        # Train student
        loss.backward()
        optimizer.step()
    
    return small_model


# Result: 10x smaller model, 90% of performance
```

**4. Efficient Architectures**

```python
# MobileNet-style depthwise separable convolutions
class EfficientConv(nn.Module):
    """Efficient convolution for mobile deployment"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        # Depthwise (channel-wise)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            groups=in_channels,  # Key: groups=in_channels
            padding=kernel_size//2
        )
        
        # Pointwise (1x1 conv)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 8-9x fewer parameters than standard convolution
```

**Cost Optimization:**

```python
def optimize_deployment(model_type, expected_requests_per_day):
    """
    Choose optimal deployment strategy.
    
    Args:
        model_type: 'image', 'audio', or 'music'
        expected_requests_per_day: Daily request volume
    
    Returns:
        recommendation: Deployment strategy
    """
    strategies = {
        'low': {  # <1000 requests/day
            'approach': 'API service (pay-per-use)',
            'cost': expected_requests_per_day * 0.01,
            'latency': 'medium'
        },
        'medium': {  # 1000-10000 requests/day
            'approach': 'Shared GPU instance',
            'cost': 500,  # monthly
            'latency': 'low'
        },
        'high': {  # >10000 requests/day
            'approach': 'Dedicated GPU cluster + quantization',
            'cost': 2000,  # monthly
            'latency': 'very low'
        }
    }
    
    if expected_requests_per_day < 1000:
        return strategies['low']
    elif expected_requests_per_day < 10000:
        return strategies['medium']
    else:
        return strategies['high']
```

---

## Slide 30: Break & Review - Mid-Week Checkpoint

### What We've Covered So Far

**Part 1: Image Generation (Slides 1-15)**
✅ VAEs - Smooth latent spaces
✅ GANs - Adversarial training
✅ Diffusion Models - State-of-the-art
✅ Stable Diffusion - Text-to-image
✅ ControlNet - Precise control
✅ Image editing - Inpainting, style transfer

**Part 2: Audio & Music (Slides 16-25)**
✅ Audio representations
✅ WaveNet - Sample-by-sample generation
✅ TTS - Text-to-speech systems
✅ Voice cloning
✅ Music RNN & Transformers
✅ MuseNet & MusicLM
✅ Music style transfer

**Part 3: Advanced Topics (Slides 26-30)**
✅ Multimodal generation
✅ Real-time considerations
✅ Quality metrics
✅ Compression & efficiency
✅ Deployment strategies

---

**Quick Quiz:**

1. **What's the main advantage of diffusion models over GANs?**
   - Answer: More stable training, no mode collapse, better quality

2. **Why is WaveNet slow for audio generation?**
   - Answer: Autoregressive, generates sample-by-sample (44,100/second)

3. **What's the key to real-time generation?**
   - Answer: Distillation, quantization, efficient architectures

4. **How do we evaluate if generated content is good?**
   - Answer: Combination of objective metrics (FID, IS) and human evaluation

5. **What's the cost-performance trade-off?**
   - Answer: Smaller models = faster/cheaper but slightly lower quality

---

**Break Activity (10 minutes):**

Try these demos:
1. Generate an image: https://huggingface.co/spaces/stabilityai/stable-diffusion
2. Generate music: https://google-research.github.io/seanet/musiclm/examples/
3. Try voice cloning: https://elevenlabs.io

---

**Coming Up (Slides 31-40):**
- Real-world business applications
- Industry case studies
- ROI calculators
- Implementation strategies
- Best practices
- Ethics and governance
- Week 5 assignment

**Take a 10-minute break! ☕**

---

**End of Batch 5 (Slides 26-30)**

*Continue to Batch 6 for Business Applications Part 1 (Slides 31-35)*
