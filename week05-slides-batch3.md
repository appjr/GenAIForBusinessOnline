# Week 5: Image Generation, Audio, and Music - Slides Batch 3 (Slides 16-20)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Continuation from Batch 2**

---

## Slide 16: Introduction to Audio Generation

### From Images to Audio - New Challenges

**Why Audio is Different from Images:**

| Aspect | Images | Audio |
|--------|--------|-------|
| **Dimensions** | 2D (height × width) | 1D (time) but high sampling rate |
| **Sampling** | ~1M pixels (512×512×3) | ~44,100 samples/second |
| **Perception** | Spatial, instant | Temporal, sequential |
| **Generation** | Independent pixels | Must maintain continuity |
| **Quality** | Some blur tolerable | Artifacts very noticeable |

**Key Challenge:** Audio requires **high temporal resolution**
- 1 second = 44,100 samples
- Tiny errors create audible glitches
- Must maintain long-range coherence

**Audio Representations:**

**1. Raw Waveform**
```
Amplitude vs Time
Sample rate: 44.1kHz (CD quality)
Pros: Direct, lossless
Cons: Very high dimensional
```

**2. Spectrogram**
```
Frequency × Time
Created via Short-Time Fourier Transform (STFT)
Pros: Lower dimensional, visual
Cons: Phase information lost
```

**3. Mel-Spectrogram**
```
Mel-scaled frequency × Time
Matches human perception
Pros: Perceptually relevant
Cons: Need vocoder to convert back
```

**Audio Processing Basics:**

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

class AudioProcessor:
    """
    Utilities for audio processing and visualization.
    
    Handles loading, converting, and visualizing audio.
    """
    
    def __init__(self, sample_rate=22050):
        """
        Args:
            sample_rate: Target sampling rate (Hz)
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, filename):
        """
        Load audio file.
        
        Args:
            filename: Path to audio file
        
        Returns:
            waveform: Audio samples
            sample_rate: Sampling rate
        """
        waveform, sr = librosa.load(filename, sr=self.sample_rate)
        return waveform, sr
    
    def save_audio(self, waveform, filename):
        """Save audio to file"""
        # Normalize to 16-bit range
        waveform = np.int16(waveform * 32767)
        wavfile.write(filename, self.sample_rate, waveform)
    
    def compute_spectrogram(self, waveform):
        """
        Compute mel-spectrogram.
        
        Args:
            waveform: Audio samples
        
        Returns:
            mel_spec: Mel-spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=128,
            fmax=8000
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def visualize_audio(self, waveform, mel_spec=None):
        """
        Visualize waveform and spectrogram.
        
        Args:
            waveform: Audio samples
            mel_spec: Optional mel-spectrogram
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Waveform
        time = np.arange(len(waveform)) / self.sample_rate
        axes[0].plot(time, waveform)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Waveform')
        axes[0].grid(True, alpha=0.3)
        
        # Spectrogram
        if mel_spec is None:
            mel_spec = self.compute_spectrogram(waveform)
        
        img = librosa.display.specshow(
            mel_spec,
            sr=self.sample_rate,
            x_axis='time',
            y_axis='mel',
            ax=axes[1],
            cmap='viridis'
        )
        axes[1].set_title('Mel-Spectrogram')
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        plt.tight_layout()
        return fig


# Usage example
if __name__ == "__main__":
    processor = AudioProcessor(sample_rate=22050)
    
    # Load audio
    waveform, sr = processor.load_audio('sample.wav')
    print(f"Loaded {len(waveform)} samples at {sr}Hz")
    print(f"Duration: {len(waveform)/sr:.2f} seconds")
    
    # Compute spectrogram
    mel_spec = processor.compute_spectrogram(waveform)
    print(f"Spectrogram shape: {mel_spec.shape}")
    
    # Visualize
    fig = processor.visualize_audio(waveform, mel_spec)
    plt.savefig('audio_visualization.png', dpi=300)
    plt.show()
```

**Business Relevance:**
- Content creation (podcasts, audiobooks)
- Voice assistants
- Music production
- Accessibility (text-to-speech)
- Entertainment (game audio, effects)

---

## Slide 17: WaveNet - Direct Waveform Generation

### Generating Audio Sample-by-Sample

**WaveNet (DeepMind, 2016):**  
Revolutionary model that generates raw audio waveforms directly.

**Key Innovation:** Dilated Causal Convolutions

**Traditional Convolution:**
```
Limited receptive field
Can only "see" nearby samples
```

**Dilated Causal Convolution:**
```
Exponentially growing receptive field
Layer 1: sees 2 samples
Layer 2: sees 4 samples  
Layer 3: sees 8 samples
...
Layer 10: sees 1024 samples
```

**Architecture:**

```
Input: Previous audio samples
    ↓
Dilated Conv Layer 1 (dilation=1)
    ↓
Dilated Conv Layer 2 (dilation=2)
    ↓
Dilated Conv Layer 3 (dilation=4)
    ↓
... (stack of dilated conv layers)
    ↓
Output: Probability distribution over next sample
```

**Simplified WaveNet Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution - only looks at past samples.
    
    Ensures autoregressive property: output at time t
    only depends on inputs at times < t.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
    
    def forward(self, x):
        """Forward pass with causal padding"""
        x = self.conv(x)
        # Remove future samples
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with gated activation.
    
    Core building block of WaveNet.
    """
    
    def __init__(self, residual_channels, dilation):
        super().__init__()
        
        self.conv_filter = CausalConv1d(
            residual_channels,
            residual_channels,
            kernel_size=2,
            dilation=dilation
        )
        
        self.conv_gate = CausalConv1d(
            residual_channels,
            residual_channels,
            kernel_size=2,
            dilation=dilation
        )
        
        self.conv_residual = nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=1
        )
        
        self.conv_skip = nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=1
        )
    
    def forward(self, x):
        """
        Gated activation: tanh(filter) ⊙ sigmoid(gate)
        
        Args:
            x: Input tensor
        
        Returns:
            residual: For next layer
            skip: For skip connection
        """
        # Gated activation
        f = self.conv_filter(x)
        g = self.conv_gate(x)
        z = torch.tanh(f) * torch.sigmoid(g)
        
        # Residual and skip connections
        residual = self.conv_residual(z) + x
        skip = self.conv_skip(z)
        
        return residual, skip


class SimpleWaveNet(nn.Module):
    """
    Simplified WaveNet for audio generation.
    
    Generates audio sample-by-sample autoregressively.
    """
    
    def __init__(self, 
                 layers=10,
                 blocks=2,
                 residual_channels=32,
                 quantization_levels=256):
        """
        Args:
            layers: Number of layers per block
            blocks: Number of blocks
            residual_channels: Hidden dimension
            quantization_levels: Audio quantization (typically 256 for 8-bit)
        """
        super().__init__()
        
        self.quantization_levels = quantization_levels
        
        # Input embedding
        self.input_conv = CausalConv1d(
            quantization_levels,
            residual_channels,
            kernel_size=2
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for b in range(blocks):
            for i in range(layers):
                dilation = 2 ** i
                self.residual_blocks.append(
                    ResidualBlock(residual_channels, dilation)
                )
        
        # Output layers
        self.output_conv1 = nn.Conv1d(residual_channels, residual_channels, 1)
        self.output_conv2 = nn.Conv1d(residual_channels, quantization_levels, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: One-hot encoded audio (batch, quantization_levels, time)
        
        Returns:
            logits: Predicted distribution over next sample
        """
        # Input embedding
        x = self.input_conv(x)
        
        # Residual blocks with skip connections
        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Sum skip connections
        x = sum(skip_connections)
        
        # Output
        x = F.relu(x)
        x = self.output_conv1(x)
        x = F.relu(x)
        x = self.output_conv2(x)
        
        return x
    
    @torch.no_grad()
    def generate(self, length, temperature=1.0):
        """
        Generate audio sample-by-sample.
        
        Args:
            length: Number of samples to generate
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            generated: Generated audio samples
        """
        # Start with silence
        generated = torch.zeros(1, self.quantization_levels, 1)
        
        for _ in range(length):
            # Predict next sample
            logits = self.forward(generated)[:, :, -1]
            
            # Sample from distribution
            probs = F.softmax(logits / temperature, dim=1)
            next_sample = torch.multinomial(probs, 1)
            
            # One-hot encode
            next_sample_onehot = F.one_hot(
                next_sample,
                num_classes=self.quantization_levels
            ).float().transpose(1, 2)
            
            # Append
            generated = torch.cat([generated, next_sample_onehot], dim=2)
        
        return generated


def mu_law_encode(audio, quantization_levels=256):
    """
    μ-law companding for better audio quantization.
    
    Compresses dynamic range of audio signal.
    """
    mu = quantization_levels - 1
    safe_audio = np.minimum(np.maximum(audio, -1.0), 1.0)
    magnitude = np.abs(safe_audio)
    signal = np.sign(safe_audio) * np.log1p(mu * magnitude) / np.log1p(mu)
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int64)


def mu_law_decode(quantized, quantization_levels=256):
    """Decode μ-law encoded audio"""
    mu = quantization_levels - 1
    signal = 2 * (quantized / mu) - 1
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude


# Training example
if __name__ == "__main__":
    print("="*70)
    print("WAVENET FOR AUDIO GENERATION")
    print("="*70)
    
    # Create model
    model = SimpleWaveNet(
        layers=10,
        blocks=2,
        residual_channels=32,
        quantization_levels=256
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Receptive field: {2**10} samples")
    
    # Generate audio
    print("\nGenerating audio...")
    with torch.no_grad():
        generated = model.generate(length=1000, temperature=1.0)
    
    print(f"Generated {generated.shape[2]} samples")
    print("\n✓ WaveNet can generate realistic audio!")
```

**Key Features:**

1. **Autoregressive:** Generates one sample at a time
2. **Causal:** Only uses past samples (no cheating!)
3. **Large receptive field:** Can capture long-term dependencies
4. **High quality:** Near human-level speech synthesis

**Limitations:**
- **Very slow:** Must generate sequentially (can't parallelize)
- **44,100 samples/second** = slow generation
- **Solution:** Parallel WaveNet (teacher-student distillation)

**Applications:**
- Google Assistant voice
- Text-to-speech systems
- Music generation
- Sound effects

---

## Slide 18: Text-to-Speech (TTS)

### From Text to Natural Speech

**TTS Pipeline:**

```
Text → Text Processing → Acoustic Model → Vocoder → Audio
  ↓           ↓               ↓              ↓
"Hello"   Phonemes      Mel-spectrogram   Waveform
```

**Modern TTS: Tacotron 2 + WaveNet**

**1. Tacotron 2 (Text → Mel-Spectrogram)**

```python
class Tacotron2(nn.Module):
    """
    Text-to-Mel-Spectrogram model.
    
    Architecture:
        Text → Encoder → Attention → Decoder → Mel-Spectrogram
    """
    
    def __init__(self, vocab_size, hidden_dim=512):
        super().__init__()
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Encoder: Text → Hidden states
        self.encoder = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Decoder: Generate mel-spectrogram
        self.decoder_lstm = nn.LSTM(
            hidden_dim + 80,  # hidden + mel
            hidden_dim,
            batch_first=True
        )
        
        # Mel predictor
        self.mel_linear = nn.Linear(hidden_dim, 80)  # 80 mel bands
        
        # Stop token predictor
        self.stop_linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, text):
        """
        Generate mel-spectrogram from text.
        
        Args:
            text: Character indices
        
        Returns:
            mel: Mel-spectrogram
            stop_tokens: When to stop generation
        """
        # Encode text
        embedded = self.embedding(text)
        encoded, _ = self.encoder(embedded)
        
        # Decode with attention
        mel_outputs = []
        stop_tokens = []
        
        # Start with zeros
        decoder_input = torch.zeros(text.size(0), 1, 80).to(text.device)
        
        for _ in range(1000):  # Max length
            # Attend to encoder outputs
            context, _ = self.attention(
                decoder_input,
                encoded,
                encoded
            )
            
            # Decode
            decoder_output, _ = self.decoder_lstm(
                torch.cat([context, decoder_input], dim=2)
            )
            
            # Predict mel frame
            mel_frame = self.mel_linear(decoder_output)
            mel_outputs.append(mel_frame)
            
            # Predict stop token
            stop_prob = torch.sigmoid(self.stop_linear(decoder_output))
            stop_tokens.append(stop_prob)
            
            # Use predicted mel as next input
            decoder_input = mel_frame
            
            # Stop if stop token predicted
            if stop_prob > 0.5:
                break
        
        mel = torch.cat(mel_outputs, dim=1)
        stops = torch.cat(stop_tokens, dim=1)
        
        return mel, stops


# Complete TTS pipeline
class TTSSystem:
    """
    Complete Text-to-Speech system.
    
    Combines Tacotron 2 (text→mel) with WaveNet (mel→audio).
    """
    
    def __init__(self):
        self.tacotron = Tacotron2(vocab_size=128)
        self.wavenet = SimpleWaveNet()
    
    def synthesize(self, text):
        """
        Convert text to audio.
        
        Args:
            text: Input text string
        
        Returns:
            audio: Generated waveform
        """
        # Text to character indices
        char_indices = torch.tensor([ord(c) for c in text]).unsqueeze(0)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            mel, _ = self.tacotron(char_indices)
        
        # Generate waveform
        with torch.no_grad():
            audio = self.wavenet.generate_from_mel(mel)
        
        return audio


# Usage
tts = TTSSystem()
audio = tts.synthesize("Hello, world!")
```

**2. Modern Alternative: FastSpeech 2**
- **Non-autoregressive** (much faster!)
- Predicts duration explicitly
- Generates entire mel-spec in parallel
- 50x faster than Tacotron 2

**Real-World TTS APIs:**

```python
# Google Cloud Text-to-Speech
from google.cloud import texttospeech

def google_tts(text, voice_name="en-US-Neural2-A"):
    """
    Use Google Cloud TTS.
    
    Args:
        text: Text to synthesize
        voice_name: Voice to use
    
    Returns:
        audio: Generated speech
    """
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_name
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,
        pitch=0.0
    )
    
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    
    return response.audio_content


# ElevenLabs (High-quality, realistic voices)
import elevenlabs

def elevenlabs_tts(text, voice_id="default"):
    """
    Use ElevenLabs for realistic TTS.
    
    Very high quality, natural sounding.
    """
    audio = elevenlabs.generate(
        text=text,
        voice=voice_id,
        model="eleven_monolingual_v1"
    )
    
    return audio
```

**Business Applications:**

**Content Creation:**
```python
# Generate audiobook from text
chapters = load_book_chapters()
for i, chapter in enumerate(chapters):
    audio = tts.synthesize(chapter)
    save_audio(audio, f"chapter_{i}.mp3")

# ROI: $200/hour voice actor → $5 AI generation
# For 10-hour book: $2000 → $50 (97.5% cost reduction)
```

**Accessibility:**
```python
# Real-time website reader
def make_accessible(webpage_text):
    audio = tts.synthesize(webpage_text)
    return audio

# Market: 285M visually impaired people worldwide
# Business opportunity: Accessibility-as-a-service
```

**Virtual Assistants:**
```python
# Dynamic responses
user_query = "What's the weather?"
response = get_weather_info()
audio = tts.synthesize(response)
play_audio(audio)

# Use: Customer service, education, gaming
```

---

## Slide 19: Voice Cloning

### Creating Custom Voices

**Voice Cloning:** Generate speech in someone's voice using minimal audio samples.

**How It Works:**

**1. Speaker Embedding**
- Extract unique characteristics of a voice
- Encode into a vector (embedding)
- Capture pitch, tone, accent, style

**2. Multi-Speaker TTS**
- Train model on many speakers
- Condition on speaker embedding
- Can generalize to new voices

**Architecture:**

```
Reference Audio → Speaker Encoder → Speaker Embedding
                                           ↓
Text Input → Tacotron 2 (conditioned on embedding) → Mel-Spec → WaveNet → Audio
```

**Implementation:**

```python
class SpeakerEncoder(nn.Module):
    """
    Extract speaker embedding from audio.
    
    Maps audio to a fixed-size vector representing speaker identity.
    """
    
    def __init__(self, mel_dim=80, embedding_dim=256):
        super().__init__()
        
        # Process mel-spectrogram
        self.lstm = nn.LSTM(
            mel_dim,
            256,
            num_layers=3,
            batch_first=True
        )
        
        # Project to embedding
        self.linear = nn.Linear(256, embedding_dim)
    
    def forward(self, mel):
        """
        Extract speaker embedding.
        
        Args:
            mel: Mel-spectrogram of reference audio
        
        Returns:
            embedding: Speaker embedding vector
        """
        _, (hidden, _) = self.lstm(mel)
        embedding = self.linear(hidden[-1])
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class MultiSpeakerTTS(nn.Module):
    """
    TTS system that can clone voices.
    
    Conditions generation on speaker embedding.
    """
    
    def __init__(self, vocab_size, embedding_dim=256):
        super().__init__()
        
        self.speaker_encoder = SpeakerEncoder(embedding_dim=embedding_dim)
        self.tacotron = Tacotron2(vocab_size)
        
        # Modify Tacotron to accept speaker embedding
        # (inject embedding into decoder)
    
    def clone_voice(self, text, reference_audio):
        """
        Generate speech in voice of reference audio.
        
        Args:
            text: Text to synthesize
            reference_audio: Audio sample of target voice
        
        Returns:
            audio: Generated speech in cloned voice
        """
        # Extract speaker embedding from reference
        mel_ref = compute_mel(reference_audio)
        speaker_embedding = self.speaker_encoder(mel_ref)
        
        # Generate with speaker conditioning
        mel = self.tacotron(text, speaker_embedding)
        audio = self.wavenet(mel)
        
        return audio


# Real-world voice cloning APIs
def use_voice_cloning(text, voice_sample_path):
    """
    Use commercial voice cloning service.
    
    Example: ElevenLabs, Resemble.ai, Descript
    """
    # ElevenLabs voice cloning
    voice = elevenlabs.clone(
        name="Custom Voice",
        description="Cloned from sample",
        files=[voice_sample_path]
    )
    
    audio = elevenlabs.generate(
        text=text,
        voice=voice
    )
    
    return audio
```

**Use Cases:**

**1. Content Localization**
```python
# Translate video while keeping original voice
original_audio = extract_audio("video.mp4")
translated_text = translate(transcribe(original_audio), target_lang="es")

# Generate Spanish audio in original speaker's voice
spanish_audio = voice_clone(translated_text, reference=original_audio)

# ROI: $50/minute professional dubbing → $1 AI dubbing
# For 90-min movie: $4500 → $90 (98% savings)
```

**2. Personalized Experiences**
```python
# Audiobook in user's favorite celebrity voice
# (with permission/licensing)
book_text = load_book()
celebrity_sample = load_voice_sample("celebrity_authorized.wav")
audiobook = voice_clone(book_text, celebrity_sample)

# Market: Premium audiobooks, personalized content
```

**3. Accessibility**
```python
# Restore voice for people who lost ability to speak
# Use recordings from before illness
patient_recordings = load_patient_voice()
speech_text = get_aac_device_text()  # AAC device input
spoken_output = voice_clone(speech_text, patient_recordings)

# Impact: Restore personal identity, emotional connection
```

**Ethical Considerations:**

⚠️ **Concerns:**
- **Deepfakes:** Impersonation, fraud
- **Consent:** Using someone's voice without permission
- **Misinformation:** Fake audio of public figures

✅ **Safeguards:**
- Watermarking generated audio
- Detection systems for deepfakes
- Legal frameworks for consent
- Responsible use policies

---

## Slide 20: Audio Classification & Understanding

### Beyond Generation - Understanding Audio

**Audio Classification Tasks:**

**1. Speech Recognition (ASR)**
- Convert speech to text
- Used in: Voice assistants, transcription

**2. Speaker Identification**
- Who is speaking?
- Used in: Security, organization

**3. Emotion Recognition**
- Detect speaker's emotional state
- Used in: Call centers, mental health

**4. Sound Event Detection**
- Identify sounds (dog bark, car horn, etc.)
- Used in: Smart homes, surveillance

**5. Music Genre Classification**
- Classify music by genre
- Used in: Music platforms, recommendation

**Audio Classification Model:**

```python
class AudioClassifier(nn.Module):
    """
    General audio classification model.
    
    Uses CNN on mel-spectrogram for classification.
    """
    
    def __init__(self, n_classes, n_mels=128):
        super().__init__()
        
        # CNN for processing spectrogram
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, mel_spec):
        """
        Classify audio from mel-spectrogram.
        
        Args:
            mel_spec: Mel-spectrogram (batch, 1, n_mels, time)
        
        Returns:
            logits: Class predictions
        """
        x = self.conv_layers(mel_spec)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


# Business application: Call center quality monitoring
class CallCenterMonitor:
    """
    Monitor call center conversations for quality.
    
    Detects emotions, keywords, compliance issues.
    """
    
    def __init__(self):
        self.emotion_classifier = AudioClassifier(n_classes=7)  # 7 emotions
        self.keyword_detector = KeywordSpotter()
        self.sentiment_analyzer = SentimentModel()
    
    def analyze_call(self, audio_path):
        """
        Comprehensive call analysis.
        
        Args:
            audio_path: Path to call recording
        
        Returns:
            report: Analysis report
        """
        # Load audio
        audio, sr = librosa.load(audio_path)
        mel = compute_mel(audio)
        
        # Detect emotions
        emotions = self.emotion_classifier(mel)
        dominant_emotion = emotions.argmax()
        
        # Detect keywords (compliance, escalation triggers)
        keywords_found = self.keyword_detector(audio)
        
        # Overall sentiment
        sentiment = self.sentiment_analyzer(audio)
        
        report = {
            'duration': len(audio) / sr,
            'dominant_emotion': dominant_emotion,
            'keywords': keywords_found,
            'sentiment': sentiment,
            'quality_score': self.compute_quality(emotions, keywords_found, sentiment)
        }
        
        return report
    
    def compute_quality(self, emotions, keywords, sentiment):
        """Calculate overall call quality score"""
        # Positive emotions + no escalation keywords + positive sentiment = high quality
        quality = (
            0.4 * (sentiment > 0.5) +
            0.3 * (dominant_emotion in [0, 5, 6]) +  # happy, calm, satisfied
            0.3 * (1 - len(keywords) / 10)  # fewer issues = better
        )
        return quality


# ROI Example
def call_center_roi():
    """
    Calculate ROI of AI call monitoring.
    """
    traditional = {
        'human_qa_analysts': 10,
        'salary_per_analyst': 50000,
        'calls_reviewed_per_analyst': 100,  # per month
        'total_cost': 10 * 50000  # $500k/year
    }
    
    ai_system = {
        'setup_cost': 50000,  # One-time
        'api_costs': 10000,  # per year
        'calls_reviewed': 'unlimited',  # 100% of calls
        'total_cost': 60000  # first year
    }
    
    savings = traditional['total_cost'] - ai_system['total_cost']
    improvement = "100% coverage vs 1% coverage"
    
    print(f"First Year Savings: ${savings:,}")
    print(f"Coverage Improvement: {improvement}")
    print(f"ROI: {(savings / ai_system['total_cost']) * 100:.1f}%")
    
    return {
        'savings': savings,
        'roi_percent': (savings / ai_system['total_cost']) * 100
    }


if __name__ == "__main__":
    roi = call_center_roi()
    print("\n" + "="*70)
    print("CALL CENTER AI MONITORING ROI")
    print("="*70)
    print(f"Annual Savings: ${roi['savings']:,}")
    print(f"ROI: {roi['roi_percent']:.1f}%")
    print("Additional Benefits:")
    print("  • 100% call coverage (vs 1% manual)")
    print("  • Real-time insights")
    print("  • Compliance monitoring")
    print("  • Agent coaching opportunities")
```

**Additional Business Applications:**

**Music Recommendation:**
- Analyze audio features
- Classify genre, mood, tempo
- Personalized playlists

**Smart Home:**
- Detect specific sounds (glass breaking, baby crying)
- Trigger appropriate responses
- Enhanced security

**Healthcare:**
- Detect breathing abnormalities
- Monitor cough patterns
- Early disease detection

---

**End of Batch 3 (Slides 16-20)**

*Continue to Batch 4 for Advanced Music Generation (Slides 21-25)*
