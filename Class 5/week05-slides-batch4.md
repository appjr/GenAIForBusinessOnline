# Week 5: Image Generation, Audio, and Music - Slides Batch 4 (Slides 21-25)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Continuation from Batch 3**

---

## Slide 21: Music Generation - Special Challenges

### Music vs Speech - What Makes Music Different?

**Key Differences:**

| Aspect | Speech | Music |
|--------|--------|-------|
| **Structure** | Sequential, linear | Harmonic, polyphonic |
| **Elements** | Phonemes, words | Notes, chords, rhythm |
| **Complexity** | 1 voice | Multiple instruments |
| **Duration** | Seconds to minutes | Minutes to hours |
| **Evaluation** | Intelligibility | Aesthetic quality |

**Music Representations:**

**1. Audio Waveform**
- Raw audio signal
- Pros: Complete information
- Cons: Very high-dimensional, hard to edit

**2. MIDI (Symbolic)**
- Musical notes with timing and velocity
- Pros: Compact, editable, interpretable
- Cons: No audio timbre information

**3. Piano Roll**
- Visual representation of MIDI
- Time × Pitch grid
- Pros: Easy to visualize and edit
- Cons: Limited to pitch information

**4. Spectrogram**
- Time-frequency representation
- Pros: Works for any audio
- Cons: Loses some structure

**Music Generation Approaches:**

```python
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

class MusicRepresentation:
    """
    Handle different music representations.
    
    Convert between MIDI, piano roll, and audio.
    """
    
    def __init__(self, fs=100):
        """
        Args:
            fs: Frames per second for piano roll
        """
        self.fs = fs
    
    def midi_to_piano_roll(self, midi_file):
        """
        Convert MIDI to piano roll representation.
        
        Args:
            midi_file: Path to MIDI file
        
        Returns:
            piano_roll: (128, time_steps) array
        """
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        piano_roll = midi_data.get_piano_roll(fs=self.fs)
        return piano_roll
    
    def piano_roll_to_midi(self, piano_roll, program=0):
        """
        Convert piano roll back to MIDI.
        
        Args:
            piano_roll: (128, time_steps) array
            program: MIDI instrument program
        
        Returns:
            midi_data: PrettyMIDI object
        """
        midi_data = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)
        
        # Find note events
        for pitch in range(128):
            note_changes = np.diff(piano_roll[pitch] > 0, prepend=0, append=0)
            note_on = np.where(note_changes == 1)[0]
            note_off = np.where(note_changes == -1)[0]
            
            for start, end in zip(note_on, note_off):
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start / self.fs,
                    end=end / self.fs
                )
                instrument.notes.append(note)
        
        midi_data.instruments.append(instrument)
        return midi_data
    
    def visualize_piano_roll(self, piano_roll, title="Piano Roll"):
        """Visualize piano roll"""
        plt.figure(figsize=(12, 6))
        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='Blues')
        plt.xlabel('Time (frames)')
        plt.ylabel('Pitch')
        plt.title(title)
        plt.colorbar(label='Velocity')
        plt.tight_layout()
        return plt.gcf()


# Usage example
if __name__ == "__main__":
    music_repr = MusicRepresentation(fs=100)
    
    # Load MIDI
    piano_roll = music_repr.midi_to_piano_roll('sample.mid')
    print(f"Piano roll shape: {piano_roll.shape}")
    print(f"Duration: {piano_roll.shape[1] / 100:.2f} seconds")
    
    # Visualize
    fig = music_repr.visualize_piano_roll(piano_roll)
    plt.savefig('piano_roll.png', dpi=300)
    plt.show()
    
    # Convert back
    midi_out = music_repr.piano_roll_to_midi(piano_roll)
    midi_out.write('reconstructed.mid')
```

**Business Relevance:**
- Background music for videos
- Game soundtracks
- Retail ambiance
- Personalized playlists
- Music therapy

---

## Slide 22: Music RNN - Simple Melody Generation

### Generating Melodies with Recurrent Networks

**Approach:** Treat music as a sequence, like text generation.

**Music RNN Architecture:**

```
Previous Notes → LSTM → Next Note Distribution
```

**Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicLSTM(nn.Module):
    """
    LSTM for melody generation.
    
    Predicts next note given previous notes.
    Similar to language model but for music.
    """
    
    def __init__(self, vocab_size=128, embedding_dim=128, hidden_dim=512, num_layers=2):
        """
        Args:
            vocab_size: Number of possible notes (88 for piano, 128 for MIDI)
            embedding_dim: Note embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Note embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Note sequence (batch, seq_len)
            hidden: Optional hidden state
        
        Returns:
            logits: Next note predictions
            hidden: Updated hidden state
        """
        # Embed notes
        embedded = self.embedding(x)
        
        # LSTM
        output, hidden = self.lstm(embedded, hidden)
        
        # Project to vocabulary
        logits = self.fc(output)
        
        return logits, hidden
    
    @torch.no_grad()
    def generate(self, start_sequence, length=100, temperature=1.0):
        """
        Generate melody autoregressively.
        
        Args:
            start_sequence: Seed notes (batch, seq_len)
            length: Number of notes to generate
            temperature: Sampling temperature
        
        Returns:
            generated: Generated note sequence
        """
        self.eval()
        generated = start_sequence
        hidden = None
        
        for _ in range(length):
            # Predict next note
            logits, hidden = self.forward(generated[:, -1:], hidden)
            
            # Sample from distribution
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_note = torch.multinomial(probs, 1)
            
            # Append
            generated = torch.cat([generated, next_note], dim=1)
        
        return generated


def train_music_lstm(model, sequences, epochs=10):
    """
    Train music generation model.
    
    Args:
        model: MusicLSTM model
        sequences: Training sequences (list of note sequences)
        epochs: Number of training epochs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("="*70)
    print("TRAINING MUSIC LSTM")
    print("="*70)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for sequence in sequences:
            # Prepare input and target
            x = sequence[:-1]
            y = sequence[1:]
            
            # Forward pass
            logits, _ = model(x.unsqueeze(0))
            loss = criterion(logits.reshape(-1, model.vocab_size), y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(sequences)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


# Complete example
if __name__ == "__main__":
    # Create model
    model = MusicLSTM(vocab_size=88, embedding_dim=128, hidden_dim=512)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load training data (simplified)
    sequences = load_midi_sequences('music_dataset/')
    
    # Train
    model = train_music_lstm(model, sequences, epochs=20)
    
    # Generate new melody
    print("\nGenerating new melody...")
    start = torch.tensor([[60, 62, 64]])  # C, D, E
    generated = model.generate(start, length=100, temperature=1.0)
    
    # Convert to MIDI
    notes_to_midi(generated[0], 'generated_melody.mid')
    print("✓ Melody saved to 'generated_melody.mid'")
```

**Magenta (Google's Music Generation)**

```python
# Using Magenta's pre-trained models
import magenta
from magenta.models.melody_rnn import melody_rnn_model

def generate_with_magenta(primer_melody, steps=128):
    """
    Generate melody using Magenta's pre-trained model.
    
    Args:
        primer_melody: Starting melody
        steps: Number of steps to generate
    
    Returns:
        generated_sequence: Generated melody
    """
    # Load pre-trained model
    bundle = magenta.music.read_bundle_file('attention_rnn.mag')
    config = melody_rnn_model.default_configs['attention_rnn']
    
    # Generate
    generator = melody_rnn_model.MelodyRnnModel(config)
    generator.initialize(bundle)
    
    generated_sequence = generator.generate(
        input_sequence=primer_melody,
        generator_options=magenta.music.GeneratorOptions(
            temperature=1.0,
            generate_sections=[
                magenta.music.GeneratorSection(
                    start_time=0,
                    end_time=steps
                )
            ]
        )
    )
    
    return generated_sequence


# Example usage
primer = create_primer_melody([60, 62, 64, 65])  # C, D, E, F
melody = generate_with_magenta(primer, steps=128)
save_sequence_to_midi(melody, 'magenta_melody.mid')
```

**Limitations:**
- Single melody line (monophonic)
- Limited long-term structure
- No harmonic awareness
- Repetitive patterns

---

## Slide 23: Music Transformer - Polyphonic Generation

### Multiple Instruments and Harmony

**Music Transformer:** Apply transformer architecture to music generation.

**Key Innovation:** Self-attention for capturing musical relationships.

**Architecture:**

```
MIDI Events → Token Embedding → Positional Encoding
                                        ↓
                                  Transformer Blocks
                                        ↓
                                  Next Event Prediction
```

**Music Tokenization:**

```python
class MusicTokenizer:
    """
    Tokenize music for transformer input.
    
    Represents MIDI events as discrete tokens.
    """
    
    def __init__(self):
        # Token types
        self.note_on_offset = 0
        self.note_off_offset = 128
        self.time_shift_offset = 256
        self.velocity_offset = 356
        
        self.vocab_size = 456  # Total tokens
    
    def encode_event(self, event_type, value):
        """
        Encode MIDI event as token.
        
        Args:
            event_type: 'note_on', 'note_off', 'time_shift', 'velocity'
            value: Event value
        
        Returns:
            token: Integer token
        """
        if event_type == 'note_on':
            return self.note_on_offset + value
        elif event_type == 'note_off':
            return self.note_off_offset + value
        elif event_type == 'time_shift':
            return self.time_shift_offset + min(value, 100)
        elif event_type == 'velocity':
            return self.velocity_offset + min(value // 4, 32)
    
    def decode_token(self, token):
        """Decode token back to MIDI event"""
        if token < 128:
            return ('note_on', token)
        elif token < 256:
            return ('note_off', token - 128)
        elif token < 356:
            return ('time_shift', token - 256)
        else:
            return ('velocity', (token - 356) * 4)
    
    def midi_to_tokens(self, midi_file):
        """Convert MIDI file to token sequence"""
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        tokens = []
        
        # Sort all events by time
        events = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                events.append(('note_on', note.start, note.pitch, note.velocity))
                events.append(('note_off', note.end, note.pitch, 0))
        
        events.sort(key=lambda x: x[1])
        
        # Convert to tokens
        current_time = 0
        for event_type, time, pitch, velocity in events:
            # Time shift
            time_diff = int((time - current_time) * 100)
            if time_diff > 0:
                tokens.append(self.encode_event('time_shift', time_diff))
            
            # Velocity (for note_on)
            if event_type == 'note_on':
                tokens.append(self.encode_event('velocity', velocity))
            
            # Note event
            tokens.append(self.encode_event(event_type, pitch))
            
            current_time = time
        
        return tokens


class MusicTransformer(nn.Module):
    """
    Transformer for polyphonic music generation.
    
    Handles multiple instruments and complex harmonies.
    """
    
    def __init__(self, vocab_size=456, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(10000, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Token sequence (batch, seq_len)
        
        Returns:
            logits: Next token predictions
        """
        seq_len = x.size(1)
        
        # Embed and add position
        embedded = self.embedding(x)
        embedded = embedded + self.pos_encoder[:seq_len]
        
        # Create causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer
        output = self.transformer(embedded, mask=mask, is_causal=True)
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        return logits
    
    @torch.no_grad()
    def generate(self, start_tokens, length=1000, temperature=1.0, top_k=40):
        """
        Generate music autoregressively.
        
        Args:
            start_tokens: Seed tokens
            length: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            generated: Generated token sequence
        """
        self.eval()
        generated = start_tokens
        
        for _ in range(length):
            # Predict next token
            logits = self.forward(generated)[:, -1, :]
            
            # Temperature and top-k sampling
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# Training and generation
if __name__ == "__main__":
    print("="*70)
    print("MUSIC TRANSFORMER")
    print("="*70)
    
    # Initialize
    tokenizer = MusicTokenizer()
    model = MusicTransformer(vocab_size=tokenizer.vocab_size)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load dataset
    midi_files = glob.glob('midi_dataset/*.mid')
    token_sequences = [tokenizer.midi_to_tokens(f) for f in midi_files]
    
    print(f"Loaded {len(token_sequences)} MIDI files")
    
    # Train (simplified)
    train_music_transformer(model, token_sequences, epochs=20)
    
    # Generate new composition
    print("\nGenerating new music...")
    start_tokens = torch.tensor([[0, 256, 360, 60]])  # Simple start
    generated_tokens = model.generate(start_tokens, length=1000, temperature=1.0)
    
    # Convert back to MIDI
    midi_out = tokenizer.tokens_to_midi(generated_tokens[0])
    midi_out.write('generated_composition.mid')
    
    print("✓ Composition saved to 'generated_composition.mid'")
```

**Advantages:**
- ✅ Polyphonic (multiple notes simultaneously)
- ✅ Long-range dependencies
- ✅ Harmonically aware
- ✅ Multiple instruments

---

## Slide 24: Modern Music Generation - MuseNet & MusicLM

### State-of-the-Art Music AI

**MuseNet (OpenAI, 2019)**

Large-scale transformer trained on MIDI data.

**Capabilities:**
- 4-minute compositions
- 10 different instruments
- Multiple genres (classical, jazz, pop, etc.)
- Can continue any music snippet

**Architecture:**
- 72-layer transformer
- 24 attention heads
- Trained on millions of MIDI files

**Usage (API):**

```python
import openai

def generate_with_musenet(prompt, genre='classical', instruments=['piano']):
    """
    Generate music with MuseNet API.
    
    Args:
        prompt: Starting MIDI sequence
        genre: Music genre
        instruments: List of instruments
    
    Returns:
        audio: Generated audio
    """
    response = openai.MuseNet.create(
        prompt=prompt,
        genre=genre,
        instruments=instruments,
        length=240  # seconds
    )
    
    return response['audio']


# Example
prompt_midi = load_midi('seed.mid')
music = generate_with_musenet(
    prompt=prompt_midi,
    genre='jazz',
    instruments=['piano', 'bass', 'drums']
)
save_audio(music, 'jazz_composition.mp3')
```

**MusicLM (Google, 2023)**

Text-to-music generation - like DALL-E but for music!

**Key Innovation:** Generate music from text descriptions.

**Architecture:**
```
Text Description → Text Encoder (BERT)
                         ↓
                   Conditioning Vector
                         ↓
Audio Tokens ← AudioLM (music language model)
                         ↓
                    Waveform
```

**Examples:**

```python
# MusicLM-style text-to-music
def text_to_music(description, duration=30):
    """
    Generate music from text description.
    
    Args:
        description: Text describing desired music
        duration: Length in seconds
    
    Returns:
        audio: Generated music
    """
    # Example descriptions:
    descriptions = [
        "A calming violin melody backed by a distorted guitar riff",
        "Meditative song, beautiful mix of violin, cello, and piano",
        "Fast techno dance track with a heavy kick and high-pitched melody",
        "Jazz fusion with saxophone and electric guitar"
    ]
    
    # Would use actual MusicLM API
    audio = generate_music_from_text(description, duration)
    
    return audio


# Business application
def create_custom_soundtrack(scene_description):
    """
    Create custom music for video scenes.
    
    Args:
        scene_description: Description of scene mood/action
    
    Returns:
        soundtrack: Custom-generated music
    """
    # Map scene to music description
    music_prompts = {
        'action': "Fast-paced electronic music with driving drums",
        'romantic': "Soft piano melody with string accompaniment",
        'suspense': "Dark ambient with low strings and subtle percussion",
        'comedy': "Playful upbeat tune with quirky instruments"
    }
    
    scene_type = classify_scene(scene_description)
    prompt = music_prompts.get(scene_type, "Neutral background music")
    
    soundtrack = text_to_music(prompt, duration=60)
    
    return soundtrack


# ROI Example
def video_production_roi():
    """Calculate ROI for AI music generation"""
    traditional = {
        'composer_fee': 5000,
        'recording_session': 3000,
        'licensing': 2000,
        'total_per_video': 10000,
        'time': '2 weeks'
    }
    
    ai_approach = {
        'api_cost': 50,
        'editing': 200,
        'total_per_video': 250,
        'time': '1 hour'
    }
    
    savings = traditional['total_per_video'] - ai_approach['total_per_video']
    
    print(f"Savings per video: ${savings} (97.5% reduction)")
    print(f"For 100 videos/year: ${savings * 100:,} saved")
    print(f"Time savings: 2 weeks → 1 hour per video")
    
    return savings
```

---

## Slide 25: Music Style Transfer & Applications

### Transforming Musical Styles

**Music Style Transfer:** Convert music from one style to another while preserving melody.

**Examples:**
- Classical → Jazz
- Pop → Rock
- Piano → Orchestra

**Approach 1: CycleGAN for Music**

```python
class MusicStyleGAN:
    """
    Style transfer for music using CycleGAN approach.
    
    Learns to translate between two musical styles.
    """
    
    def __init__(self, style_a='classical', style_b='jazz'):
        self.style_a = style_a
        self.style_b = style_b
        
        # Generators: A→B and B→A
        self.g_ab = MusicGenerator()
        self.g_ba = MusicGenerator()
        
        # Discriminators
        self.d_a = MusicDiscriminator()
        self.d_b = MusicDiscriminator()
    
    def transfer_style(self, audio, source_style, target_style):
        """
        Transfer music from source to target style.
        
        Args:
            audio: Input audio
            source_style: Original style
            target_style: Desired style
        
        Returns:
            transformed: Audio in target style
        """
        # Convert to spectrogram
        spec = audio_to_spectrogram(audio)
        
        # Apply style transfer
        if source_style == self.style_a and target_style == self.style_b:
            spec_transformed = self.g_ab(spec)
        else:
            spec_transformed = self.g_ba(spec)
        
        # Convert back to audio
        audio_transformed = spectrogram_to_audio(spec_transformed)
        
        return audio_transformed


# Real-world application
def adaptive_background_music(activity_type, intensity_level):
    """
    Generate adaptive music for fitness/gaming apps.
    
    Args:
        activity_type: Type of activity (running, yoga, gaming)
        intensity_level: 0-10 intensity scale
    
    Returns:
        music: Dynamically generated/adapted music
    """
    # Base track selection
    base_tracks = {
        'running': load_track('upbeat_electronic.mid'),
        'yoga': load_track('calm_ambient.mid'),
        'gaming': load_track('action_soundtrack.mid')
    }
    
    base = base_tracks[activity_type]
    
    # Adapt based on intensity
    if intensity_level < 3:
        # Low intensity: Calm, slow
        music = adapt_tempo(base, target_bpm=80)
        music = style_transfer(music, target='ambient')
    elif intensity_level < 7:
        # Medium intensity
        music = adapt_tempo(base, target_bpm=120)
    else:
        # High intensity: Fast, energetic
        music = adapt_tempo(base, target_bpm=160)
        music = add_intensity(music, level=intensity_level)
    
    return music


# Business applications
def music_application_examples():
    """
    Real-world music AI applications with ROI.
    """
    
    applications = {
        'video_game_music': {
            'description': 'Dynamic adaptive soundtracks',
            'traditional_cost': 50000,  # Composer + recording
            'ai_cost': 2000,
            'roi': 96,
            'use_case': 'Generate infinite variations based on gameplay'
        },
        
        'retail_ambiance': {
            'description': 'Custom store background music',
            'traditional_cost': 500,  # Monthly licensing
            'ai_cost': 50,
            'roi': 90,
            'use_case': 'Generate brand-specific music continuously'
        },
        
        'fitness_apps': {
            'description': 'Workout-adaptive music',
            'traditional_cost': 10000,  # Licensed tracks
            'ai_cost': 500,
            'roi': 95,
            'use_case': 'Match music tempo to heart rate/activity'
        },
        
        'meditation_apps': {
            'description': 'Personalized soundscapes',
            'traditional_cost': 5000,
            'ai_cost': 200,
            'roi': 96,
            'use_case': 'Generate calming music customized to preferences'
        },
        
        'content_creation': {
            'description': 'YouTube/podcast background music',
            'traditional_cost': 200,  # per video licensing
            'ai_cost': 5,
            'roi': 97.5,
            'use_case': 'Generate royalty-free custom music for each video'
        }
    }
    
    print("="*70)
    print("MUSIC AI BUSINESS APPLICATIONS")
    print("="*70)
    
    for app_name, details in applications.items():
        savings = details['traditional_cost'] - details['ai_cost']
        roi = details['roi']
        
        print(f"\n{app_name.upper().replace('_', ' ')}")
        print(f"  Use Case: {details['use_case']}")
        print(f"  Traditional Cost: ${details['traditional_cost']:,}")
        print(f"  AI Cost: ${details['ai_cost']:,}")
        print(f"  Savings: ${savings:,} ({roi}%)")


if __name__ == "__main__":
    music_application_examples()
```

**Music AI Market:**
- **Size:** $1.5B in 2023 → $5B by 2028
- **Growth:** 30% CAGR
- **Key Players:** Spotify, YouTube, TikTok, gaming companies

**Ethical Considerations:**
- Copyright and ownership
- Musician displacement concerns
- Authenticity vs AI-generated
- Fair compensation models

**Best Practices:**
- Transparent AI usage disclosure
- Hybrid human-AI workflows
- Support for human musicians
- Responsible licensing

---

**End of Batch 4 (Slides 21-25)**

*Continue to Batch 5 for Advanced Topics (Slides 26-30)*
