"""
Audio Processing Utilities
From: week05-slides-batch3.md - Slide 16

Load, save, and visualize audio files.
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

class AudioProcessor:
    """Utilities for audio processing and visualization"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, filename):
        """Load audio file"""
        waveform, sr = librosa.load(filename, sr=self.sample_rate)
        return waveform, sr
    
    def save_audio(self, waveform, filename):
        """Save audio to file"""
        waveform = np.int16(waveform * 32767)
        wavfile.write(filename, self.sample_rate, waveform)
        print(f"✓ Saved audio to {filename}")
    
    def compute_spectrogram(self, waveform):
        """Compute mel-spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=128,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def visualize_audio(self, waveform, mel_spec=None):
        """Visualize waveform and spectrogram"""
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


if __name__ == "__main__":
    # Demo with synthetic audio
    processor = AudioProcessor(sample_rate=22050)
    
    # Generate test tone
    duration = 2.0
    t = np.linspace(0, duration, int(processor.sample_rate * duration))
    frequency = 440  # A4 note
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Save
    processor.save_audio(waveform, 'test_tone.wav')
    
    # Visualize
    fig = processor.visualize_audio(waveform)
    plt.savefig('audio_visualization.png', dpi=300)
    print("✓ Saved visualization to audio_visualization.png")
    plt.show()
