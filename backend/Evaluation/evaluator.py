import torch
import librosa
import numpy as np
import laion_clap
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)

class AudioEvaluator:
    def __init__(self):
        # Initialize CLAP (Requires weights download on first run)
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt() 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_clap_score(self, audio_path, text_prompt):
        """Measures Text-to-Audio Alignment (Higher is better)"""
        audio_embed = self.clap_model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
        text_embed = self.clap_model.get_text_embedding([text_prompt], use_tensor=True)
        
        similarity = torch.nn.functional.cosine_similarity(audio_embed, text_embed)
        return similarity.item()

    def get_audio_richness(self, audio_path):
        """Measures Spectral Flatness and Entropy (Proxies for quality/complexity)"""
        y, sr = librosa.load(audio_path)
        
        # Spectral Flatness: 1.0 = white noise, 0.0 = pure tone. 
        # We want a mid-range for complex scores.
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # Spectral Entropy: How 'unpredictable' the sound is.
        S = np.abs(librosa.stft(y))
        psd = np.sum(S**2, axis=1)
        psd /= np.sum(psd)
        spec_entropy = entropy(psd)
        
        return flatness, spec_entropy

    def evaluate_sync(self, audio_path, action_keywords=["shot", "bang", "crash", "door"]):
        """Check if peaks exist in audio (simple onset detection)"""
        y, sr = librosa.load(audio_path)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        # Returns number of sharp transients found
        return len(onsets)

# --- EXECUTION ---
evaluator = AudioEvaluator()

# Inputs
test_audio = "generated_score_01.wav"
test_prompt = "A high-speed car chase ending in a loud crash."

# 1. Alignment Check
c_score = evaluator.get_clap_score(test_audio, test_prompt)

# 2. Quality Check
flat, ent = evaluator.get_audio_richness(test_audio)

# 3. Structural Check
peak_count = evaluator.evaluate_sync(test_audio)

logger.info(f"""
--- Evaluation Results (No Ground Truth) ---
Text-Audio Alignment (CLAP): {c_score:.4f}  (Target: >0.25 for good match)
Spectral Entropy (Richness): {ent:.4f}      (Target: Higher = more complex music)
Spectral Flatness (Noise):   {flat:.4f}     (Target: Lower = more tonal/musical)
Detected Audio Onsets:       {peak_count}   (Number of dynamic events)
""")