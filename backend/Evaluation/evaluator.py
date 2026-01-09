import base64
import io
import torch
import librosa
import numpy as np
import laion_clap
from scipy.stats import entropy
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

class AudioEvaluator:
    def __init__(self):
        # Initialize CLAP (Requires weights download on first run)
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt() 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _base64_to_temp_file(self, audio_base64):
        """Convert base64 audio to a temporary file and return the path"""
        # Handle data URL format (data:audio/wav;base64,...)
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_base64)
        
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                temp_file.write(audio_bytes)
            return temp_path
        except Exception as e:
            os.close(temp_fd)
            raise e
    
    def get_clap_score(self, audio_base64, text_prompt):
        """Measures Text-to-Audio Alignment (Higher is better) from base64 audio"""
        temp_path = None
        try:
            # Convert base64 to temporary file
            temp_path = self._base64_to_temp_file(audio_base64)
            
            # Get embeddings
            audio_embed = self.clap_model.get_audio_embedding_from_filelist(x=[temp_path], use_tensor=True)
            text_embed = self.clap_model.get_text_embedding([text_prompt], use_tensor=True)
            
            similarity = torch.nn.functional.cosine_similarity(audio_embed, text_embed)
            return similarity.item()
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_path}: {e}")

    def get_audio_richness(self, audio_base64):
        """Measures Spectral Flatness and Entropy (Proxies for quality/complexity) from base64 audio"""
        # Handle data URL format (data:audio/wav;base64,...)
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes))
        
        # Spectral Flatness: 1.0 = white noise, 0.0 = pure tone. 
        # We want a mid-range for complex scores.
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # Spectral Entropy: How 'unpredictable' the sound is.
        S = np.abs(librosa.stft(y))
        psd = np.sum(S**2, axis=1)
        psd /= np.sum(psd)
        spec_entropy = entropy(psd)
        
        return flatness, spec_entropy

    def evaluate_sync_from_audio_base64(self, audio_base64, action_keywords=["shot", "bang", "crash", "door"]):
        """Check if peaks exist in audio (simple onset detection) from base64 audio"""
        # Handle data URL format (data:audio/wav;base64,...)
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes))
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        # Returns number of sharp transients found
        return len(onsets)
    
    def get_noise_floor(self, audio_base64):
        """Calculate noise floor in dB from base64 audio"""
        # Handle data URL format (data:audio/wav;base64,...)
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes))
        
        # Calculate RMS (Root Mean Square) in dB
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.power_to_db(rms, ref=np.max)
        
        # Noise floor is the minimum RMS value
        noise_floor_db = np.min(rms_db)
        
        return noise_floor_db

# if __name__ == "__main__":
#     evaluator = AudioEvaluator()
#     evaluator.evaluate_sync("generated_score.wav")
#     c_score = evaluator.get_clap_score("generated_score.wav", "i followed a dog where i heard a gunshot and footsteps apporaching me")
#     flat, ent = evaluator.get_audio_richness("generated_score.wav")
#     peak_count = evaluator.evaluate_sync("generated_score.wav")
#     print(f"""
#     --- Evaluation Results (No Ground Truth) ---
#     Text-Audio Alignment (CLAP): {c_score:.4f}  (Target: >0.25 for good match)
#     Spectral Entropy (Richness): {ent:.4f}      (Target: Higher = more complex music)
#     Spectral Flatness (Noise):   {flat:.4f}     (Target: Lower = more tonal/musical)
#     Detected Audio Onsets:       {peak_count}   (Number of dynamic events)
#     """)