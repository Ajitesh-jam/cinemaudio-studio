import sys
import os

# Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)

import base64
import io
import json
import logging
import os
import tempfile
from typing import List

import librosa
import numpy as np
import pandas as pd
import torch
import laion_clap
from scipy.stats import entropy

from Variable.dataclases import AudioCue

logger = logging.getLogger(__name__)


YT_JSONL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "yt_videos", "yt_dataset.jsonl"
)


class AudioEvaluator:
    def __init__(self):
        logger.info("Loading CLAP Model for Evaluation...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use default CLAP configuration to match the released checkpoint
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt()
        self.clap_model.to(self.device)

        logger.info("Loading and embedding YouTube Ground Truth Dataset from JSONL...")
        # Manually parse JSONL for robustness (handles blank lines, comments, etc.)
        yt_records = []
        with open(YT_JSONL_PATH, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yt_records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipping malformed JSONL line %d in %s: %s",
                        idx,
                        YT_JSONL_PATH,
                        str(e),
                    )
                    continue
        self.yt_stories = pd.DataFrame(yt_records)
        
        # Pre-compute all text embeddings for the YT dataset once.
        # Filter to valid, non-empty strings to satisfy the tokenizer.
        raw_texts = self.yt_stories["story_prompt"]
        yt_texts = [str(t).strip() for t in raw_texts if isinstance(t, str) and str(t).strip()]
        if not yt_texts:
            logger.warning(
                "No valid non-empty 'story_prompt' texts found in YT dataset; "
                "semantic matching will be disabled."
            )
            self.yt_story_embeds = None
        else:
            logger.info("Embedding %d YT story prompts for retrieval...", len(yt_texts))
            with torch.no_grad():
                self.yt_story_embeds = self.clap_model.get_text_embedding(
                    yt_texts, use_tensor=True
                )
            
        self.THRESHOLD_STORY = 1 # Increased slightly for better ground truth matching
        self.THRESHOLD_CUE = 0.70

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
        # Use the maximum RMS value (float) as reference for dB conversion
        rms_db = librosa.power_to_db(rms, ref=float(np.max(rms)))
        
        # Noise floor is the minimum RMS value
        noise_floor_db = np.min(rms_db)
        
        return noise_floor_db
    

    def _find_closest_yt_story(self, generated_story_text: str):
        """Finds the absolute closest matching story in the dataset."""
        if self.yt_story_embeds is None:
            logger.warning(
                "YT story embeddings are not available; skipping semantic retrieval."
            )
            return None
        with torch.no_grad():
            gen_embed = self.clap_model.get_text_embedding([generated_story_text], use_tensor=True)
            
            # Compute cosine similarity against all YT stories at once (Matrix Math = Fast)
            similarities = torch.nn.functional.cosine_similarity(gen_embed, self.yt_story_embeds)
            
            logger.info(f"\n\nSimilarities: {similarities}\n\n")
            
            max_sim_value, max_sim_index = torch.max(similarities, dim=0)
            logger.info(f"Max Similarity Value: {max_sim_value.item()}")
            logger.info(f"Max Similarity Index: {max_sim_index.item()}")
            
            if max_sim_value.item() >= self.THRESHOLD_STORY:
                # Return the actual row from the dataframe
                closest_story = self.yt_stories.iloc[max_sim_index.item()]
                logger.info(f"Closest YT story: {self.yt_stories.iloc[max_sim_index.item()]}")
                return closest_story
        return None

        # Parse the ground truth cues (assuming they are stored as JSON/dicts in the CSV)
        import ast
        try:
            yt_cues = ast.literal_eval(closest_story["cues"])
        except:
            yt_cues = closest_story["cues"] # If already a list of dicts

        # 1. Embed all cue descriptions for matching
        with torch.no_grad():
            gen_cue_texts = [cue.audio_class for cue in generated_cues]
            yt_cue_texts = [cue["audio_class"] for cue in yt_cues]
            
            gen_cue_embeds = self.clap_model.get_text_embedding(gen_cue_texts, use_tensor=True)
            yt_cue_embeds = self.clap_model.get_text_embedding(yt_cue_texts, use_tensor=True)

        aligned_pairs = []
        
        # Find which generated cues match which YT cues
        for i, gen_embed in enumerate(gen_cue_embeds):
            sims = torch.nn.functional.cosine_similarity(gen_embed.unsqueeze(0), yt_cue_embeds)
            best_match_val, best_match_idx = torch.max(sims, dim=0)
            
            if best_match_val.item() >= self.THRESHOLD_CUE:
                aligned_pairs.append({
                    "gen": generated_cues[i],
                    "yt": yt_cues[best_match_idx.item()]
                })

        # --- SCORE 1: COVERAGE (Recall) ---
        # How much of the original cinematic mix did we successfully generate?
        coverage_score_percent = (len(aligned_pairs) / max(len(yt_cues), 1)) * 100.0

        # --- SCORE 2: CINEMATIC SYNC (Normalized MSE) ---
        if len(aligned_pairs) == 0:
            sync_score_percent = 0.0
        else:
            total_error = 0.0
            for pair in aligned_pairs:
                gen = pair["gen"]
                yt = pair["yt"]
                
                # Calculate differences (converting ms to seconds if necessary for scale)
                # Assuming your model outputs in seconds based on our previous logic
                t_diff = (gen.start_time_ms - yt["starting_time"]) ** 2
                d_diff = (gen.duration_ms - yt["duration"]) ** 2
                
                # Normalize weight penalty (dB differences are large, so we scale them down)
                w_diff = ((getattr(gen, 'weight_db', 0) - yt.get("weight_db", 0)) * 0.1) ** 2 
                
                total_error += (t_diff + d_diff + w_diff)
            
            mse = total_error / len(aligned_pairs)
            
            # Map MSE to a 0-100 score. (e.g., an MSE of 0 = 100 score. MSE > 50 = 0 score)
            sync_score_percent = max(0.0, 100.0 - (mse * 2.0))

        return {
            "closest_yt_match_prompt": closest_story["story_prompt"],
            "coverage_score": round(coverage_score_percent, 2),
            "sync_score": round(sync_score_percent, 2),
            "matched_cues_count": len(aligned_pairs)
        }   
        
    def yt_coverage_score(self, generated_story_text: str, generated_cues: List[AudioCue]):
        """Get YT sync score from story text and audio cues"""
        closest_story = self._find_closest_yt_story(generated_story_text)
        if closest_story is None:
            return {"error": "No semantic match found in YT dataset to establish ground truth."}
            
        # Parse the ground truth cues (assuming they are stored as JSON/dicts in the CSV)
        import ast
        try:
            yt_cues = ast.literal_eval(closest_story["cues"])
        except:
            yt_cues = closest_story["cues"] # If already a list of dicts

        with torch.no_grad():
            gen_cue_texts = [cue.audio_class for cue in generated_cues]
            yt_cue_texts = [cue["audio_class"] for cue in yt_cues]
            
            gen_cue_embeds = self.clap_model.get_text_embedding(gen_cue_texts, use_tensor=True)
            yt_cue_embeds = self.clap_model.get_text_embedding(yt_cue_texts, use_tensor=True)

        aligned_pairs = []
    
        for i, gen_embed in enumerate(gen_cue_embeds):
            sims = torch.nn.functional.cosine_similarity(gen_embed.unsqueeze(0), yt_cue_embeds)
            best_match_val, best_match_idx = torch.max(sims, dim=0)
            
            if best_match_val.item() >= self.THRESHOLD_CUE:
                aligned_pairs.append({
                    "gen": generated_cues[i],
                    "yt": yt_cues[best_match_idx.item()]
                })
        coverage_score_percent = (len(aligned_pairs) / max(len(yt_cues), 1)) * 100.0  
        return coverage_score_percent
    
    def yt_sync_score(self, generated_story_text: str, generated_cues: List[AudioCue]):
        """Get YT sync score from story text and audio cues"""
        closest_story = self._find_closest_yt_story(generated_story_text)
        if closest_story is None:
            return {"error": "No semantic match found in YT dataset to establish ground truth."}
            
        # Parse the ground truth cues (assuming they are stored as JSON/dicts in the CSV)
        import ast
        try:
            yt_cues = ast.literal_eval(closest_story["cues"])
        except:
            yt_cues = closest_story["cues"] # If already a list of dicts

        # 1. Embed all cue descriptions for matching
        with torch.no_grad():
            gen_cue_texts = [cue.audio_class for cue in generated_cues]
            yt_cue_texts = [cue["audio_class"] for cue in yt_cues]
            
            gen_cue_embeds = self.clap_model.get_text_embedding(gen_cue_texts, use_tensor=True)
            yt_cue_embeds = self.clap_model.get_text_embedding(yt_cue_texts, use_tensor=True)

        aligned_pairs = []
        
        # Find which generated cues match which YT cues
        for i, gen_embed in enumerate(gen_cue_embeds):
            sims = torch.nn.functional.cosine_similarity(gen_embed.unsqueeze(0), yt_cue_embeds)
            best_match_val, best_match_idx = torch.max(sims, dim=0)
            
            if best_match_val.item() >= self.THRESHOLD_CUE:
                aligned_pairs.append({
                    "gen": generated_cues[i],
                    "yt": yt_cues[best_match_idx.item()]
                })

            # --- SCORE 2: CINEMATIC SYNC (Normalized MSE) ---
        if len(aligned_pairs) == 0:
            sync_score_percent = 0.0
        else:
            total_error = 0.0
            for pair in aligned_pairs:
                gen = pair["gen"]
                yt = pair["yt"]
                
                # Calculate differences (converting ms to seconds if necessary for scale)
                # Assuming your model outputs in seconds based on our previous logic
                t_diff = (gen.start_time_ms - yt["starting_time"]) ** 2
                d_diff = (gen.duration_ms - yt["duration"]) ** 2
                
                # Normalize weight penalty (dB differences are large, so we scale them down)
                w_diff = ((getattr(gen, 'weight_db', 0) - yt.get("weight_db", 0)) * 0.1) ** 2 
                
                total_error += (t_diff + d_diff + w_diff)
            
            mse = total_error / len(aligned_pairs)
            
            # Map MSE to a 0-100 score. (e.g., an MSE of 0 = 100 score. MSE > 50 = 0 score)
            sync_score_percent = max(0.0, 100.0 - (mse * 2.0))

        return sync_score_percent    
    
    def yt_coverage_and_sync_score(self, generated_story_text: str, generated_cues: List[AudioCue]):
        """Get YT coverage and sync score from story text and audio cues"""
        coverage_score = self.yt_coverage_score(generated_story_text, generated_cues)
        sync_score = self.yt_sync_score(generated_story_text, generated_cues)
        return {
            "coverage_score": coverage_score,
            "sync_score": sync_score
        }

if __name__ == "__main__":
    evaluator = AudioEvaluator()
    # evaluator.evaluate_sync("generated_score.wav")
    # c_score = evaluator.get_clap_score("generated_score.wav", "i followed a dog where i heard a gunshot and footsteps apporaching me")
    # flat, ent = evaluator.get_audio_richness("generated_score.wav")
    # peak_count = evaluator.evaluate_sync("generated_score.wav")
    # print(f"""
    # --- Evaluation Results (No Ground Truth) ---
    # Text-Audio Alignment (CLAP): {c_score:.4f}  (Target: >0.25 for good match)
    # Spectral Entropy (Richness): {ent:.4f}      (Target: Higher = more complex music)
    # Spectral Flatness (Noise):   {flat:.4f}     (Target: Lower = more tonal/musical)
    # Detected Audio Onsets:       {peak_count}   (Number of dynamic events)
    # """)
    
    # {"audio_class": "male character gasps and strains", "starting_time": 0.5, "duration": 9.0, "weight_db": -10.0}, {"audio_class": "intense water splashing SFX", "starting_time": 0.0, "duration": 10.0, "weight_db": -12.0}]
  
    story_prompt = "A lone, mud-streaked figure with a mix of terror and grim determination battles relentlessly against a powerful jungle river, gasping for breath amidst churning water and dense, unforgiving foliage."
    cues = [AudioCue(id=0, audio_class="male character gasps and strains", audio_type="SFX", start_time_ms=500, duration_ms=9000, weight_db=-10.0), AudioCue(id=1, audio_class="intense water splashing SFX", audio_type="SFX", start_time_ms=0, duration_ms=10000, weight_db=-12.0)]
    print(f"Story Prompt: {story_prompt}")
    print(f"Cues: {cues}\n\n")
    coverage_and_sync_score = evaluator.yt_coverage_and_sync_score(story_prompt, cues)
    print(f"Coverage and Sync Score: {coverage_and_sync_score}")


