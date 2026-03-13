import os
import random
import sys
# Add project root to path
# import sys
# import os

# # Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# # Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)
print("Project root added to sys.path:", project_root)
    
# Cinemaudio-studio root (for tango_new when using Tango2)
cinema_studio_root = os.path.abspath(os.path.join(project_root, ".."))
if cinema_studio_root not in sys.path:
    sys.path.append(cinema_studio_root)    


import base64
import io
import json
import logging
import os
from typing import Any, List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from pydub import AudioSegment
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from model.parlerTTSModel import ParlerTTSModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helper.audio_conversions import audio_to_base64

parler_tts_ins = ParlerTTSModel.get_instance()
logger = logging.getLogger(__name__)


MODEL_PATH = "model/dl_based_alignment_predictor.pth"
EMBEDDER_PATH = "model/embedder.pth"


class WordAligner:
    def __init__(self, model_id: str = "openai/whisper-base"):
        """
        Initializes the Whisper model via Hugging Face pipeline.
        The key parameter is return_timestamps="word" to get per-word timing.
        """
        logger.info(f"[*] Loading {model_id} for Word Alignment...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # The magic parameter here is return_timestamps="word"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            chunk_length_s=30,
            device=self.device,
            return_timestamps="word",
        )
        logger.info("[+] Aligner ready.")

    def _align_audio_array(self, audio_array: np.ndarray, sr: int) -> List[dict]:
        """Internal method to resample and run inference."""
        # Whisper strictly requires 16kHz audio
        if sr != 16000:
            audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=16000)

        # Ensure it's 1D (mono)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Run the model
        result: Any = self.pipe(audio_array)

        # Hugging Face ASR pipeline can return either a dict with "chunks"
        # or a simpler structure. We handle the dict-with-chunks case here.
        chunks = result["chunks"] if isinstance(result, dict) and "chunks" in result else []

        # Format the output to match the requested structure
        formatted_timestamps: List[dict] = []
        for chunk in chunks:
            # Sometimes the end timestamp can be None at the very end of the file
            start_time = (
                round(chunk["timestamp"][0], 2)
                if chunk["timestamp"][0] is not None
                else 0.0
            )
            end_time = (
                round(chunk["timestamp"][1], 2)
                if chunk["timestamp"][1] is not None
                else start_time + 0.2
            )

            formatted_timestamps.append(
                {
                    "word": chunk["text"].strip(),
                    "start": start_time,
                    "end": end_time,
                }
            )

        return formatted_timestamps

    def get_timestamps_from_base64(self, base64_audio: str) -> List[dict]:
        """Processes a base64 encoded audio string."""
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(base64_audio)

        # Load bytes into numpy array using soundfile
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))

        return self._align_audio_array(audio_array, sr)

    def get_timestamps_from_parler_audio(self, parler_audio: AudioSegment) -> List[dict]:
        """
        Processes the audio output from ParlerTTSModel (pydub.AudioSegment).
        """
        samples = parler_audio.get_array_of_samples()
        audio_array = np.asarray(samples, dtype=np.float32) / 32768.0
        sr = parler_audio.frame_rate
        return self._align_audio_array(audio_array, sr)


class CinematicMixPredictor(nn.Module):
    
    def __init__(self, embed_dim=384, n_heads=4, attention_dropout=0.1):
        """
        CinematicMixPredictor rewritten as a model using self and cross attention
        to fuse story, class, and whisper embedding inputs.

        Input tokens: [story_embedding, class_embedding, whisper_anchor_tensor, extra_features]
        self-attention and cross-attention is used to learn the relationship.
        """
        super(CinematicMixPredictor, self).__init__()

        self.embed_dim = embed_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.n_heads = n_heads

        # Project whisper_anchor + extra_features (2 floats) up to 'embed_dim'
        self.anchor_project = nn.Linear(2, embed_dim)

        # LayerNorms for inputs
        self.story_ln = nn.LayerNorm(embed_dim)
        self.class_ln = nn.LayerNorm(embed_dim)
        self.anchor_ln = nn.LayerNorm(embed_dim)

        # Self-attention for contextualizing all 3 representations together
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads=n_heads, dropout=attention_dropout, batch_first=True
        )

        # Simple FeedForward after attention
        self.ffn = nn.Sequential(
            nn.Linear(3 * embed_dim, 1024),
            nn.ReLU(),
            nn.Dropout(attention_dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # [start_time, weight_db, duration]
        )

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.to(self.device)

    def forward(self, x):
        """
        x: shape [batch, input_dim]
            - where input_dim = 2*embed_dim + 2

        Input packing:
            x[:, :embed_dim]          = story embedding
            x[:, embed_dim:2*embed_dim] = class embedding
            x[:, 2*embed_dim:]        = [whisper_anchor, extra_feature] (2 floats)
        """

        # Allow passing a single example vector of shape [input_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)


        # Split x into components
        story_emb = x[:, :self.embed_dim]
        class_emb = x[:, self.embed_dim:2 * self.embed_dim]
        anchor_feats = x[:, 2 * self.embed_dim:]  # should have shape [batch, 2]

        # Project anchor_feats into 'embed_dim'
        anchor_emb = self.anchor_project(anchor_feats)

        # Layer normalization
        story_emb = self.story_ln(story_emb)
        class_emb = self.class_ln(class_emb)
        anchor_emb = self.anchor_ln(anchor_emb)

        # Stack as tokens [batch, 3, embed_dim]
        tokens = torch.stack([story_emb, class_emb, anchor_emb], dim=1)

        # Apply (self + cross) attention to all tokens
        attn_out, _ = self.self_attn(tokens, tokens, tokens)

        # Flatten all three tokens
        attn_flat = attn_out.reshape(attn_out.shape[0], -1)  # [batch, 3*embed_dim]

        # Feedforward to output
        output = self.ffn(attn_flat)
        return output
        return self.output_layer(x)
    
    def train_model(self, dataloader, epochs=50, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for inputs, targets in dataloader:
                # Ensure tensors are on the same device as the model
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                epoch_loss += loss.item()
                num_batches += 1
                loss.backward()
                optimizer.step()

            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
            
        return losses
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self

    def make_whisper_embedding(
        self, story_prompt: str, narrator_audio_base64: str
    ) -> List[dict]:
        """
        Uses a Whisper-based ASR pipeline (via Hugging Face) to generate
        word-level timestamps (whisper_json) from the narrator audio.
        """
        if not hasattr(self, "_word_aligner") or self._word_aligner is None:
            self._word_aligner = WordAligner()

        words_json = self._word_aligner.get_timestamps_from_base64(narrator_audio_base64)
        logger.info(f"Generated Whisper JSON: {words_json}")
        return words_json
    
    def get_timestemap_of_most_relevant_word(self, words_json: List[dict], audio_class: str) -> dict:
        """
        Returns the timestamp of the most relevant word in the story for the given audio class,
        using SentenceTransformer embeddings to pick the word whose embedding is closest to
        the audio_class embedding.
        """
        # Get clip embedding for audio_class
        audio_class_emb = self.embedder.encode(audio_class, convert_to_tensor=True, device=str(self.device))

        # Get word embeddings for all words in words_json
        word_texts = [w["word"] for w in words_json]
        word_embs = self.embedder.encode(word_texts, convert_to_tensor=True, device=str(self.device))

        # Compute cosine similarity between each word embedding and the audio_class embedding
        from torch.nn.functional import cosine_similarity
        sims = cosine_similarity(word_embs, audio_class_emb.unsqueeze(0))

        best_idx = int(torch.argmax(sims).item())
        logger.info(f"Most relevant word: {words_json[best_idx]}")
        return words_json[best_idx]
        

    def predict(
        self,
        story_prompt: str,
        audio_classes: List[str],
        narrator_audio_base64: str,
    ):
        """
        Takes the story and a list of sounds to generate parameters for.
        """
        results: List[Any] = []
        whisper_json = self.make_whisper_embedding(story_prompt, narrator_audio_base64)
        
        # Embed the story once (it's the same for all audio classes)
        story_emb = self.embedder.encode(
            story_prompt,
            convert_to_tensor=True,
            device=str(self.device),
        )
        
        # Prepare Whisper anchor embedding once from provided timestamps (or fallback)
        if whisper_json is not None:
            transcript_text = " ".join([w["word"] for w in whisper_json])
        else:
            transcript_text = story_prompt

        whisper_emb = self.embedder.encode(
            transcript_text,
            convert_to_tensor=True,
            device=str(self.device),
        )
        
        with torch.no_grad():
            for audio_class in audio_classes:
                # 1. Embed the Audio Class
                class_emb = self.embedder.encode(
                    audio_class,
                    convert_to_tensor=True,
                    device=str(self.device),
                )
                
                # 2. Concatenate into our input vector X
                # Shape: [384] + [384] + [2] = [768]
                most_relevant_word = self.get_timestemap_of_most_relevant_word(whisper_json, audio_class)
                # Prepare the anchor tensor with the most relevant word start/end (as a tensor of shape [2])
                anchor_tensor = torch.tensor([most_relevant_word["start"], most_relevant_word["end"]], dtype=torch.float32, device=story_emb.device)
                x = torch.cat((story_emb, class_emb, anchor_tensor), dim=0)
                
                # 4. Predict the mixing parameters
                outputs = self(x)
                
                results.append(outputs.tolist())
                
        return results



def train_model(epochs=50, learning_rate=0.001):    
    descriptions = ["A male speaker with a neutral tone delivers his words clearly and confidently in a casual, everyday setting.", "A female speaker with a high-pitched voice is delivering her speech at a really fast speed in a noisy environment.", "A male speaker with a low-pitched voice is delivering his speech at a really slow speed in a quiet environment.","A female speaker with a low-pitched voice is delivering her speech at a really fast speed in a noisy environment."]

    model = CinematicMixPredictor()
 
    datapath = "../data/yt_videos/yt_dataset.jsonl"
    dataset = []
    with open(datapath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            story_prompt = data["story_prompt"]
            
            
            random_description = random.choice(descriptions)
                
            narrator_audio_segment = ParlerTTSModel.generate(prompt=story_prompt, description=random_description)
            narrator_audio_base64 = audio_to_base64(narrator_audio_segment)

            audio_classes = []
            for audio_cues in data["cues"]:
                audio_classes.append({
                    "audio_class": audio_cues["audio_class"],
                    "weight_db": audio_cues["weight_db"],
                    "start_time_ms": audio_cues["starting_time"],
                    "duration_ms": audio_cues["duration"],
                })
            dataset.append((story_prompt, audio_classes, narrator_audio_base64))
    
    # Build training examples: one example per (story, cue)
    input_tensors = []
    target_tensors = []

    for story_prompt, audio_classes, narrator_audio_base64 in dataset:
        # Whisper word-level timestamps for this narrated story
        whisper_json = model.make_whisper_embedding(story_prompt, narrator_audio_base64)

        # Story embedding (same for all cues of this story)
        story_emb = model.embedder.encode(
            story_prompt,
            convert_to_tensor=True,
            device=str(model.device),
        )

        for cue in audio_classes:
            audio_class = cue["audio_class"]

            # Class embedding
            class_emb = model.embedder.encode(
                audio_class,
                convert_to_tensor=True,
                device=str(model.device),
            )

            # Anchor from most relevant word timestamps
            most_relevant_word = model.get_timestemap_of_most_relevant_word(
                whisper_json, audio_class
            )
            anchor_tensor = torch.tensor(
                [most_relevant_word["start"], most_relevant_word["end"]],
                dtype=torch.float32,
                device=model.device,
            )

            # Concatenate into model input vector
            x = torch.cat((story_emb, class_emb, anchor_tensor), dim=0)

            # Target: [start_time_ms, weight_db, duration_ms]
            y = torch.tensor(
                [
                    cue["start_time_ms"],
                    cue["weight_db"],
                    cue["duration_ms"],
                ],
                dtype=torch.float32,
                device=model.device,
            )

            input_tensors.append(x)
            target_tensors.append(y)

    # Stack and create DataLoader
    inputs_tensor = torch.stack(input_tensors)
    targets_tensor = torch.stack(target_tensors)

    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(inputs_tensor, targets_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    losses = model.train_model(train_loader, epochs, learning_rate)
    model.save_model(MODEL_PATH)
    
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()
    plt.tight_layout()
    plt.savefig("loss_curve.png", bbox_inches="tight")
    plt.close()
    
    print("Model saved to", MODEL_PATH)
    
    
    
    
    



# # testing the model
# if __name__ == "__main__":
#     model = CinematicMixPredictor()
#     # model.load_model(MODEL_PATH)
#     story_prompt = "A dog barking in the forest while it is raining heavily."
#     description = "A male speaker with a neutral tone delivers his words clearly and confidently in a casual, everyday setting."
#     audio_classes = ["Dog barking", "Rain falling", "Suspense music"]
#     narrator_audio_segment = ParlerTTSModel.generate(
#         prompt=story_prompt, description=description
#     )
#     narrator_audio_base64 = audio_to_base64(narrator_audio_segment)
   
#     results = model.predict(story_prompt, audio_classes, narrator_audio_base64)
    
    
#     print(json.dumps(results, indent=2))
    
    
    
    
    
# train model 
train_model()