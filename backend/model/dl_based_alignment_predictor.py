import os
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
    def __init__(self, embed_dim=384):
        """
        A Multi-Task Feed-Forward Network to predict mixing parameters.
        embed_dim depends on the SentenceTransformer model (all-MiniLM-L6-v2 is 384).
        Input size = (Story Embed) + (Class Embed) + (Whisper embedding)
                   = 384 + 384 + 384 = 1152
        """
        super(CinematicMixPredictor, self).__init__()
        
        input_size = 2*(embed_dim) + 1
        
        # Hidden Layers
        self.layer1 = nn.Linear(input_size, 2048)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.layer2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        
        # Output Layer: predicts [start_time, weight_db, duration]
        self.output_layer = nn.Linear(1024, 3)
        
        self.embedder = self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Ensure model layers live on the same device as embeddings
        self.to(self.device)

    def forward(self, x):
        logger.info(f"inference input shape: {x.shape}")
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        
        return self.output_layer(x)
    
    def train_model(self, dataloader, epochs=50, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
            
        return self
    
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
                # Shape: [384] + [384] + [384] = [1152]
                most_relevant_word = self.get_timestemap_of_most_relevant_word(whisper_json, audio_class)
                x = torch.cat((story_emb, class_emb, whisper_emb), dim=0)
                x = torch.cat((x, most_relevant_word["start"]), dim=0)
                
                # 4. Predict the mixing parameters
                outputs = self(x)
                
                results.append(outputs.tolist())
                
        return results




# testing the model
if __name__ == "__main__":
    model = CinematicMixPredictor()
    # model.load_model(MODEL_PATH)
    story_prompt = "A dog barking in the forest while it is raining heavily."
    description = "A male speaker with a neutral tone delivers his words clearly and confidently in a casual, everyday setting."
    audio_classes = ["Dog barking", "Rain falling", "Suspense music"]
    narrator_audio_segment = ParlerTTSModel.generate(
        prompt=story_prompt, description=description
    )
    narrator_audio_base64 = audio_to_base64(narrator_audio_segment)
   
    results = model.predict(story_prompt, audio_classes, narrator_audio_base64)
    
    
    print(json.dumps(results, indent=2))
    
    
    
    # bank name
    # baknk addrews
    # isfc 
    # acorun number
    # pan  number
    