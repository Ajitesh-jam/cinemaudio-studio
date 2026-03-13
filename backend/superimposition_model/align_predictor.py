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


import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

from model.dl_based_alignment_predictor import CinematicMixPredictor



# ==========================================
# 1. THE DATASET CLASS
# ==========================================
class CinematicMixDataset(Dataset):
    def __init__(self, data_samples, embedder):
        """
        data_samples: A list of dictionaries containing your ground truth data.
        embedder: The SentenceTransformer model to convert text to vectors.
        """
        self.embedder = embedder
        self.samples = data_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Embed the text (No gradients needed here, so we use torch.no_grad())
        with torch.no_grad():
            story_emb = self.embedder.encode(sample["story_prompt"], convert_to_tensor=True)
            class_emb = self.embedder.encode(sample["audio_class"], convert_to_tensor=True)
        
        # 2. Get the Whisper anchor hint
        anchor_tensor = torch.tensor([sample["whisper_anchor"]], dtype=torch.float32).to(story_emb.device)
        
        # 3. Concatenate into the [769] input vector
        x = torch.cat((story_emb, class_emb, anchor_tensor), dim=0)
        
        # 4. The Ground Truth Labels (Y)
        y = torch.tensor([
            sample["true_start_time"], 
            sample["true_weight_db"], 
            sample["true_duration"]
        ], dtype=torch.float32)
        
        return x, y

    def train_mixing_model(model, dataloader, epochs=50, learning_rate=0.001):
        """
        Trains the CinematicMixPredictor using MSE Loss and the Adam optimizer.
        """
        # Mean Squared Error is standard for predicting continuous numbers
        criterion = nn.MSELoss()
        
        # Adam is the best all-around optimizer for MLPs
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        model.train() # Set model to training mode
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_x, batch_y in dataloader:
                # 1. Zero out old gradients
                optimizer.zero_grad()
                
                # 2. Forward Pass: Make a prediction
                predictions = model(batch_x)
                
                # 3. Calculate the Error (Loss) between prediction and ground truth
                loss = criterion(predictions, batch_y)
                
                # 4. Backward Pass: Calculate how much to change the weights
                loss.backward()
                
                # 5. Optimize: Update the weights
                optimizer.step()
                
                epoch_loss += loss.item()
                
            # Print progress every 10 epochs
            avg_loss = epoch_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss (MSE): {avg_loss:.4f}")
                
        print("Training complete!")
        return model

class MixingInferencePipeline:
    def __init__(self, model_path=None):
        # 1. Decide device and load the NLP Embedder on it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading SentenceTransformer on device: {self.device} ...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=str(self.device))
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        
        # 2. Initialize our DL Model on the same device
        self.model = CinematicMixPredictor(embed_dim=self.embed_dim).to(self.device)
        
        # If you have trained weights, load them here:
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            
        self.model.eval() # Set to evaluation mode

    def _extract_whisper_anchor(self, audio_class: str, whisper_json: list) -> float:
        """
        Helper function: Tries to find the trigger word in the Whisper JSON.
        If the audio_class is "Dog barking", it looks for "dog" or "bark" in the text.
        """
        if not whisper_json:
            return 0.0
            
        # Very basic keyword matching
        keywords = audio_class.lower().split()
        
        for segment in whisper_json:
            word = segment.get("word", "").lower().strip()
            # If any keyword matches the spoken word, return its start time
            for kw in keywords:
                if kw in word:
                    return float(segment.get("start", 0.0))
        return 0.0

    def predict_parameters(self, story_prompt: str, audio_classes: list, whisper_json: list = None):
        """
        Takes the story and a list of sounds to generate parameters for.
        """
        results = []
        
        # Embed the story once (it's the same for all audio classes)
        story_emb = self.embedder.encode(
            story_prompt,
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
                
                # 2. Extract the Whisper Anchor (The hint)
                anchor_time = self._extract_whisper_anchor(audio_class, whisper_json)
                anchor_tensor = torch.tensor([anchor_time], dtype=torch.float32).to(story_emb.device)
                
                # 3. Concatenate into our input vector X
                # Shape: [384] + [384] + [1] = [769]
                x = torch.cat((story_emb, class_emb, anchor_tensor), dim=0)
                
                # 4. Predict (ensure input is on the same device as the model)
                x = x.to(self.device)
                predictions = self.model(x.unsqueeze(0)) # Add batch dimension
                
                # Extract the 3 continuous values
                start_time, weight_db, duration = predictions[0].tolist()
                
                # Post-process to ensure no negative time or duration
                results.append({
                    "audio_class": audio_class,
                    "predicted_start_time": max(0.0, start_time),
                    "predicted_weight_db": weight_db,
                    "predicted_duration": max(0.5, duration) # Minimum 0.5s duration
                })
                
        return results

# ==========================================
# USAGE EXAMPLE FOR YOUR FASTAPI BACKEND
# ==========================================
if __name__ == "__main__":
    # Initialize your pipeline
    pipeline = MixingInferencePipeline()
    
    story = "A dog barking in the forest while it is raining heavily."
    classes_to_predict = ["Dog barking", "Rain falling", "Suspense music"]
    
    # Mock Whisper JSON (optional)
    mock_whisper = [
        {"word": "A", "start": 0.0, "end": 0.2},
        {"word": "dog", "start": 0.2, "end": 0.6},
        {"word": "barking", "start": 0.6, "end": 1.1}
    ]
    
    # Run the prediction
    predicted_params = pipeline.predict_parameters(
        story_prompt=story,
        audio_classes=classes_to_predict,
        whisper_json=mock_whisper
    )
    
    print(json.dumps(predicted_params, indent=2))
    
    
    
if __name__ == "__main__":
    # 1. Initialize the Embedder and Model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embed_dim = embedder.get_sentence_embedding_dimension()
    model = CinematicMixPredictor(embed_dim=embed_dim)
    
    # 2. Prepare your Ground Truth Training Data
    # This is what you want the model to learn to output!
    training_data = [
        {
            "story_prompt": "A dog barks in the quiet forest.",
            "audio_class": "Dog barking",
            "whisper_anchor": 1.2,        # The word 'dog' was spoken at 1.2s
            "true_start_time": 1.2,       # We want the SFX to start exactly at 1.2s
            "true_weight_db": -5.0,       # Loud but not overpowering
            "true_duration": 2.0          # 2 second bark
        },
        {
            "story_prompt": "A dog barks in the quiet forest.",
            "audio_class": "Quiet forest ambience",
            "whisper_anchor": 0.0,        # Ambience doesn't have a specific anchor
            "true_start_time": 0.0,       # Starts immediately
            "true_weight_db": -20.0,      # Very quiet background noise
            "true_duration": 10.0         # Lasts the whole scene
        }
        # ... Add hundreds of these examples here ...
    ]
    
    # 3. Create Dataset and DataLoader
    dataset = CinematicMixDataset(training_data, embedder)
    # Batch size of 16 or 32 is usually good. We use 2 here since we only have 2 examples.
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 4. Train the Model
    trained_model = train_mixing_model(model, dataloader, epochs=100, learning_rate=0.005)
    
    # 5. Save the trained weights to use in your FastAPI app!
    torch.save(trained_model.state_dict(), "cinematic_mixer_weights.pth")
    print("Model saved to cinematic_mixer_weights.pth")    