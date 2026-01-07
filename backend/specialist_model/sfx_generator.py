from headers.imports import *
from Variable.configurations import STEPS,SFX_RATE,SFX_GAIN
import numpy as np
from tangoflux import TangoFluxInference
from IPython.display import Audio

logger = logging.getLogger(__name__)

logger.info("Loading model...")
model = TangoFluxInference(name='declare-lab/TangoFlux')

def sfx_generator(prompt: str, duration_ms: int):
    """ Generates a short sound effect."""
    logger.info(f"Generating: '{prompt}' ({duration_ms}ms)")
    
    # Generate audio using the TangoFlux model and return an AudioSegment
    # The model expects duration in seconds
    duration_s = int(duration_ms / 1000.0)
    # Generate raw waveform (mono, 32kHz by default)
    audio_arr = model.generate(prompt, steps=STEPS, duration=duration_s)
    
    # Validate audio array before processing
    if audio_arr is None or audio_arr.numel() == 0:
        raise ValueError(f"Failed to generate audio for prompt: '{prompt}'. Model returned empty array.")
    
    # TangoFlux returns torch.Tensor, convert to numpy array
    waveform = audio_arr.squeeze().cpu().numpy()
    
    # Validate waveform
    if waveform.size == 0:
        raise ValueError(f"Generated audio waveform is empty for prompt: '{prompt}'")
    
    logger.debug(f"Audio clip from prompt {prompt} generated (shape: {waveform.shape})")
    
    # Only try to display Audio if in notebook environment and data is valid
    try:
        if waveform.size > 0:
            Audio(data=audio_arr, rate=SFX_RATE)
    except (ValueError, AttributeError) as e:
        logger.debug(f"Could not display audio (this is OK if not in notebook): {e}")
    
    # Convert numpy array to AudioSegment (16-bit PCM, 32kHz, mono)
    # Apply volume reduction (SFX_GAIN) before converting to int16
    audio_bytes = (waveform * 32767 * SFX_GAIN).astype(np.int16).tobytes()
    segment = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=SFX_RATE,
        channels=1,
    )
    return segment


## TESTING

# import sys
# import os
# import torch
# # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)
# print("Project root added to sys.path:", project_root)

# if __name__ == "__main__":
#     # Test the SFX generator
#     test_prompt = "A dog barking loudly"
#     test_duration_ms = 3000  # 3 seconds
#     sfx_audio = sfx_generator(test_prompt, test_duration_ms)
    
    
#     import torch 
#     import torchaudio

#     # Convert int16 samples to float32 and normalize to [-1, 1] range
#     samples = torch.tensor(sfx_audio.get_array_of_samples(), dtype=torch.float32)
#     samples = samples / 32767.0  # Normalize from int16 range to float32 range
#     samples = samples.unsqueeze(0)  # Add channel dimension
    
#     # Create Debug directory if it doesn't exist
#     os.makedirs("Debug", exist_ok=True)
    
#     torchaudio.save("Debug/test_" + test_prompt.replace(" ", "_") + ".wav", samples, SFX_RATE)
#     print(f"[TEST] Generated SFX AudioSegment: {sfx_audio}")