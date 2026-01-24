import threading
import numpy as np
from helper.lib import TangoFluxModel
from pydub import AudioSegment
import logging
from Variable.configurations import STEPS, SFX_RATE, SFX_GAIN

logger = logging.getLogger(__name__)

def sfx_generator(prompt: str, duration_ms: int):
    """Generates a short sound effect."""
    logger.info(f"Generating: '{prompt}' ({duration_ms}ms)")

    duration_s = int(duration_ms / 1000.0)
    audio_arr = TangoFluxModel.generate(prompt, steps=STEPS, duration=duration_s)

    if audio_arr is None or audio_arr.numel() == 0:
        raise ValueError(f"Failed to generate audio for prompt: '{prompt}'. Model returned empty array.")

    waveform = audio_arr.squeeze().cpu().numpy()

    if waveform.size == 0:
        raise ValueError(f"Generated audio waveform is empty for prompt: '{prompt}'")

    logger.debug(f"Audio clip from prompt {prompt} generated (shape: {waveform.shape})")

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
