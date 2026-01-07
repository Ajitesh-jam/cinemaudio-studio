from headers.imports import *
from Variable.configurations import STEPS,EMOTIONAL_RATE,EMOTIONAL_GAIN
import numpy as np
from tangoflux import TangoFluxInference 

logger = logging.getLogger(__name__)

logger.info("Loading model...")
model = TangoFluxInference(name='declare-lab/TangoFlux')
def emotional_music_generator(prompt: str, duration_ms: int):
    """ Generates a background music track."""
    logger.info(f"Generating: '{prompt}' ({duration_ms}ms)")
    duration_s = int(duration_ms / 1000.0)
    audio_arr = model.generate(prompt, steps=STEPS, duration=duration_s)
    
    # Validate audio array before processing
    if audio_arr is None or audio_arr.numel() == 0:
        raise ValueError(f"Failed to generate audio for prompt: '{prompt}'. Model returned empty array.")
    
    # Convert to numpy array
    waveform = audio_arr.squeeze().cpu().numpy()
    
    # Validate waveform
    if waveform.size == 0:
        raise ValueError(f"Generated audio waveform is empty for prompt: '{prompt}'")
    
    logger.debug(f"Audio clip from prompt {prompt} generated (shape: {waveform.shape})")
    
    # Only try to display Audio if in notebook environment and data is valid
    try:
        if waveform.size > 0:
            Audio(data=audio_arr, rate=EMOTIONAL_RATE)
    except (ValueError, AttributeError) as e:
        logger.debug(f"Could not display audio (this is OK if not in notebook): {e}")
    # Convert to 16-bit PCM
    audio_bytes = (waveform * 32767 * EMOTIONAL_GAIN).astype(np.int16).tobytes()
    segment = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=EMOTIONAL_RATE,
        channels=1,
    )
    return segment


# TESTING


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
#     # Test the Emotional Music generator
#     test_prompt = "A calm and soothing background music with emotional undertones"
#     test_duration_ms = 5000  # 5 seconds
#     music_segment = emotional_music_generator(test_prompt, test_duration_ms)
#     samples = np.array(music_segment.get_array_of_samples())
#     music = torch.from_numpy(samples).unsqueeze(0).float() / 32767.0
#     # Play the generated music (uncomment the next line if running in an environment that supports audio playback)
#     # play(music_segment)
#     torchaudio.save("Debug/test_" + test_prompt.replace(" ", "_") + ".wav", music, EMOTIONAL_RATE)
#     print(f"[TEST] Generated Emotional Music AudioSegment: {music_segment}")