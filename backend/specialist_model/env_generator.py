import sys
import os

# Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)

from headers.imports import *
from Variable.configurations import STEPS,ENV_RATE,ENV_GAIN
import numpy as np
# import os
# import sys
# import importlib
from tangoflux import TangoFluxInference

logger = logging.getLogger(__name__)

logger.info("Loading model...")
model = TangoFluxInference(name='declare-lab/TangoFlux')


def environment_generator(prompt: str, duration_ms: int):
    """ Generates an ambient environmental sound."""
    logger.info(f"Generating: '{prompt}' ({duration_ms}ms)")

    duration_s = int(duration_ms / 1000.0)
    audio_arr = model.generate(prompt, steps=STEPS, duration=duration_s)
    
    # Validate audio array before processing
    if audio_arr is None or audio_arr.numel() == 0:
        logger.error(f"Failed to generate audio for prompt: '{prompt}'. Model returned empty array.")
        # Return silent audio segment as fallback
        return AudioSegment.silent(duration=duration_ms)
    
    # Convert to numpy array
    waveform = audio_arr.squeeze().cpu().numpy()
    
    # Validate waveform
    if waveform.size == 0:
        logger.error(f"Generated audio waveform is empty for prompt: '{prompt}'")
        # Return silent audio segment as fallback
        return AudioSegment.silent(duration=duration_ms)
    
    logger.debug(f"Audio clip from prompt {prompt} generated (shape: {waveform.shape})")
    
    # Only try to display Audio if in notebook environment and data is valid
    try:
        if waveform.size > 0:
            Audio(data=audio_arr, rate=ENV_RATE)
    except (ValueError, AttributeError) as e:
        logger.debug(f"Could not display audio (this is OK if not in notebook): {e}")
    # Convert to 16-bit PCM
    audio_bytes = (waveform * 32767 * ENV_GAIN).astype(np.int16).tobytes()

    segment = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=ENV_RATE,
        channels=1,
    )
    return segment


# TESTING


# if __name__ == "__main__":  
#     # Import torch and torchaudio only for testing
#     import torch
#     import torchaudio
    
#     logger.info("Project root added to sys.path: %s", project_root)
    
#     # test Env generator
#     test_prompt="gunshot ambient sound"
#     test_duration_ms=4000  # 4 seconds
#     env_audio=environment_generator(test_prompt,test_duration_ms)
    
#     samples = np.array(env_audio.get_array_of_samples())
#     env = torch.from_numpy(samples).unsqueeze(0).float() / 32767.0
#     # Play the generated environment sound (uncomment the next line if running in an environment that supports audio playback)
#     # play(env_audio)
    
#     # Create Debug directory if it doesn't exist
#     os.makedirs("Debug", exist_ok=True)
#     torchaudio.save("Debug/test_" + test_prompt.replace(" ", "_") + ".wav", env, ENV_RATE)
    # logger.info("Generated Environment AudioSegment")
