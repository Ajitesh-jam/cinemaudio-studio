# import sys
# import os
# import torchaudio
# import torch

# # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)
# print("Project root added to sys.path:", project_root)


from headers.imports import *
from Variable.dataclases import AudioCue
from Variable.model_map import SPECIALIST_MAP
# from Variable.audio_classes_dict import SOUND_KEYWORDS


logger = logging.getLogger(__name__)


def create_audio_from_audiocue(audio_cue: AudioCue)->AudioSegment:
    """
    creates a single audio clip from a single audio cue.
    """
    logger.info(f"Creating audio from audio cue: {audio_cue.audio_class} ({audio_cue.audio_type})")
    specialist_func = SPECIALIST_MAP[audio_cue.audio_type]
    audio_clip = specialist_func(audio_cue.audio_class, audio_cue.duration_ms)
    
    fade_time = min(audio_cue.fade_ms, audio_cue.duration_ms // 2) 
    processed_clip = audio_clip.fade_in(fade_time).fade_out(fade_time)
    processed_clip = processed_clip + audio_cue.weight_db 
    return processed_clip
   

def save_audio_from_audiocue(audio_cue: AudioCue, output_path: str):
    processed_clip = create_audio_from_audiocue(audio_cue)
    processed_clip.export(output_path, format="wav")
    logger.info(f"Saved audio to {output_path}")
    return processed_clip




# # test this function
# if __name__ == "__main__":
#     # Mock test audio cues
#     test_cues = [
#         AudioCue(audio_class="rain", audio_type="AMBIENCE", start_time_ms=0, duration_ms=5000, fade_ms=1000, weight_db=-5.0),
#         # AudioCue(audio_class="Footsteps on gravel", audio_type="SFX", start_time_ms=2000, duration_ms=3000, fade_ms=500, weight_db=0.0),
#         # AudioCue(audio_class="Soft piano music", audio_type="MUSIC", start_time_ms=0, duration_ms=7000, fade_ms=2000, weight_db=-10.0),
#     ]
#     total_duration = 8000  # 8 seconds
#     final_audio = play_audio(test_cues, total_duration)
    

#     # Convert int16 samples to float32 and normalize to [-1, 1] range
#     samples = torch.tensor(final_audio.get_array_of_samples()).float() / 32768.0
#     print("Final audio tensor shape:", samples.shape)
#     torchaudio.save("test_play_audio.wav", samples.unsqueeze(0), 44100)    