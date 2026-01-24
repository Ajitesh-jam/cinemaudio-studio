import numpy as np
from pydub import AudioSegment

from headers.imports import *
from Variable.dataclases import AudioCue, NarratorCue, Cue
from Variable.model_map import SPECIALIST_MAP
from helper.lib import ParlerTTSModel

logger = logging.getLogger(__name__)

def _tts_numpy_to_audio_segment(audio_arr: np.ndarray, duration_ms: int) -> AudioSegment:
    """Convert TTS numpy output (float32) to AudioSegment."""
    model = ParlerTTSModel.get_instance()["model"]
    sample_rate = model.config.sampling_rate
    gain = 0.9
    audio_arr = np.clip(audio_arr, -1.0, 1.0)
    audio_bytes = (audio_arr * 32767 * gain).astype(np.int16).tobytes()
    seg = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    if len(seg) > duration_ms:
        seg = seg[:duration_ms]
    return seg  # type: ignore[return-value]


def create_audio_from_audiocue(audio_cue: Cue) -> AudioSegment:
    """
    Create a single audio clip from a single cue (AudioCue or NarratorCue).
    """
    logger.info(f"Creating audio from cue: {audio_cue}\n\n")
    
    if isinstance(audio_cue, NarratorCue):
        logger.info(f"Creating audio from narrator cue: {audio_cue.id} ({audio_cue.audio_type})")
        specialist_func = SPECIALIST_MAP[audio_cue.audio_type]
        audio_arr = specialist_func(audio_cue.story, audio_cue.narrator_description)
        clip = _tts_numpy_to_audio_segment(audio_arr, audio_cue.duration_ms)
        fade_ms = min(100, audio_cue.duration_ms // 4)
        faded = clip.fade_in(fade_ms).fade_out(fade_ms)
        return faded  # type: ignore[return-value]
    else:
        logger.info(f"Creating audio from audio cue: {audio_cue.audio_class} ({audio_cue.audio_type})")
        specialist_func = SPECIALIST_MAP[audio_cue.audio_type]
        audio_clip = specialist_func(audio_cue.audio_class, audio_cue.duration_ms)
        fade_time = min(audio_cue.fade_ms, audio_cue.duration_ms // 2)
        processed_clip = audio_clip.fade_in(fade_time).fade_out(fade_time)
        processed_clip = processed_clip + audio_cue.weight_db
        return processed_clip


def save_audio_from_audiocue(audio_cue: Cue, output_path: str) -> AudioSegment:
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
