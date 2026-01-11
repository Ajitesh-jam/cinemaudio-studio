import base64
import io
from pydub import AudioSegment
from Variable.dataclases import AudioCue


# Helper function to convert AudioCue to dict
def audio_cue_to_dict(cue: AudioCue) -> dict:
    """Convert AudioCue to dictionary"""
    return {
        "audio_class": cue.audio_class,
        "audio_type": cue.audio_type,
        "start_time_ms": cue.start_time_ms,
        "duration_ms": cue.duration_ms,
        "weight_db": cue.weight_db,
        "fade_ms": cue.fade_ms
    }

# Helper function to convert AudioSegment to base64
def audio_to_base64(audio: AudioSegment, format: str = "wav") -> str:
    """Convert AudioSegment to base64 encoded string"""
    buffer = io.BytesIO()
    audio.export(buffer, format=format)
    buffer.seek(0)
    audio_bytes = buffer.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    return audio_base64

def base64_to_audio(audio_base64: str) -> AudioSegment:
    """Convert base64 encoded string to AudioSegment"""
    audio_bytes = base64.b64decode(audio_base64)
    return AudioSegment.from_file(io.BytesIO(audio_bytes))