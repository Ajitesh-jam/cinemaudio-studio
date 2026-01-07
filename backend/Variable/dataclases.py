from headers.imports import dataclass
from pydantic import BaseModel, Field
from typing import Optional, List
from Variable.configurations import READING_SPEED_WPS


@dataclass
class AudioCue:
    """Stores all information needed for a single sound event."""
    id: int
    audio_class: str      # Prompt to send to the specialist (e.g., "rain", "dog bark")
    audio_type:str
    start_time_ms: int     # When the sound should start
    duration_ms: int       # How long the sound should play
    weight_db: float       # Volume adjustment in decibels (dB)
    fade_ms: int = 500     # Default fade in/out time

@dataclass
class AudioCueWithAudioBase64:
    audio_cue: AudioCue
    audio_base64: str
    duration_ms: int

# Request/Response Models
class DecideCuesRequest(BaseModel):
    story_text: str = Field(..., description="The story text to analyze")
    speed_wps: Optional[float] = Field(READING_SPEED_WPS, description="Words per second reading speed")

class DecideCuesResponse(BaseModel):
    cues: List[AudioCue]
    total_duration_ms: int
    message: str

class GenerateAudioFromCuesRequest(BaseModel):
    cues: List[AudioCue]
    total_duration_ms: int

class GenerateAudioFromCuesResponse(BaseModel):
    audio_cues: List[AudioCueWithAudioBase64]
    message: str = Field(..., description="Message indicating success or failure")

class GenerateFromStoryRequest(BaseModel):
    story_text: str = Field(..., description="The story text to process")
    speed_wps: Optional[float] = Field(READING_SPEED_WPS, description="Words per second reading speed")
    
class GenerateFromStoryResponse(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded WAV audio data") 
    
class GenerateAudioCuesWithAudioBase64Request(BaseModel):
    story_text: str = Field(..., description="The story text to process")
    speed_wps: Optional[float] = Field(READING_SPEED_WPS, description="Words per second reading speed")
    cues: List[AudioCueWithAudioBase64]

class GenerateAudioCuesWithAudioBase64Response(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded WAV audio data")
    message: str = Field(..., description="Message indicating success or failure")