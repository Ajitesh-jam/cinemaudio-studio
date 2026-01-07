from headers.imports import Dict, Tuple

# --- Reading Speed Configuration ---
READING_SPEED_WPS = 2.0  # Words Per Second

# --- Duration Configuration ---
DEFAULT_SFX_DURATION_MS = 2000  # 2 seconds (for short SFX)
DEFAULT_FADE_MS = 500           # 500ms fade in/out for most sounds

# --- Weight/Volume Configuration ( LOUD/LITTLE logic ) ---

DEFAULT_WEIGHT_DB = 0.0
LOUD_WEIGHT_DB = 6.0
FAINT_WEIGHT_DB = -6.0

MODIFIER_WORDS = {
    "loud": LOUD_WEIGHT_DB,
    "faint": FAINT_WEIGHT_DB,
    "little": FAINT_WEIGHT_DB,
    "soft": FAINT_WEIGHT_DB,
    "quiet": FAINT_WEIGHT_DB,
    "roaring": LOUD_WEIGHT_DB,
    "blaring": LOUD_WEIGHT_DB,
}




# Specialist model configuartions
STEPS=50


SFX_RATE=44100
SFX_GAIN=0.5

ENV_RATE=44100
ENV_GAIN=0.7
    
EMOTIONAL_RATE=44100
EMOTIONAL_GAIN=0.8