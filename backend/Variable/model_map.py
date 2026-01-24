from specialist_model.sfx_generator import sfx_generator
from specialist_model.env_generator import environment_generator
from specialist_model.emotional_generator import emotional_music_generator
from specialist_model.text_to_speech_generator import text_to_speech_generator
# Audio Type mapping
SPECIALIST_MAP = {
    "SFX": sfx_generator,
    "AMBIENCE": environment_generator,
    "MUSIC": emotional_music_generator,
    "NARRATOR": text_to_speech_generator,
}
