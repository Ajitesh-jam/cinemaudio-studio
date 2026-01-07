
from csv import Error
import sys
import os

# Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)
print("Project root added to sys.path:", project_root)

from headers.imports import *
from Variable.dataclases import AudioCue, AudioCueWithAudioBase64
from Variable.model_map import SPECIALIST_MAP
from Tools.play_audio import create_audio_from_audiocue
from Tools.decide_audio import decide_audio_cues
from Variable.configurations import READING_SPEED_WPS
# from Variable.audio_classes_dict import SOUND_KEYWORDS

logger = logging.getLogger(__name__)


def superimpose_audio(audio_cues: List[AudioCue], total_duration_ms: int):
    """
    Superimposes all audio cues into a single track.
    """
    logger.info("Starting audio superimposition process...")
    logger.info(f"Creating silent audio canvas of {total_duration_ms}ms.")
    final_audio = AudioSegment.silent(duration=total_duration_ms)
    for cue in audio_cues:
        final_audio = final_audio.overlay(create_audio_from_audiocue(cue))
    return final_audio



def superimpose_audio_cues(audio_cues: List[AudioCue], total_duration_ms: int):
    """
    Superimposes all audio cues into a single track.
    """
    logger.info("Starting audio superimposition process...")
    logger.info(f"Creating silent audio canvas of {total_duration_ms}ms.")
    final_audio = AudioSegment.silent(duration=total_duration_ms)
    for cue in audio_cues:
        final_audio = final_audio.overlay(create_audio_from_audiocue(cue), position=cue.start_time_ms)
    return final_audio


def superimpose_audio_cues_with_audio_base64(audio_cues: List[AudioCueWithAudioBase64], total_duration_ms: int):
    """
    Superimposes all audio cues with audio base64 into a single track.
    """
    logger.info("Starting audio superimposition process...")
    logger.info(f"Creating silent audio canvas of {total_duration_ms}ms.")
    final_audio = AudioSegment.silent(duration=total_duration_ms)
    for cue in audio_cues:
        final_audio = final_audio.overlay(cue.audio_base64, position=cue.audio_cue.start_time_ms)
    return final_audio

def superimposition_model(story_text: str, speed_wps: float):
    """
    Superimposes all audio cues with audio base64 into a single track.
    """
    try:
        cues, total_duration = decide_audio_cues(story_text, speed_wps)
        final_audio = superimpose_audio(cues, total_duration)
        return final_audio
    except Exception as e:
        logger.error(f"Error in superimposition model: {e}", exc_info=True)
        raise Error(f"Error in superimposition model: {e}")


## TESTING  


# test this function
# if __name__ == "__main__":
#     story = "i ran towards the shelter where i heard cat meowing"
#     cues, total_duration = decide_audio(story, READING_SPEED_WPS)
#     final_audio = AudioSegment.silent(duration=total_duration)
#     index = 0
#     for cue in cues:
#         index += 1
#         logger.info(f"Overlaying '{cue.audio_class}' at {cue.start_time_ms}ms.")
#         final_audio = final_audio.overlay(create_audio_from_audiocue(cue), position=cue.start_time_ms)
#         final_audio.export("Debug/"+story[:20].replace(" ","_")+"_intermediate_output_"+str(index)+".wav", format="wav")
        
#     logger.info("Exporting final audio to output file...")    
#     final_audio.export("Output/"+story[:20].replace(" ","_")+"_final_output.wav", format="wav")
   