# import sys
# import os

# # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)
# print("Project root added to sys.path:", project_root)

from csv import Error
import base64
import io
from operator import and_
from pydub import AudioSegment
import logging
from typing import List, Sequence
from Variable.dataclases import Cue, AudioCueWithAudioBase64, AudioCue
from Tools.play_audio import create_audio_from_audiocue
from Tools.decide_audio import decide_audio_cues
from Variable.configurations import READING_SPEED_WPS
from helper.audio_conversions import base64_to_audio
from Utils.prompts import prompt_to_fill_missing_audio_cues
from Utils.llm import query_llm
# from Variable.audio_classes_dict import SOUND_KEYWORDS


logger = logging.getLogger(__name__)


class SuperimpositionModel:
    def __init__(self):
        self.model = None
          
    def superimpose_audio(self, audio_cues: Sequence[Cue], total_duration_ms: int):
        """
        Superimposes all audio cues into a single track.
        """
        logger.info("Starting audio superimposition process...")
        logger.info(f"Creating silent audio canvas of {total_duration_ms}ms.")
        final_audio = AudioSegment.silent(duration=total_duration_ms)
        for cue in audio_cues:
            final_audio = final_audio.overlay(create_audio_from_audiocue(cue))
        return final_audio    
    
    def superimpose_audio_cues(self, audio_cues: Sequence[Cue], total_duration_ms: int):
        """
        Superimposes all audio cues into a single track.
        """
        logger.info("Starting audio superimposition process...")
        logger.info(f"Creating silent audio canvas of {total_duration_ms}ms.")
        final_audio = AudioSegment.silent(duration=total_duration_ms)
        for cue in audio_cues:
            final_audio = final_audio.overlay(create_audio_from_audiocue(cue), position=cue.start_time_ms)
        return final_audio
    
    def check_missing_audio_cues(self, story_text: str, audio_cues: List[AudioCueWithAudioBase64], total_duration_ms: int):
        """
        Checks and fills the audio cues with audio base64 into a single track.
        """
        logger.info("Starting audio superimposition process...")
        logger.info("checking coverage with Gemini and filling the missing audio cues ")
        
        audio_cues_to_fill = []
        
        for cue in audio_cues:
            if cue.audio_base64 is not None and isinstance(cue.audio_cue, AudioCue):
                audio_cues_to_fill.append(cue.audio_cue.audio_class)
        
        not_covered_audio_cues = []     
        if audio_cues_to_fill:
            prompt = prompt_to_fill_missing_audio_cues.format(story_text=story_text, audio_cues=audio_cues_to_fill)
            response = query_llm(llm_name="gemini", model_name="gemini-2.5-flash", prompt=prompt)
            if response:
                not_covered_audio_cues = response.get("audio_cues", [])
                
        return not_covered_audio_cues    
    
    def superimposition_model(self, story_text: str, speed_wps: float):
        """
        Superimposes all audio cues with audio base64 into a single track.
        """
        try:
            cues, total_duration = decide_audio_cues(story_text, speed_wps)
            final_audio = self.superimpose_audio(cues, total_duration)
            return final_audio
        except Exception as e:
            logger.error(f"Error in superimposition model: {e}", exc_info=True)
            raise Error(f"Error in superimposition model: {e}")
        
    def superimpose_audio_cues_with_audio_base64(self, story_text: str, audio_cues: List[AudioCueWithAudioBase64], total_duration_ms: int):
        """
        Superimposes all audio cues with audio base64 into a single track.
        """
        logger.info("Starting audio superimposition process...")
        logger.info(f"Creating silent audio canvas of {total_duration_ms}ms.")
        final_audio = AudioSegment.silent(duration=total_duration_ms)
        for cue in audio_cues:
            # Convert base64 string to AudioSegment before overlaying
            base_segment = base64_to_audio(cue.audio_base64)

            # Apply gain in dB based on weight_db (no repetition)
            weight_db = getattr(cue.audio_cue, "weight_db", 0) or 0
            segment_with_gain = base_segment + weight_db

            # Stretch/trim to match the cue's duration_ms (no looping)
            desired_duration = getattr(cue.audio_cue, "duration_ms", len(segment_with_gain)) or len(segment_with_gain)
            current_duration = len(segment_with_gain)

            if current_duration > desired_duration:
                audio_segment = segment_with_gain[:desired_duration]
            elif current_duration < desired_duration:
                padding = AudioSegment.silent(duration=desired_duration - current_duration)
                audio_segment = segment_with_gain + padding
            else:
                audio_segment = segment_with_gain

            final_audio = final_audio.overlay(audio_segment, position=cue.audio_cue.start_time_ms)
        return final_audio
        
    
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
   