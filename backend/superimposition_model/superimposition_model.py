# import sys
# import os

# # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)
# print("Project root added to sys.path:", project_root)

from csv import Error
from operator import and_
from pydub import AudioSegment
import logging
from typing import List, Sequence, cast
from Variable.dataclases import Cue, AudioCueWithAudioBase64, AudioCue
from Tools.play_audio import create_audio_from_audiocue
from helper.parallel_audio_generation import parallel_audio_generation
from Tools.decide_audio import decide_audio_cues
from Variable.configurations import READING_SPEED_WPS
from helper.audio_conversions import base64_to_audio, audio_to_base64
from Utils.prompts import prompt_to_fill_missing_audio_cues
from Utils.llm import query_llm
from helper.lib import read_movie_bgms_csv
from pydub.scipy_effects import high_pass_filter
from pydub.effects import normalize, compress_dynamic_range
from Variable.configurations import ModelConfig
model_config = ModelConfig()
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
            movie_bgms_csv = read_movie_bgms_csv()
            prompt = prompt_to_fill_missing_audio_cues.format(story_text=story_text, audio_cues=audio_cues_to_fill, movie_bgms_csv=movie_bgms_csv)
            response = query_llm(llm_name="gemini", model_name="gemini-2.5-flash", prompt=prompt)
            if response:
                not_covered_audio_cues = response.get("audio_cues", [])
        
        logger.info(f"Not covered audio cues: {not_covered_audio_cues}")        
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
    
   # Assuming AudioCueWithAudioBase64 and base64_to_audio are defined in your project
    def superimpose_audio_cues_with_audio_base64_with_dsp(self, story_text: str, audio_cues: List, total_duration_ms: int):
        """
        Superimposes audio cues using cinematic DSP: Fades, EQ, Ducking, and Compression.
        """
        logger.info("Starting cinematic DSP audio superimposition...")
        
        # We build two separate canvases: one for the voice, one for the background
        narrator_mix = AudioSegment.silent(duration=total_duration_ms)
        background_mix = AudioSegment.silent(duration=total_duration_ms)
        
        # Keep track of when the narrator is speaking for our Ducking algorithm
        narrator_timestamps = []

        # --- STEP 1: Process Individual Cues (Fades & EQ) ---
        for cue in audio_cues:
            base_segment = base64_to_audio(cue.audio_base64)
            cue_type = getattr(cue.audio_cue, "type", "sfx").lower() # e.g., 'narrator', 'bgm', 'sfx'
            
            # 1a. Apply Baseline Gain
            weight_db = getattr(cue.audio_cue, "weight_db", 0) or 0
            segment_with_gain = base_segment + weight_db

            # 1b. Smooth Trimming (Prevent clicks/pops)
            desired_duration = getattr(cue.audio_cue, "duration_ms", len(segment_with_gain)) or len(segment_with_gain)
            if len(segment_with_gain) > desired_duration:
                audio_segment = segment_with_gain[:desired_duration]
            elif len(segment_with_gain) < desired_duration:
                padding = AudioSegment.silent(duration=desired_duration - len(segment_with_gain))
                audio_segment = segment_with_gain + padding
            else:
                audio_segment = segment_with_gain

            # Help the type checker: ensure this is treated as an AudioSegment
            audio_segment = cast(AudioSegment, audio_segment)

            # 1c. Apply Fades (50ms for SFX/Voice to prevent pops, 500ms for Music/Ambience)
            fade_ms = getattr(cue.audio_cue, "fade_ms", None)
            if fade_ms is None:
                fade_ms = 500 if cue_type in ["bgm", "ambience"] else 50
            
            # Ensure fade isn't longer than the audio itself
            fade_ms = min(fade_ms, len(audio_segment) // 2)
            if fade_ms > 0:
                audio_segment = audio_segment.fade_in(fade_ms).fade_out(fade_ms)

            # 1d. Apply EQ (Frequency Masking) & Route to correct mix
            start_time = cue.audio_cue.start_time_ms
            
            if cue_type == "narrator":
                narrator_mix = narrator_mix.overlay(audio_segment, position=start_time)
                narrator_timestamps.append((start_time, start_time + desired_duration))
            else:
                # High-Pass Filter BGM/Ambience at 250Hz to leave bass/mids open for the narrator's voice
                if cue_type in ["bgm", "ambience"]:
                    try:
                        audio_segment = high_pass_filter(audio_segment, cutoff_freq=250)
                    except Exception as e:
                        logger.warning(f"Could not apply high pass filter: {e}")
                
                background_mix = background_mix.overlay(audio_segment, position=start_time)

        # --- STEP 2: Automatic Ducking (Modulation) ---
        # We lower the background_mix volume by 8dB whenever the narrator speaks
        ducking_reduction_db = -3.0
        duck_fade_ms = 300 # Smooth transition in and out of the duck
        
        for (start, end) in narrator_timestamps:
            # Define the ducking window (slightly before and after the speech)
            duck_start = max(0, start - duck_fade_ms)
            duck_end = min(len(background_mix), end + duck_fade_ms)
            
            # Slice the background mix, reduce its volume, and fade the edges
            if duck_start < duck_end:
                # Help the type checker by treating the slice as an AudioSegment
                slice_seg = cast(AudioSegment, background_mix[duck_start:duck_end])
                ducked_slice = cast(AudioSegment, slice_seg + ducking_reduction_db)

                # Treat all pieces as AudioSegment
                prefix = cast(AudioSegment, background_mix[:duck_start])
                middle = cast(AudioSegment, ducked_slice)
                suffix = cast(AudioSegment, background_mix[duck_end:])
                
                # Crossfade the ducked slice back into the main background mix to avoid harsh drops
                background_mix = prefix.append(middle, crossfade=min(duck_fade_ms, len(middle)//2)).append(suffix, crossfade=0)

        # --- STEP 3: Master Superimposition & Post-Processing ---
        logger.info("Overlaying narrator on ducked background mix...")
        final_audio = background_mix.overlay(narrator_mix, position=0)

        # 3a. Master Bus Compression (Glues the track together and tames loud peaks)
        final_audio = compress_dynamic_range(final_audio, threshold=-15.0, ratio=3.0, attack=5.0, release=50.0)

        # 3b. Peak Normalization (Ensure the final track is loud and clear, peaking at -1.0 dBFS)
        final_audio = normalize(final_audio, headroom=1.0)

        logger.info("Audio superimposition complete.")
        return final_audio    

            
    def normal_superimpose_audio_cues_with_audio_base64(self, story_text: str, audio_cues: List[AudioCueWithAudioBase64], total_duration_ms: int):
            
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
    
    
    def superimpose_audio_cues_with_audio_base64(self, story_text: str, audio_cues: List[AudioCueWithAudioBase64], total_duration_ms: int):
        """
        Superimposes all audio cues with audio base64 into a single track.
        """
        logger.info("Starting audio superimposition process...")
        logger.info(f"Creating silent audio canvas of {total_duration_ms}ms.")
        
        if model_config.use_dsp:
            return self.superimpose_audio_cues_with_audio_base64_with_dsp(story_text, audio_cues, total_duration_ms)
        else:
            return self.normal_superimpose_audio_cues_with_audio_base64(story_text, audio_cues, total_duration_ms)
        
            
        
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
   