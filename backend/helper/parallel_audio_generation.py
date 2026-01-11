from typing import List
import concurrent.futures
import logging
import multiprocessing
from Variable.dataclases import AudioCue, AudioCueWithAudioBase64
from helper.audio_conversions import audio_to_base64
from Tools.play_audio import create_audio_from_audiocue

logger = logging.getLogger(__name__)

def process_cue(cue: AudioCue):
    """
    Processes a single cue in a separate process.
    Each process has its own memory space and model instance, enabling true parallelism.
    """
    try:
        # Generate audio (each process has its own model instance)
        audio_data = create_audio_from_audiocue(cue)
        
        # Convert to base64
        base64_data = audio_to_base64(audio_data)
        
        return AudioCueWithAudioBase64(
            audio_cue=cue,
            audio_base64=base64_data,
            duration_ms=cue.duration_ms
        )
    except Exception as e:
        logger.error(f"Failed to process cue {getattr(cue, 'id', 'unknown')}: {e}")
        raise  # Re-raise to be caught by the executor loop

def parallel_audio_generation(cues: List[AudioCue]):
    """
    Generate audio for multiple cues in parallel using ProcessPoolExecutor.
    Each process runs independently with its own model instance, enabling true parallelism.
    """
    if not cues:
        return []
    
    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    # Each process has its own memory space and can load its own model instance
    # Limit workers based on CPU cores and available memory/VRAM
    max_workers = min(len(cues), multiprocessing.cpu_count(), 4)  # Cap at 4 to avoid memory issues
    
    logger.info(f"Starting parallel audio generation for {len(cues)} cues with {max_workers} workers")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_cue = {executor.submit(process_cue, cue): cue for cue in cues}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_cue):
            cue = future_to_cue[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
                    logger.info(f"Successfully generated audio for cue {getattr(cue, 'id', 'N/A')}")
            except IndexError as e:
                # Specific handling for the TangoFlux scheduler bug
                logger.error(f"IndexError (Scheduler Bug) in cue {getattr(cue, 'id', 'N/A')}: {e}")
            except Exception as e:
                logger.error(f"General error in cue {getattr(cue, 'id', 'N/A')}: {e}")

    # Sort results by the original cue order to maintain timeline sequence
    results.sort(key=lambda x: x.audio_cue.start_time_ms)
    
    logger.info(f"Completed parallel audio generation: {len(results)}/{len(cues)} cues generated successfully")
    
    return results