from typing import List
from Variable.dataclases import AudioCue
from Variable.dataclases import AudioCueWithAudioBase64
from helper.audio_conversions import audio_to_base64
from Tools.play_audio import create_audio_from_audiocue
import concurrent.futures
import logging
import threading

logger = logging.getLogger(__name__)

# Lock to serialize model access (TangoFlux models are not thread-safe)
_model_lock = threading.Lock()

def process_cue(cue):
    """
    Process a single audio cue and generate audio.
    Uses a lock to serialize model access since TangoFlux models are not thread-safe.
    """
    # Serialize access to models to avoid "Already borrowed" errors
    with _model_lock:
        try:
            return AudioCueWithAudioBase64(
                audio_cue=cue,
                audio_base64=audio_to_base64(create_audio_from_audiocue(cue)),
                duration_ms=cue.duration_ms
            )
        except IndexError as e:
            # Handle scheduler IndexError (step_index out of bounds)
            if "out of bounds" in str(e) and ("step_index" in str(e) or "dimension" in str(e)):
                logger.warning(f"IndexError in scheduler for cue {cue.id}, this may be a tangoflux scheduler bug. Error: {e}")
                # Re-raise to let the caller handle it
                raise
            else:
                # Re-raise other IndexErrors
                raise
        except Exception as e:
            logger.error(f"Error processing cue {cue.id}: {e}")
            raise

def parallel_audio_generation(cues: List[AudioCue]):
    """
    Generates audio for each cue in parallel and returns a list of AudioCueWithAudioBase64.
    Handles errors gracefully and continues processing other cues.
    """
    results = []
    errors = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map each cue to a future
        futures = {executor.submit(process_cue, cue): cue for cue in cues}
        for future in concurrent.futures.as_completed(futures):
            cue = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Processed cue {result.audio_cue.id}")
            except IndexError as e:
                error_msg = f"IndexError processing cue {cue.id}: {e}"
                logger.error(error_msg)
                errors.append((cue.id, error_msg))
            except Exception as e:
                error_msg = f"Error processing cue {cue.id}: {e}"
                logger.error(error_msg)
                errors.append((cue.id, error_msg))
    
    if errors:
        logger.warning(f"Failed to process {len(errors)} cues: {[cue_id for cue_id, _ in errors]}")
    
    logger.info(f"Successfully processed {len(results)} out of {len(cues)} cues")
    return results
        