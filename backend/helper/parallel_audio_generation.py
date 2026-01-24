from typing import List, Optional
import concurrent.futures
import logging
import multiprocessing
import threading
from Variable.dataclases import Cue, AudioCueWithAudioBase64
from Variable.configurations import PARALLEL_EXECUTION, PARALLEL_WORKERS
from helper.audio_conversions import audio_to_base64
from Tools.play_audio import create_audio_from_audiocue
from helper.lib import TangoFluxModel, _thread_local

logger = logging.getLogger(__name__)

def process_cue(cue: Cue, worker_id: Optional[int] = None):
    """
    Processes a single cue in a worker thread.
    
    Args:
        cue: Audio cue to process
        worker_id: Optional worker ID for parallel execution (uses model pool)
    """
    try:
        # Store worker_id in thread-local for use in generators
        if worker_id is not None:
            _thread_local.worker_id = worker_id
        
        audio_data = create_audio_from_audiocue(cue)
        base64_data = audio_to_base64(audio_data)
        return AudioCueWithAudioBase64(
            audio_cue=cue,
            audio_base64=base64_data,
            duration_ms=cue.duration_ms,
        )
    except Exception as e:
        logger.error(f"Failed to process cue {getattr(cue, 'id', 'unknown')}: {e}")
        raise


def parallel_audio_generation(cues: List[Cue]):
    """
    Generate audio for multiple cues in parallel or sequentially based on configuration.
    
    If PARALLEL_EXECUTION=True: Uses ThreadPoolExecutor with model pool (one model per worker)
    If PARALLEL_EXECUTION=False: Processes sequentially with single model instance
    """
    if not cues:
        return []
    
    results = []
    
    if PARALLEL_EXECUTION:
        # Parallel mode: use ThreadPoolExecutor with model pool
        max_workers = min(len(cues), PARALLEL_WORKERS)
        
        # Initialize model pool for parallel execution
        TangoFluxModel.initialize_pool(max_workers)
        
        logger.info(
            f"Starting PARALLEL audio generation for {len(cues)} cues with {max_workers} workers"
        )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks with worker IDs
            future_to_cue = {
                executor.submit(process_cue, cue, worker_id % max_workers): cue
                for worker_id, cue in enumerate(cues)
            }

            for future in concurrent.futures.as_completed(future_to_cue):
                cue = future_to_cue[future]
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                        logger.info(
                            f"Successfully generated audio for cue {getattr(cue, 'id', 'N/A')}"
                        )
                except IndexError as e:
                    logger.error(
                        f"IndexError (Scheduler Bug) in cue {getattr(cue, 'id', 'N/A')}: {e}"
                    )
                except Exception as e:
                    logger.error(f"General error in cue {getattr(cue, 'id', 'N/A')}: {e}")
    else:
        # Sequential mode: process one at a time
        logger.info(
            f"Starting SEQUENTIAL audio generation for {len(cues)} cues"
        )
        
        for cue in cues:
            try:
                data = process_cue(cue, worker_id=None)
                if data:
                    results.append(data)
                    logger.info(
                        f"Successfully generated audio for cue {getattr(cue, 'id', 'N/A')}"
                    )
            except IndexError as e:
                logger.error(
                    f"IndexError (Scheduler Bug) in cue {getattr(cue, 'id', 'N/A')}: {e}"
                )
            except Exception as e:
                logger.error(f"General error in cue {getattr(cue, 'id', 'N/A')}: {e}")

    results.sort(key=lambda x: x.audio_cue.start_time_ms)

    logger.info(
        f"Completed audio generation: {len(results)}/{len(cues)} cues generated successfully"
    )

    return results
