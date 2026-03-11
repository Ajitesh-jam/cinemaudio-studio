from typing import Optional, Type, List, Tuple
import concurrent.futures
import threading
import os
import logging
import tempfile

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from Variable.configurations import (
    TANGO_FLUX,
    ELEVEN_LABS,
    TANGO2,
    PARALLEL_EXECUTION,
    PARALLEL_WORKERS,
    SFX_RATE,
)
load_dotenv()

from model.tangoflux_model import TangoFluxModel
from model.elevenlabs_model import ElevenLabsModel
from model.tango2_model import Tango2Model
from superimposition_model.superimposition_model import SuperimpositionModel
from model.tangoflux_model import TangoFluxModel
# from model.elevenlabs_model import ElevenLabsModel
from model.tango2_model import Tango2Model
from model.parlerTTSModel import ParlerTTSModel
# Thread-local storage for worker IDs
_thread_local = threading.local()

logger = logging.getLogger(__name__)

parler_tts_model_ins = ParlerTTSModel()
tango2_model_ins = Tango2Model()
tango_flux_model_ins = TangoFluxModel()
superimposition_model_ins = SuperimpositionModel()

# Private dictionary to hold our preloaded model instances
_model_registry = {}

def init_models():
    """Instantiate and preload all models into memory."""
    print("Loading models into memory... This might take a moment.")
    
        # Preload ParlerTTS model (optional: may fail if transformers lacks encodec)
    try:
        parler_tts_model_ins.get_instance()
        logger.info("Preloaded ParlerTTS model")
    except Exception as e:
        logger.warning("ParlerTTS preload skipped (will lazy-load on first use or fail then): %s", e)

    # Preload Tango2 model
    try:
        tango2_model_ins.get_instance()
        logger.info("Preloaded Tango2 model")
    except Exception as e:
        logger.warning("Tango2 preload skipped (will lazy-load on first use or fail then): %s", e)

    # Preload TangoFlux models based on execution mode (optional: may fail if tangoflux/diffusers incompatible)
    try:
        if PARALLEL_EXECUTION:
            logger.info(f"Pre-initializing TangoFlux model pool with {PARALLEL_WORKERS} workers for parallel execution...")
            TangoFluxModel.initialize_pool(PARALLEL_WORKERS)
            logger.info(f"Preloaded {PARALLEL_WORKERS} TangoFlux model instances for parallel execution")
        else:
            TangoFluxModel.get_instance()
            logger.info("Preloaded TangoFlux model (sequential mode)")
    except Exception as e:
        logger.warning("TangoFlux preload skipped (use Tango2 for SFX if needed): %s", e)

    logger.info("Specialist model preload finished.\n\n")
    
    # Initialize instances here. 
    # This is where they are loaded into RAM/VRAM.
    _model_registry["parlertts"] = parler_tts_model_ins
    _model_registry["tango2"] = tango2_model_ins
    _model_registry["tangoflux"] = tango_flux_model_ins
    _model_registry["superimposition"] = superimposition_model_ins
    # _model_registry["elevenlabs"] = eleven_labs_model_ins
    
    print("All models successfully loaded!")

def get_model(model_name: str):
    """Return the model class for the given model name.

    Args:
        model_name: One of "TangoFlux", "ElevenLabs", or "Tango2" (case-insensitive).

    Returns:
        The model class (e.g. TangoFluxModel, ElevenLabsModel, Tango2Model).

    Example:
        model_cls = get_model("Tango2")
        audio = model_cls.generate(prompt, steps=100, duration=5)
    """
    
    logger.info(f"Getting model: {model_name.strip().lower()}")
    logger.info(f"Model registry: {_model_registry[model_name.strip().lower()]}")
    return _model_registry[model_name.strip().lower()]


def generate_sound(
    prompt: str,
    steps: int = 100,
    duration: int = 10,
    model_name: str = TANGO2,
    worker_id: Optional[int] = None,
    **kwargs,
):
    """Generate sound effect using the specified model.

    Args:
        prompt: Text description of the sound to generate.
        steps: Number of diffusion steps (used by TangoFlux and Tango2).
        duration: Duration in seconds (used by TangoFlux and Tango2).
        model_name: "TangoFlux", "ElevenLabs", or "Tango2".
        worker_id: Optional worker ID for TangoFlux parallel execution.
        **kwargs: Passed through to the model's generate().

    Returns:
        Audio tensor or bytes depending on the model.
    """
    model_cls = get_model(model_name)
    if model_name.strip().lower() == "elevenlabs":
        return model_cls.generate(prompt, **kwargs)
    return model_cls.generate(prompt, steps=steps, duration=duration, worker_id=worker_id, **kwargs)

