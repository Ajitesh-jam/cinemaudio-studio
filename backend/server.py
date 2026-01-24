"""
FastAPI Server for Audio Generation System
Provides API endpoints for:
1. Deciding audio cues from story text
2. Generating audio from audio cues
3. Generating final superimposed audio
"""

import os
import sys
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project-specific modules
from Variable.dataclases import (
    AudioCue,
    AudioCueWithAudioBase64,
    Cue,
    DecideCuesRequest,
    DecideCuesResponse,
    EvaluateAudioRequest,
    EvaluateAudioResponse,
    GenerateAudioFromCuesRequest,
    GenerateAudioFromCuesResponse,
    GenerateFromStoryRequest,
    GenerateFromStoryResponse,
    GenerateAudioCuesWithAudioBase64Request,
    GenerateAudioCuesWithAudioBase64Response
)
from helper.audio_conversions import dict_to_cue

from Variable.configurations import READING_SPEED_WPS, PARALLEL_EXECUTION, PARALLEL_WORKERS
from Tools.decide_audio import decide_audio_cues
from superimposition_model.superimposition_model import superimpose_audio_cues, superimpose_audio_cues_with_audio_base64,superimposition_model
from Evaluation.evaluator import AudioEvaluator
from helper.audio_conversions import audio_to_base64
from helper.parallel_audio_generation import parallel_audio_generation
from helper.lib import TangoFluxModel, ParlerTTSModel

# Configure logging to explicitly output to stdout/stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s\n',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout for all logs
    ],
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="Audio Generation API",
    description="Background Mellow APIs for a cinematic storytelling experience",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def preload_models():
    """Preload specialist models at startup so they are downloaded once before first request."""
    logger.info("Preloading specialist models...")
    
    # Preload ParlerTTS model (always needed)
    ParlerTTSModel.get_instance()
    logger.info("Preloaded ParlerTTS model")
    
    # Preload TangoFlux models based on execution mode
    if PARALLEL_EXECUTION:
        # Parallel mode: pre-initialize model pool with PARALLEL_WORKERS instances
        logger.info(f"Pre-initializing TangoFlux model pool with {PARALLEL_WORKERS} workers for parallel execution...")
        TangoFluxModel.initialize_pool(PARALLEL_WORKERS)
        logger.info(f"Preloaded {PARALLEL_WORKERS} TangoFlux model instances for parallel execution")
    else:
        # Sequential mode: preload single instance
        TangoFluxModel.get_instance()
        logger.info("Preloaded TangoFlux model (sequential mode)")
    
    logger.info("All specialist models preloaded\n\n")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Audio Generation API",
        "version": "1.0.0",
        "endpoints": {
            "decide_cues": "/api/v1/decide-cues",
            "generate_audio": "/api/v1/generate-audio",
            "generate_from_story": "/api/v1/generate-from-story",
            "health": "/api/v1/health"
        }
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/decide-cues", response_model=DecideCuesResponse)
async def decide_audio_cues_handler(request: DecideCuesRequest):
    """
    Decide audio cues from story text.
    
    This endpoint analyzes the story text and returns a list of audio cues
    with timing information.
    """
    try:
        logger.info(f"Deciding audio cues for story: {request.story_text[:50]}...")
        speed_wps = request.speed_wps if request.speed_wps is not None else READING_SPEED_WPS
        cues, total_duration = decide_audio_cues(
            request.story_text,
            speed_wps
        )
        return DecideCuesResponse(
            cues=cues,
            total_duration_ms=total_duration,
            message=f"Successfully generated {len(cues)} audio cues"
        )

    except Exception as e:
        logger.error(f"Error deciding audio cues: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing story: {str(e)}"
        )

@app.post("/api/v1/generate-audio", response_model=GenerateAudioFromCuesResponse)
async def generate_audio_from_cues_handler(request: GenerateAudioFromCuesRequest):
    """
    Generate final superimposed audio from audio cues.

    This endpoint takes a list of audio cues and generates the final
    superimposed audio track.
    """
    try:
        
        logger.info(f"Request: {request}\n\n")
        
        
        logger.info(f"Generating audio from {len(request.cues)} cues")
        cues = [dict_to_cue(c.model_dump()) for c in request.cues]
        audio_cues = parallel_audio_generation(cues)
        return GenerateAudioFromCuesResponse(
            audio_cues=audio_cues,
            message="Successfully generated audio"
        )
    
    except Exception as e:
        logger.error(f"Error generating audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating audio: {str(e)}"
        )

@app.post("/api/v1/generate-audio-cues-with-audio-base64", response_model=GenerateAudioCuesWithAudioBase64Response)
async def generate_audio_cues_with_audio_base64(request: GenerateAudioCuesWithAudioBase64Request):
    """
    Generate audio cues with audio base64 from input cues and story text.
    """
    try:
        logger.info(f"Generating audio cues with audio base64 from story: {request.story_text[:50]}...")
        audio_cues = []
        for cue in request.cues:
            raw = cue.audio_cue
            if hasattr(raw, "__dataclass_fields__"):
                resolved: Cue = raw  # type: ignore[assignment]
            else:
                if hasattr(raw, "model_dump"):
                    d = raw.model_dump()  # type: ignore[union-attr]
                elif hasattr(raw, "keys"):
                    d = dict(raw)  # type: ignore[arg-type]
                else:
                    d = {f: getattr(raw, f, None) for f in ("id", "audio_type", "start_time_ms", "duration_ms", "audio_class", "weight_db", "fade_ms", "story", "narrator_description")}
                resolved = dict_to_cue(d)
            audio_cues.append(
                AudioCueWithAudioBase64(
                    audio_cue=resolved,
                    audio_base64=cue.audio_base64,
                    duration_ms=cue.duration_ms
                )
            )
        total_duration_ms = max(
            (c.audio_cue.start_time_ms + c.audio_cue.duration_ms) for c in audio_cues
        )

        final_audio = superimpose_audio_cues_with_audio_base64(audio_cues, total_duration_ms)
        return GenerateAudioCuesWithAudioBase64Response(
            audio_base64=audio_to_base64(final_audio),
            message="Successfully generated audio cues with audio base64",
        )
    except Exception as e:
        logger.error(f"Error generating audio cues with audio base64: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating audio cues with audio base64: {str(e)}"
        )

@app.post("/api/v1/generate-from-story", response_model=GenerateFromStoryResponse)
async def generate_from_story(request: GenerateFromStoryRequest):
    """
    Complete pipeline: Generate audio from story text.
    
    This endpoint combines deciding audio cues and generating final audio
    in a single call.
    """
    try:
        logger.info(f"Generating audio from story: {request.story_text[:50]}...")
        
        # Step 1: Decide audio cues
        speed_wps = request.speed_wps if request.speed_wps is not None else READING_SPEED_WPS
        
        final_audio = superimposition_model(request.story_text, speed_wps)
        return GenerateFromStoryResponse(audio_base64=audio_to_base64(final_audio))
    except Exception as e:
        logger.error(f"Error generating audio from story: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating audio from story: {str(e)}"
        )

@app.post("/api/v1/evaluate-audio", response_model=EvaluateAudioResponse)
async def evaluate_audio(request: EvaluateAudioRequest):
    """
    Evaluate audio based on text and audio base64.
    Returns CLAP score, spectral richness, noise floor, and audio onsets.
    """
    try:
        logger.info("Evaluating audio...")
        evaluator = AudioEvaluator()
        
        # Get CLAP score (text-audio alignment)
        clap_score = evaluator.get_clap_score(request.audio_base64, request.text)
        
        # Get spectral richness (returns flatness, entropy)
        flatness, spectral_entropy = evaluator.get_audio_richness(request.audio_base64)
        
        # Get noise floor
        noise_floor = evaluator.get_noise_floor(request.audio_base64)
        
        # Get audio onsets (sync detection)
        audio_onsets = evaluator.evaluate_sync_from_audio_base64(request.audio_base64)
        
        return EvaluateAudioResponse(
            clap_score=float(clap_score),
            spectral_richness=float(spectral_entropy),  # Use entropy as spectral richness
            noise_floor=float(noise_floor),
            audio_onsets=int(audio_onsets),
            message="Successfully evaluated audio"
        )   
    except Exception as e:
        logger.error(f"Error evaluating audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating audio: {str(e)}"
        )

if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info("No authentication required - API is open")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )