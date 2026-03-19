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

import uuid
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    
# Cinemaudio-studio root (for tango_new when using Tango2)
cinema_studio_root = os.path.abspath(os.path.join(project_root, "."))
if cinema_studio_root not in sys.path:
    sys.path.append(cinema_studio_root)    
    
from Variable.configurations import model_config

# Import project-specific modules
from Variable.dataclases import (
    Cue,
    DecideCuesRequest,
    DecideCuesResponse,
    EvaluateAudioRequest,
    EvaluateAudioResponse,
    AudioCueWithAudioBase64,
    GenerateFromStoryRequest,
    GenerateFromStoryResponse,
    GenerateAudioFromCuesRequest,
    GenerateAudioFromCuesResponse,
    GenerateAudioCuesWithAudioBase64Request,
    GenerateAudioCuesWithAudioBase64Response,
    CheckMissingCuesResponse,
)
from helper.audio_conversions import dict_to_cue, audio_cue_to_dict
from superimposition_model.superimposition_model import SuperimpositionModel
from Variable.configurations import READING_SPEED_WPS, model_config
from Tools.decide_audio import decide_audio_cues
from Evaluation.evaluator import AudioEvaluator
from helper.audio_conversions import audio_to_base64
from helper.parallel_audio_generation import parallel_audio_generation

from helper.lib import init_models 
from helper.progress_hub import ensure_channel, publish, sse_subscribe
superimposition_model_ins = SuperimpositionModel()
# Configure logging to explicitly output to stdout/stderr with colored output

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

color_formatter = ColorFormatter('%(name)s - %(levelname)s - %(message)s\n')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(color_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler],
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
    init_models()


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

@app.get("/api/v1/progress/stream")
async def progress_stream(request_id: str, request: Request):
    """
    Server-Sent Events (SSE) stream for per-request progress events.
    Frontend should connect with EventSource and pass request_id.
    """
    ensure_channel(request_id)

    async def event_iter():
        async for chunk in sse_subscribe(request_id):
            # Stop streaming if client disconnected.
            if await request.is_disconnected():
                return
            yield chunk

    return StreamingResponse(
        event_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # If behind nginx, avoid buffering SSE.
            "X-Accel-Buffering": "no",
        },
    )

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
            speed_wps,
            narrator_enabled=model_config.use_narrator,
            movie_bgms_enabled=model_config.use_movie_bgms
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
        request_id = getattr(request, "request_id", None) or str(uuid.uuid4())
        ensure_channel(request_id)
        cues = [dict_to_cue(c.model_dump()) for c in request.cues]
        logger.info(f"Cues converted to dataclasses: {cues}")
        audio_cues = parallel_audio_generation(cues, request_id=request_id, publish_progress=publish)
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

def _request_to_audio_cues(request: GenerateAudioCuesWithAudioBase64Request):
    """Build list of AudioCueWithAudioBase64 from request and compute total_duration_ms."""
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
    ) if audio_cues else 0
    return audio_cues, total_duration_ms


@app.post("/api/v1/check-missing-audio-cues", response_model=CheckMissingCuesResponse)
async def check_missing_audio_cues(request: GenerateAudioCuesWithAudioBase64Request):
    """
    Check which cues are not covered by the story (LLM). Returns missing cues so the frontend
    can show them in a loading state before calling generate-audio-cues-with-audio-base64.
    """
    try:
        audio_cues, total_duration_ms = _request_to_audio_cues(request)
        if not audio_cues:
            return CheckMissingCuesResponse(missing_cues=[], total_duration_ms=0)
        not_covered_classes = superimposition_model_ins.check_missing_audio_cues(
            request.story_text, audio_cues, total_duration_ms
        )
        missing_cues = [
            audio_cue_to_dict(c.audio_cue)
            for c in audio_cues
            if getattr(c.audio_cue, "audio_class", None) in not_covered_classes
        ]
        return CheckMissingCuesResponse(missing_cues=missing_cues, total_duration_ms=total_duration_ms)
    except Exception as e:
        logger.error(f"Error checking missing audio cues: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/generate-audio-cues-with-audio-base64", response_model=GenerateAudioCuesWithAudioBase64Response)
async def generate_audio_cues_with_audio_base64(request: GenerateAudioCuesWithAudioBase64Request):
    """
    Generate audio cues with audio base64 from input cues and story text.
    """
    try:
        logger.info(f"Generating audio cues with audio base64 from story: {request.story_text[:50]}...")
        audio_cues, total_duration_ms = _request_to_audio_cues(request)
        logger.info(f"superimposed cues: {len(audio_cues)}")

        if model_config.fill_coverage_by_llm and audio_cues:
            not_covered_classes = superimposition_model_ins.check_missing_audio_cues(
                request.story_text, audio_cues, total_duration_ms
            )
            if not_covered_classes:
                logger.info(f"Not covered audio cues: {not_covered_classes}")
                missing_cue_objects = [
                    c.audio_cue for c in audio_cues
                    if getattr(c.audio_cue, "audio_class", None) in not_covered_classes
                ]
                generated_missing = parallel_audio_generation(missing_cue_objects)
                audio_cues.extend(generated_missing)

        final_audio = superimposition_model_ins.superimpose_audio_cues_with_audio_base64(
            request.story_text, audio_cues, total_duration_ms
        )
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
        
        final_audio = superimposition_model_ins.superimposition_model(request.story_text, speed_wps)
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