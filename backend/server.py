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
import base64
import io
from typing import List, Optional, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from pydub import AudioSegment

# Import project-specific modules
from Variable.dataclases import (
    AudioCue,
    AudioCueWithAudioBase64,
    DecideCuesRequest,
    DecideCuesResponse,
    GenerateAudioFromCuesRequest,
    GenerateAudioFromCuesResponse,
    GenerateFromStoryRequest,GenerateFromStoryResponse,
    GenerateAudioCuesWithAudioBase64Request,GenerateAudioCuesWithAudioBase64Response
)
from Variable.configurations import READING_SPEED_WPS
from Tools.decide_audio import decide_audio_cues
from Tools.play_audio import create_audio_from_audiocue
from superimposition_model.superimposition_model import superimpose_audio_cues, superimpose_audio_cues_with_audio_base64,superimposition_model

# Configure logging to explicitly output to stdout/stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout for all logs
    ],
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)
# Ensure logger outputs to stdout
logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="Audio Generation API",
    description="API for generating audio cues and final audio from story text",
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


# Helper function to convert AudioCue to dict
def audio_cue_to_dict(cue: AudioCue) -> dict:
    """Convert AudioCue to dictionary"""
    return {
        "audio_class": cue.audio_class,
        "audio_type": cue.audio_type,
        "start_time_ms": cue.start_time_ms,
        "duration_ms": cue.duration_ms,
        "weight_db": cue.weight_db,
        "fade_ms": cue.fade_ms
    }

# Helper function to convert AudioSegment to base64
def audio_to_base64(audio: AudioSegment, format: str = "wav") -> str:
    """Convert AudioSegment to base64 encoded string"""
    buffer = io.BytesIO()
    audio.export(buffer, format=format)
    buffer.seek(0)
    audio_bytes = buffer.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    return audio_base64

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
        logger.info(f"Generating audio from {len(request.cues)} cues")
        cues = [
            AudioCue(
                id=cue.id,
                audio_class=cue.audio_class,
                audio_type=cue.audio_type,
                start_time_ms=cue.start_time_ms,
                duration_ms=cue.duration_ms,
                weight_db=cue.weight_db,
                fade_ms=cue.fade_ms
            )
            for cue in request.cues
        ]
        audio_cues = []
        for cue in cues:
            audio_cues.append(
                AudioCueWithAudioBase64(
                    audio_cue=cue,
                    audio_base64=audio_to_base64(create_audio_from_audiocue(cue)),
                    duration_ms=cue.duration_ms
                )
            )
        
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
    Generate audio cues with audio base64 from story text.
    """
    try:
        logger.info(f"Generating audio cues with audio base64 from story: {request.story_text[:50]}...")
        audio_cues = [
            AudioCueWithAudioBase64(
                audio_cue=AudioCue(
                    id=cue.id,
                    audio_class=cue.audio_class,
                    audio_type=cue.audio_type,
                    start_time_ms=cue.start_time_ms,
                    duration_ms=cue.duration_ms,
                    weight_db=cue.weight_db,
                    fade_ms=cue.fade_ms
                ),
                audio_base64=cue.audio_base64,
                duration_ms=cue.duration_ms
            )
            for cue in request.cues
        ]
        final_audio = superimpose_audio_cues_with_audio_base64(audio_cues, request.total_duration_ms)
        return GenerateAudioCuesWithAudioBase64Response(audio_base64=audio_to_base64(final_audio),message="Successfully generated audio cues with audio base64")
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
        return GenerateFromStoryResponse(audio_base64=audio_to_base64(final_audio),message="Successfully generated audio from story")
    except Exception as e:
        logger.error(f"Error generating audio from story: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating audio from story: {str(e)}"
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