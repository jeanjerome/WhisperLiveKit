#!/usr/bin/env python3
# main.py - Entry point for WhisperLiveKit server

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.websocket import WebSocketHandler
from config import load_config_from_args, setup_cli_parser, setup_logging
from core.state import SessionManager
from services.audio import AudioService
from services.diarization import DiarizationService
from services.transcription import TranscriptionService

# Global service instances
audio_service = None
transcription_service = None
diarization_service = None
session_manager = None
websocket_handler = None
app_config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the FastAPI application lifecycle.
    Initializes services at startup and cleans them up on shutdown.
    """
    global audio_service, transcription_service, diarization_service
    
    # Initialize services based on configuration
    audio_service = await initialize_audio()
    
    if app_config.asr.enabled:
        transcription_service, _ = await initialize_transcription()
    
    if app_config.diarization.enabled:
        diarization_service = await initialize_diarization()
        
    yield
    
    # Services handle their own cleanup mechanisms

async def initialize_audio():
    """
    Initializes the audio processing service.
    
    
    Returns:
        AudioService: Configured audio service
    """
    global app_config
    
    # Configure audio service based on app configuration
    audio_config = {
        "sample_rate": app_config.audio.sample_rate,
        "channels": app_config.audio.channels,
        "sample_width": app_config.audio.sample_width,
        "audio": app_config.audio.audio
    }
    
    service = AudioService(audio_config)
    
    # Initialize the service
    success = await service.initialize()
    if not success:
        if app_config.audio.audio == "pyaudio":
            audio_config["audio"] = "ffmpeg"
            service = AudioService(audio_config)
            success = await service.initialize()
            if not success:
                raise RuntimeError("No audio backend available")
        else:
            raise RuntimeError("Failed to initialize audio backend. Check your installation.")
    
    return service

async def initialize_transcription():
    """
    Initializes the transcription service and loads models.
    
    Returns:
        tuple: (service, tokenizer)
    """
    global app_config
    
    asr_config = app_config.asr.model_dump()
    
    # Adapt parameter names for backend compatibility
    asr_config["lan"] = asr_config.pop("language", "auto")
    
    service = TranscriptionService(asr_config)
    await service.initialize()
    
    # Warm up the model if a file is specified
    if app_config.server.warmup_file:
        await service.warmup(app_config.server.warmup_file)
        
    return service, service.tokenizer


async def initialize_diarization():
    """
    Initializes the diarization service.
    
    Returns:
        DiarizationService: Configured diarization service
    """
    global app_config
    
    service = DiarizationService(app_config.diarization.model_dump())
    await service.initialize()
    return service


def load_html_template():
    """
    Loads the HTML template for the home page.
    
    Returns:
        str: HTML content for the home page
    """
    try:
        file_path = os.path.join("web", "live_transcription.html")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to load HTML template: {e}")
        return "<html><body><h1>Error: HTML template not found</h1></body></html>"


async def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    global session_manager, websocket_handler, audio_service, transcription_service, diarization_service
    
    # Create application with lifecycle manager
    app = FastAPI(lifespan=lifespan)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize services if not already done
    if audio_service is None:
        audio_service = await initialize_audio()
    
    if app_config.asr.enabled and transcription_service is None:
        transcription_service, _ = await initialize_transcription()
    
    if app_config.diarization.enabled and diarization_service is None:
        diarization_service = await initialize_diarization()
    
    # Create session manager
    session_manager = SessionManager()
    
    # Create WebSocket handler after services initialization
    websocket_handler = WebSocketHandler(
        session_manager=session_manager,
        audio_service=audio_service,
        transcription_service=transcription_service,
        diarization_service=diarization_service,
        config={
            "sample_rate": app_config.audio.sample_rate,
            "channels": app_config.audio.channels,
            "min_chunk_size": app_config.asr.min_chunk_size
        }
    )
    
    # Load HTML template
    html_content = load_html_template()
    
    # Define routes
    @app.get("/")
    async def get_root():
        """Returns the HTML UI page."""
        return HTMLResponse(html_content)
    
    @app.websocket("/asr")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time audio transcription."""
        await websocket_handler.handle_connection(websocket)
        
    # Add an endpoint to get information about the audio backend
    @app.get("/api/info")
    async def get_info():
        """Returns information about the system configuration."""
        return {
            "audio_backend": app_config.audio.audio,
            "transcription_enabled": app_config.asr.enabled,
            "transcription_model": app_config.asr.model,
            "transcription_backend": app_config.asr.backend,
            "diarization_enabled": app_config.diarization.enabled,
        }
    
    return app


def main():
    """
    Main entry point for the application.
    Parses arguments, configures the application, and starts the server.
    """
    global app_config
    
    # Parse command line arguments
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Load configuration
    app_config = load_config_from_args(args)
    
    # Set up logging
    logger = setup_logging(app_config.server.log_level)
    logger.info("Starting WhisperLiveKit Server")
    
    # Log which audio backend is being used
    logger.info(f"Using {app_config.audio.audio} audio backend")
    
    # Create and run the application with uvicorn
    app = asyncio.run(create_app())
    
    uvicorn.run(
        app,
        host=app_config.server.host,
        port=app_config.server.port,
        log_level=app_config.server.log_level.lower()
    )


if __name__ == "__main__":
    main()