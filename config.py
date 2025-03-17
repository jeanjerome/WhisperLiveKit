#!/usr/bin/env python3
# config.py - Configuration management for WhisperLiveKit

import argparse
import logging
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ServerConfig(BaseModel):
    """Server configuration parameters."""
    host: str = "localhost"
    port: int = 8000
    warmup_file: Optional[str] = None
    log_level: str = "INFO"
    
    @field_validator('log_level')
    def check_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v

class AudioConfig(BaseModel):
    """Audio processing configuration parameters."""
    audio: str = "ffmpeg"
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit audio
    
    @field_validator('audio')
    def check_audio(cls, v):
        if v not in ["ffmpeg", "pyannote"]:
            raise ValueError("audio must be either 'ffmpeg' or 'pyannote'")
        return v
    

class ASRConfig(BaseModel):
    """Automatic Speech Recognition (ASR) configuration parameters."""
    enabled: bool = True
    model: str = "tiny"
    model_cache_dir: Optional[str] = None
    model_dir: Optional[str] = None
    language: str = "auto"
    task: str = "transcribe"
    backend: str = "faster-whisper"
    vad: bool = True
    vac: bool = False
    vac_chunk_size: float = 0.04
    min_chunk_size: float = 1.0
    buffer_trimming: str = "segment"
    buffer_trimming_sec: float = 15.0
    confidence_validation: bool = False
    
    @field_validator('backend')
    def check_backend(cls, v):
        valid_backends = ["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"]
        if v not in valid_backends:
            raise ValueError(f"Backend must be one of {valid_backends}")
        return v
        
    @field_validator('buffer_trimming')
    def check_buffer_trimming(cls, v):
        if v not in ["sentence", "segment"]:
            raise ValueError("buffer_trimming must be either 'sentence' or 'segment'")
        return v
        
    @field_validator('model')
    def check_model(cls, v):
        valid_models = "tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(",")
        if v not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")
        return v


class DiarizationConfig(BaseModel):
    """Speaker diarization configuration parameters."""
    enabled: bool = False
    sample_rate: int = 16000
    use_microphone: bool = False


class AppConfig(BaseModel):
    """Global application configuration."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)


def setup_cli_parser() -> argparse.ArgumentParser:
    """
    Configure the command line argument parser.
    Uses the same arguments as the original script.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
    
    # Server arguments
    parser.add_argument("--host", type=str, default="localhost", 
                      help="The host address to bind the server to.")
    parser.add_argument("--port", type=int, default=8000, 
                      help="The port number to bind the server to.")
    parser.add_argument("--warmup-file", type=str, default=None, dest="warmup_file",
                      help="The path to a speech audio wav file to warm up Whisper.")
    
    # Audio backend arguments
    parser.add_argument("--audio", type=str, default="ffmpeg",
                      help="The audio library to capture microphone access (ffmpeg or pyaudio).")

    # Application-specific arguments
    parser.add_argument("--transcription", type=bool, default=True,
                      help="To disable to only see live diarization results.")
    parser.add_argument("--diarization", type=bool, default=False,
                      help="Whether to enable speaker diarization.")
    parser.add_argument("--confidence-validation", type=bool, default=False,
                      help="Accelerates validation of tokens using confidence scores.")
    
    # Include ASR arguments from whisper_streaming_custom module
    from whisper_streaming_custom.whisper_online import add_shared_args
    add_shared_args(parser)
    
    return parser


def load_config_from_args(args: argparse.Namespace) -> AppConfig:
    """
    Load configuration from CLI arguments.
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        AppConfig: Application configuration
    """
    # Configure server
    server_dict = {
        "host": args.host,
        "port": args.port,
        "warmup_file": args.warmup_file,
        "log_level": args.log_level if hasattr(args, 'log_level') else "INFO"
    }
    
    # Configure audio
    audio_dict = {
        "audio": args.audio
    }
    
    # Configure ASR
    asr_dict = {
        "enabled": args.transcription,
        "model": args.model,
        "model_cache_dir": args.model_cache_dir,
        "model_dir": args.model_dir,
        "language": args.lan,
        "task": args.task,
        "backend": args.backend,
        "vad": args.vad,
        "vac": args.vac,
        "vac_chunk_size": args.vac_chunk_size,
        "min_chunk_size": args.min_chunk_size,
        "buffer_trimming": args.buffer_trimming,
        "buffer_trimming_sec": args.buffer_trimming_sec,
        "confidence_validation": args.confidence_validation
    }
    
    # Configure diarization
    diarization_dict = {
        "enabled": args.diarization
    }
    
    config = AppConfig(
        server=ServerConfig(**server_dict),
        audio=AudioConfig(**audio_dict),
        asr=ASRConfig(**asr_dict),
        diarization=DiarizationConfig(**diarization_dict)
    )
    
    return config


def setup_logging(log_level: str):
    """
    Configure the logging system.
    
    Args:
        log_level: Logging level
        
    Returns:
        logging.Logger: Main application logger
    """
    level = getattr(logging, log_level)
    
    # Global configuration
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    
    # Configure important loggers explicitly
    api_logger = logging.getLogger("api")
    api_logger.setLevel(level)
    
    websocket_logger = logging.getLogger("api.websocket")
    websocket_logger.setLevel(level)
    
    # Reduce noise from third-party libraries
    for logger_name in ["uvicorn", "asyncio", "diart"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Main application logger
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(level)
    
    return app_logger