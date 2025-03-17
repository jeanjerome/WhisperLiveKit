#!/usr/bin/env python3
# services/transcription.py - Audio transcription service for WhisperLiveKit

import asyncio
import logging
import sys

from whisper_streaming_custom.online_asr import OnlineASRProcessor
from whisper_streaming_custom.whisper_online import backend_factory, online_factory, warmup_asr

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Service responsible for audio transcription.
    Encapsulates the functionality of the whisper_streaming_custom module
    within a modular architecture.
    """
    
    def __init__(self, config=None):
        """
        Initialize the transcription service.
        
        Args:
            config: Configuration for the ASR service (optional)
        """
        self.config = config or {}
        self.asr = None
        self.tokenizer = None
        self.initialized = False
        self.logfile = self.config.get("logfile", sys.stderr)
        
    async def initialize(self):
        """
        Initialize the ASR backend asynchronously.
        """
        if self.initialized:
            return
            
        try:
            # Convert configuration to an args-like object for backend_factory
            class ConfigObject:
                pass
                
            args = ConfigObject()
            for key, value in self.config.items():
                setattr(args, key, value)
            
            # Ensure required attributes are present
            required_attrs = ["model", "model_cache_dir", "model_dir", "lan", "task", 
                             "vad", "buffer_trimming", "buffer_trimming_sec", "backend"]
            for attr in required_attrs:
                if not hasattr(args, attr):
                    setattr(args, attr, self.config.get(attr, None))
            
            # Initialize the ASR backend
            self.asr, self.tokenizer = await asyncio.to_thread(backend_factory, args)
            self.initialized = True
            logger.info(f"ASR backend initialized with model {args.model}")
        except Exception as e:
            logger.error(f"Failed to initialize ASR backend: {e}", exc_info=True)
            raise
        
    def create_processor(self) -> OnlineASRProcessor:
        """
        Create an online ASR processor for a user session.
        
        Returns:
            OnlineASRProcessor: Configured online ASR processor
        """
        if not self.initialized:
            raise RuntimeError("TranscriptionService must be initialized before creating processors")
            
        # Convert configuration to an args-like object for online_factory
        class ConfigObject:
            pass
            
        args = ConfigObject()
        for key, value in self.config.items():
            setattr(args, key, value)
        
        # Ensure required attributes are present
        required_attrs = ["min_chunk_size", "vac", "vac_chunk_size", "confidence_validation"]
        for attr in required_attrs:
            if not hasattr(args, attr):
                setattr(args, attr, self.config.get(attr, None))
        
        return online_factory(
            args,
            self.asr, 
            self.tokenizer,
            logfile=self.logfile
        )
        
    async def warmup(self, warmup_file=None):
        """
        Warm up the ASR model for better initial performance.
        
        Args:
            warmup_file: Path to an audio file for warm-up (optional)
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            await asyncio.to_thread(
                warmup_asr, self.asr, warmup_file
            )
            logger.info("ASR model warmed up successfully")
        except Exception as e:
            logger.warning(f"Error warming up ASR model: {e}")