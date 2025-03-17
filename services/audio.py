#!/usr/bin/env python3
# services/audio.py - Audio processing service for WhisperLiveKit

import asyncio
import logging

from processors.audio import FFmpegAudioProcessor, PyAudioProcessor, AudioProcessorInterface

logger = logging.getLogger(__name__)


class AudioService:
    """
    Service responsible for audio processing operations.
    Manages audio capture, format conversion, and processing.
    """
    
    def __init__(self, config=None):
        """
        Initialize the audio service.
        
        Args:
            config: Configuration parameters (optional)
        """
        self.config = config or {}
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.channels = self.config.get("channels", 1)
        self.sample_width = self.config.get("sample_width", 2)  # 16-bit audio
        
        # Audio backend to use
        self.audio_backend = self.config.get("audio", "ffmpeg")
        
    async def initialize(self):
        """
        Initialize the audio service.
        Verify the appropriate audio backend is available.
        """
        if self.audio_backend == "pyaudio":
            try:
                # Try to import PyAudio
                import pyaudio
                from pydub import AudioSegment as PydubSegment
                
                # Check if PyAudio is working properly
                pa = pyaudio.PyAudio()
                device_count = pa.get_device_count()
                default_device = pa.get_default_input_device_info()
                pa.terminate()
                
                logger.info(f"PyAudio backend initialized. Found {device_count} audio devices.")
                logger.info(f"Default input device: {default_device['name']}")
                return True
            except ImportError as e:
                logger.error(f"PyAudio or pydub not installed: {e}")
                logger.error("To use PyAudio backend, install required packages: pip install pyaudio pydub")
                return False
            except Exception as e:
                logger.error(f"Error initializing PyAudio: {e}")
                return False
        else:  # "ffmpeg"
            try:
                # Try to import ffmpeg
                import ffmpeg
                
                # Check if FFmpeg is available
                version = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-version", 
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await version.communicate()
                
                if version.returncode != 0:
                    logger.warning(f"FFmpeg might not be available: {stderr.decode()}")
                    return False
                else:
                    logger.info("FFmpeg backend initialized successfully")
                    return True
            except ImportError as e:
                logger.error(f"ffmpeg-python not installed: {e}")
                logger.error("To use FFmpeg backend, install required package: pip install ffmpeg-python")
                return False
            except Exception as e:
                logger.error(f"Error checking FFmpeg: {e}")
                return False
            
    def create_processor(self) -> AudioProcessorInterface:
        """
        Create an audio processor instance for a session.
        
        Returns:
            AudioProcessorInterface: Configured audio processor
        """
        if self.audio_backend == "pyaudio":
            logger.info("Creating PyAudio processor")
            return PyAudioProcessor(
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_width=self.sample_width
            )
        else:  # "ffmpeg"
            logger.info("Creating FFmpeg processor")
            return FFmpegAudioProcessor(
                sample_rate=self.sample_rate,
                channels=self.channels
            )