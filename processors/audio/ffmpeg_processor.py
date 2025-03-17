#!/usr/bin/env python3
# processors/audio/ffmpeg_processor.py - FFmpeg-based audio processor

import asyncio
import logging
from typing import Tuple, Any

import numpy as np

from .interface import AudioProcessorInterface

logger = logging.getLogger(__name__)


class FFmpegAudioProcessor(AudioProcessorInterface):
    """
    Audio processor using FFmpeg for decoding WebM streams to PCM.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the FFmpeg audio processor.
        
        Args:
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._process = None
        self._pcm_buffer = bytearray()
    
    @property
    def process(self) -> Any:
        """Get the FFmpeg process."""
        return self._process
    
    @process.setter
    def process(self, value: Any) -> None:
        """Set the FFmpeg process."""
        self._process = value
    
    @property
    def pcm_buffer(self) -> bytearray:
        """Get the PCM buffer."""
        return self._pcm_buffer
    
    @pcm_buffer.setter
    def pcm_buffer(self, value: bytearray) -> None:
        """Set the PCM buffer."""
        self._pcm_buffer = value
        
    async def start_decoder(self) -> Any:
        """
        Start an FFmpeg process to decode WebM streams to PCM.
        
        Returns:
            The FFmpeg process
        """
        # Import here to ensure error happens only when the processor is actually used
        import ffmpeg
        
        if self.process:
            await self.stop_decoder()
            
        self._process = (
            ffmpeg.input("pipe:0", format="webm")
            .output(
                "pipe:1",
                format="s16le",
                acodec="pcm_s16le",
                ac=self.channels,
                ar=str(self.sample_rate),
            )
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        
        self._pcm_buffer = bytearray()
        logger.debug("FFmpeg decoder started")
        return self.process
        
    async def stop_decoder(self) -> None:
        """
        Stop the FFmpeg decoder if it's running.
        """
        if self.process:
            try:
                self.process.stdin.close()
                await asyncio.get_event_loop().run_in_executor(None, self.process.wait)
            except Exception as e:
                logger.warning(f"Error stopping FFmpeg process: {e}")
            finally:
                self._process = None
                logger.debug("FFmpeg decoder stopped")
                
    async def process_audio_chunk(self, chunk: bytes, bytes_per_sec: int, max_bytes: int) -> Tuple[np.ndarray, bool]:
        """
        Process a WebM audio chunk and return a PCM array.
        
        Args:
            chunk: WebM audio data
            bytes_per_sec: Number of bytes per second (for sample rate)
            max_bytes: Maximum number of bytes to process
            
        Returns:
            Tuple (PCM array, reset indicator)
        """
        if not self.process:
            await self.start_decoder()
            
        try:
            # Send data to FFmpeg
            self.process.stdin.write(chunk)
            self.process.stdin.flush()
            
            # Read PCM output data
            try:
                # Read with timeout to avoid blocking
                stdout_data = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.process.stdout.read, bytes_per_sec
                    ),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("FFmpeg read timeout. Restarting...")
                await self.start_decoder()
                return np.array([], dtype=np.float32), True
                
            if not stdout_data:
                logger.info("FFmpeg stdout closed or no data.")
                return np.array([], dtype=np.float32), False
                
            # Add data to buffer
            self._pcm_buffer.extend(stdout_data)
            
            # Process buffer if enough data is available
            if len(self._pcm_buffer) >= bytes_per_sec:
                if len(self._pcm_buffer) > max_bytes:
                    logger.warning(
                        f"Audio buffer is too large: {len(self._pcm_buffer) / bytes_per_sec:.2f} seconds. "
                        f"The model probably struggles to keep up. Consider using a smaller model."
                    )
                
                # Convert int16 -> float32 (normalized)
                pcm_array = (
                    np.frombuffer(self._pcm_buffer[:max_bytes], dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                
                # Keep the rest for the next cycle
                self._pcm_buffer = self._pcm_buffer[max_bytes:]
                
                return pcm_array, False
            
            # Not enough data to process
            return np.array([], dtype=np.float32), False
            
        except Exception as e:
            logger.warning(f"Error in FFmpeg processing: {e}")
            await self.start_decoder()
            return np.array([], dtype=np.float32), True
            
    def get_buffer_size(self) -> int:
        """
        Get the current size of the PCM buffer.
        
        Returns:
            Buffer size in bytes
        """
        return len(self._pcm_buffer)