#!/usr/bin/env python3
# processors/audio/interface.py - Interface for audio processors

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

import numpy as np


class AudioProcessorInterface(ABC):
    """
    Interface for audio processors.
    Defines the common methods that all audio processors must implement.
    """
    
    @property
    @abstractmethod
    def process(self) -> Any:
        """
        Get the underlying process or stream object.
        This is mainly for compatibility with the websocket handler.
        
        Returns:
            Any: The process or a proxy object that provides compatible behavior
        """
        pass
    
    @property
    @abstractmethod
    def pcm_buffer(self) -> bytearray:
        """
        Get the PCM buffer.
        
        Returns:
            bytearray: The PCM buffer
        """
        pass
    
    @pcm_buffer.setter
    @abstractmethod
    def pcm_buffer(self, value: bytearray) -> None:
        """
        Set the PCM buffer.
        
        Args:
            value: The new PCM buffer
        """
        pass
    
    @abstractmethod
    async def start_decoder(self) -> Any:
        """
        Start the audio decoder/processor.
        
        Returns:
            Any: The processor-specific stream or process object
        """
        pass
        
    @abstractmethod
    async def stop_decoder(self) -> None:
        """
        Stop the audio decoder/processor and clean up resources.
        """
        pass
        
    @abstractmethod
    async def process_audio_chunk(self, chunk: bytes, bytes_per_sec: int, max_bytes: int) -> Tuple[np.ndarray, bool]:
        """
        Process an audio chunk and return a PCM array.
        
        Args:
            chunk: Audio data in bytes
            bytes_per_sec: Number of bytes per second (for sample rate)
            max_bytes: Maximum number of bytes to process
            
        Returns:
            Tuple (PCM array, reset indicator)
        """
        pass
        
    @abstractmethod
    def get_buffer_size(self) -> int:
        """
        Get the current size of the PCM buffer.
        
        Returns:
            Buffer size in bytes
        """
        pass