#!/usr/bin/env python3
# processors/audio/pyaudio_processor.py - PyAudio-based audio processor

import logging
import traceback
from typing import Tuple, Any
import numpy as np
from queue import Queue
from threading import Thread

from .interface import AudioProcessorInterface

logger = logging.getLogger(__name__)


class PyAudioStreamStdoutProxy:
    """
    A proxy object that mimics the stdout attribute of a subprocess.
    This allows us to make PyAudio compatible with the WebSocketHandler
    which expects a process.stdout interface.
    """
    def __init__(self, data_queue: Queue):
        self.data_queue = data_queue
        
    def read(self, size: int) -> bytes:
        """
        Read data from the queue.
        
        Args:
            size: Number of bytes to read
            
        Returns:
            bytes: The data read, or empty bytes if no data is available
        """
        try:
            if self.data_queue.empty():
                return b''
                
            data = bytearray()
            while len(data) < size and not self.data_queue.empty():
                chunk = self.data_queue.get_nowait()
                data.extend(chunk)
                
            return bytes(data)
        except Exception as e:
            logger.error(f"Error reading from PyAudio queue: {e}")
            return b''


class PyAudioStdinProxy:
    """
    A proxy object that mimics the stdin attribute of a subprocess.
    This is used for compatibility with the WebSocketHandler.
    In PyAudio we don't need to write to stdin, so this is mostly a no-op.
    """
    def write(self, data: bytes) -> int:
        """
        Write data (does nothing in PyAudio case).
        
        Args:
            data: Data to write
            
        Returns:
            int: Number of bytes written (always returns len(data))
        """
        return len(data)
        
    def flush(self) -> None:
        """
        Flush data (does nothing in PyAudio case).
        """
        pass


class PyAudioProcessProxy:
    """
    A proxy object that mimics a subprocess.Popen object.
    This allows us to make PyAudio compatible with the WebSocketHandler.
    """
    def __init__(self, data_queue: Queue):
        self.stdout = PyAudioStreamStdoutProxy(data_queue)
        self.stdin = PyAudioStdinProxy()


class PyAudioProcessor(AudioProcessorInterface):
    """
    Audio processor using PyAudio for direct microphone access.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2):
        """
        Initialize the PyAudio processor.
        
        Args:
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels
            sample_width: Sample width in bytes (2 = 16-bit)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.pa = None
        self.stream = None
        self.device_index = None
        self.chunk_size = 1024  # Default chunk size
        
        # Thread control variables
        self._pcm_buffer = bytearray()
        self.data_queue = Queue()
        self.run_thread = False
        self.thread = None
        
        # Process proxy for compatibility with WebSocket handler
        self._process = PyAudioProcessProxy(self.data_queue)
        
        # WebM data handling (for compatibility with existing code)
        self.webm_buffer = bytearray()
    
    @property
    def process(self) -> Any:
        """Get the process proxy."""
        return self._process
    
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
        Start the PyAudio processor to capture audio.
        
        Returns:
            The process proxy
        """
        # Import here to ensure errors only happen when actually used
        import pyaudio
        
        if self.pa is not None:
            await self.stop_decoder()
            
        try:
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            
            # Get default microphone or device
            if self.device_index is None:
                self.device_index = self._get_default_microphone()
                
            # Get format
            py_format = self._get_pyaudio_format(self.sample_width, False)
            
            # Open PyAudio stream
            self.stream = self.pa.open(
                format=py_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_index
            )
            
            # Start processing thread
            self.run_thread = True
            self.thread = Thread(target=self._processing_thread)
            self.thread.daemon = True
            self.thread.start()
            
            # Reset buffer
            self._pcm_buffer = bytearray()
            
            logger.debug("PyAudio processor started")
            return self.process
            
        except Exception as e:
            logger.error(f"Error starting PyAudio: {e}")
            traceback.print_exc()
            if self.pa:
                self.pa.terminate()
                self.pa = None
            return None
        
    async def stop_decoder(self) -> None:
        """
        Stop the PyAudio processor if it's running.
        """
        if self.thread and self.thread.is_alive():
            self.run_thread = False
            self.thread.join(timeout=1.0)
            self.thread = None
            
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                logger.warning(f"Error stopping PyAudio stream: {e}")
                
        if self.pa:
            try:
                self.pa.terminate()
                self.pa = None
            except Exception as e:
                logger.warning(f"Error terminating PyAudio: {e}")
                
        # Clear data queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                break
                
        logger.debug("PyAudio processor stopped")
        
    async def process_audio_chunk(self, chunk: bytes, bytes_per_sec: int, max_bytes: int) -> Tuple[np.ndarray, bool]:
        """
        Process an audio chunk and return a PCM array.
        For WebM compatibility, the chunk is stored but not used directly.
        The actual audio comes from the PyAudio stream.
        
        Args:
            chunk: WebM audio data (not used directly, maintained for API compatibility)
            bytes_per_sec: Number of bytes per second (for sample rate)
            max_bytes: Maximum number of bytes to process
            
        Returns:
            Tuple (PCM array, reset indicator)
        """
        if not self.pa or not self.stream:
            await self.start_decoder()
            
        try:
            # Save WebM data for compatibility, but don't process it directly
            if chunk:
                self.webm_buffer.extend(chunk)
                
            # Get data from the processing thread via queue
            pcm_data = bytearray()
            while not self.data_queue.empty() and len(pcm_data) < max_bytes:
                try:
                    data = self.data_queue.get_nowait()
                    pcm_data.extend(data)
                except:
                    break
                    
            # Add to buffer
            self._pcm_buffer.extend(pcm_data)
            
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
            logger.warning(f"Error in PyAudio processing: {e}")
            try:
                await self.start_decoder()
            except:
                logger.error("Failed to restart PyAudio after error")
            return np.array([], dtype=np.float32), True
            
    def get_buffer_size(self) -> int:
        """
        Get the current size of the PCM buffer.
        
        Returns:
            Buffer size in bytes
        """
        return len(self._pcm_buffer)
        
    def _get_default_microphone(self) -> int:
        """
        Detect and return the default microphone index.
        
        Returns:
            int: The default input device index
        """
        try:
            default_mic = self.pa.get_default_input_device_info()['index']
            return default_mic
        except:
            # Fallback if no default device found
            for i in range(self.pa.get_device_count()):
                device_info = self.pa.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    return device_info['index']
            return 0  # Last resort
            
    def _get_pyaudio_format(self, sample_width: int, is_float: bool) -> int:
        """
        Convert format parameters to Pyaudio constants.
        
        Args:
            sample_width: Sample width in bytes
            is_float: If the format is floating point
            
        Returns:
            int: Pyaudio format constant
        """
        # Import here to ensure errors only happen when actually used
        import pyaudio
        
        if is_float:
            return pyaudio.paFloat32
        elif sample_width == 1:
            return pyaudio.paInt8
        elif sample_width == 2:
            return pyaudio.paInt16
        elif sample_width == 4:
            return pyaudio.paInt32
        else:
            return pyaudio.paInt16  # Default format
            
    def _processing_thread(self) -> None:
        """Processing thread that captures data and puts it in the queue."""
        try:
            while self.run_thread and self.stream:
                try:
                    # Record an audio chunk
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Put data in the queue
                    self.data_queue.put(data)
                        
                except Exception as e:
                    logger.error(f"Error reading stream: {e}")
                    traceback.print_exc()
                    # Short pause to avoid CPU overload in case of error
                    try:
                        import time
                        time.sleep(0.1)
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
            traceback.print_exc()
        finally:
            # Properly close the stream in case of error
            if self.stream and self.run_thread:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                except:
                    pass