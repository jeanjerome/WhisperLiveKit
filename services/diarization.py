#!/usr/bin/env python3
# services/diarization.py - Speaker diarization service for WhisperLiveKit

import asyncio
import logging
import re
import threading
from typing import Any, List, Tuple

import numpy as np
from diart import SpeakerDiarization
from diart.inference import StreamingInference
from diart.sources import AudioSource, MicrophoneAudioSource
from pyannote.core import Annotation
from rx.core import Observer

from core.timed_objects import SpeakerSegment

logger = logging.getLogger(__name__)


def extract_number(s: str) -> int:
    """
    Extract a speaker number from an identifier string.
    
    Args:
        s: Speaker identifier string
        
    Returns:
        int: Extracted number or None if not found
    """
    m = re.search(r'\d+', s)
    return int(m.group()) if m else None


class DiarizationObserver(Observer):
    """Observer that logs all data emitted by the diarization pipeline and stores speaker segments."""
    
    def __init__(self):
        self.speaker_segments = []
        self.processed_time = 0
        self.segment_lock = threading.Lock()
    
    def on_next(self, value: Tuple[Annotation, Any]):
        annotation, audio = value
        
        logger.debug("--- New Diarization Result ---")
        
        duration = audio.extent.end - audio.extent.start
        logger.debug(f"Audio segment: {audio.extent.start:.2f}s - {audio.extent.end:.2f}s (duration: {duration:.2f}s)")
        
        with self.segment_lock:
            if audio.extent.end > self.processed_time:
                self.processed_time = audio.extent.end            
            if annotation and len(annotation._labels) > 0:
                logger.debug("Speaker segments:")
                for speaker, label in annotation._labels.items():
                    for start, end in zip(label.segments_boundaries_[:-1], label.segments_boundaries_[1:]):
                        logger.debug(f"  {speaker}: {start:.2f}s-{end:.2f}s")
                        self.speaker_segments.append(SpeakerSegment(
                            speaker=speaker,
                            start=start,
                            end=end
                        ))
            else:
                logger.debug("No speakers detected in this segment")
                
    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.speaker_segments.copy()
    
    def clear_old_segments(self, older_than: float = 30.0):
        """Clear segments older than the specified time."""
        with self.segment_lock:
            current_time = self.processed_time
            self.speaker_segments = [
                segment for segment in self.speaker_segments 
                if current_time - segment.end < older_than
            ]
    
    def on_error(self, error):
        """Handle an error in the stream."""
        logger.debug(f"Error in diarization stream: {error}")
        
    def on_completed(self):
        """Handle the completion of the stream."""
        logger.debug("Diarization stream completed")


class WebSocketAudioSource(AudioSource):
    """
    Custom AudioSource that blocks in read() until close() is called.
    Use push_audio() to inject PCM chunks.
    """
    def __init__(self, uri: str = "websocket", sample_rate: int = 16000):
        super().__init__(uri, sample_rate)
        self._closed = False
        self._close_event = threading.Event()

    def read(self):
        self._close_event.wait()

    def close(self):
        if not self._closed:
            self._closed = True
            self.stream.on_completed()
            self._close_event.set()

    def push_audio(self, chunk: np.ndarray):
        if not self._closed:
            new_audio = np.expand_dims(chunk, axis=0)
            logger.debug(f'New audio chunk shape: {new_audio.shape}')
            self.stream.on_next(new_audio)


class DiartDiarization:
    """
    Core diarization processor using Diart library.
    Handles the streaming inference and speaker assignment.
    """
    def __init__(self, sample_rate: int = 16000, use_microphone: bool = False):
        self.pipeline = SpeakerDiarization()        
        self.observer = DiarizationObserver()
        
        if use_microphone:
            self.source = MicrophoneAudioSource()
            self.custom_source = None
        else:
            self.custom_source = WebSocketAudioSource(uri="websocket_source", sample_rate=sample_rate)
            self.source = self.custom_source
            
        self.inference = StreamingInference(
            pipeline=self.pipeline,
            source=self.source,
            do_plot=False,
            show_progress=False,
        )
        self.inference.attach_observers(self.observer)
        asyncio.get_event_loop().run_in_executor(None, self.inference)

    async def diarize(self, pcm_array: np.ndarray):
        """
        Process audio data for diarization.
        Only used when working with WebSocketAudioSource.
        
        Args:
            pcm_array: PCM audio data as numpy array
            
        Returns:
            List of speaker segments
        """
        if self.custom_source:
            self.custom_source.push_audio(pcm_array)            
        self.observer.clear_old_segments()        
        return self.observer.get_segments()

    def close(self):
        """Close the audio source."""
        if self.custom_source:
            self.custom_source.close()

    def assign_speakers_to_tokens(self, end_attributed_speaker, tokens: list) -> float:
        """
        Assign speakers to tokens based on timing overlap with speaker segments.
        Uses the segments collected by the observer.
        
        Args:
            end_attributed_speaker: End time of the last attributed speaker
            tokens: List of tokens to assign speakers to
            
        Returns:
            Updated end time of the last attributed speaker
        """
        segments = self.observer.get_segments()
        
        for token in tokens:
            for segment in segments:
                if not (segment.end <= token.start or segment.start >= token.end):
                    token.speaker = extract_number(segment.speaker) + 1
                    end_attributed_speaker = max(token.end, end_attributed_speaker)
        return end_attributed_speaker


class DiarizationService:
    """
    Service that encapsulates existing diarization functionality
    within a modular architecture.
    """
    
    def __init__(self, config=None):
        """
        Initialize the diarization service.
        
        Args:
            config: Optional configuration for the service
                - sample_rate: Sample rate (default: 16000)
                - use_microphone: Use microphone (default: False)
        """
        self.config = config or {}
        self.initialized = False
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.use_microphone = self.config.get("use_microphone", False)
        
    async def initialize(self):
        """
        Initialize the diarization service.
        """
        if self.initialized:
            return
            
        try:
            # Verify diart is available
            import diart
            from pyannote.audio import Model
            
            # Log that we're ready
            logger.info("Diarization service initialized successfully")
            self.initialized = True
                
        except ImportError as e:
            logger.error(f"Failed to initialize diarization service: {str(e)}")
            raise RuntimeError(
                "Diarization service requires diart and pyannote.audio. "
                "Please install them with pip."
            ) from e
    
    def create_processor(self) -> DiartDiarization:
        """
        Create a diarization processor for a user session.
        
        Returns:
            A configured DiartDiarization instance
        """
        if not self.initialized:
            raise RuntimeError("DiarizationService must be initialized before creating processors")
        
        return DiartDiarization(
            sample_rate=self.sample_rate,
            use_microphone=self.use_microphone
        )
        
    # Additional utility methods we can add to the service
    
    def extract_number(self, s: str) -> int:
        """
        Extract a speaker number from an identifier string.
        Reimplementation of the existing utility function.
        
        Args:
            s: Speaker identifier string
            
        Returns:
            int: Extracted number or None if not found
        """
        return extract_number(s)