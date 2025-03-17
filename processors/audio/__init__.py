"""
Audio processors package.
Contains processors for different audio backends.
"""

from .interface import AudioProcessorInterface
from .ffmpeg_processor import FFmpegAudioProcessor
from .pyaudio_processor import PyAudioProcessor

__all__ = ['AudioProcessorInterface', 'FFmpegAudioProcessor', 'PyAudioProcessor']