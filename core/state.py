#!/usr/bin/env python3
# core/state.py - Session state management for WhisperLiveKit

import asyncio
import logging
import uuid
from time import time
from typing import Any, Dict, List, Optional, Tuple

from core.timed_objects import ASRToken

logger = logging.getLogger(__name__)


class SessionState:
    """
    Manages the state of a transcription/diarization session for a user.
    Encapsulates the original SharedState class in a modular architecture.
    """
    
    def __init__(self):
        """Initialize the session state."""
        self.tokens: List[ASRToken] = []
        self.buffer_transcription: str = ""
        self.buffer_diarization: str = ""
        self.full_transcription: str = ""
        self.end_buffer: float = 0
        self.end_attributed_speaker: float = 0
        self.lock = asyncio.Lock()
        self.beg_loop = time()
        self.sep: str = " "  # Default separator
        self.last_response_content: str = ""  # To track response changes
        
    async def update_transcription(self, new_tokens, buffer, end_buffer, full_transcription, sep):
        """
        Update the transcription state with new data.
        
        Args:
            new_tokens: New ASR tokens
            buffer: Text currently in buffer
            end_buffer: Timestamp of buffer end
            full_transcription: Complete transcription
            sep: Token separator
        """
        async with self.lock:
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.full_transcription = full_transcription
            self.sep = sep
            
    async def update_diarization(self, end_attributed_speaker, buffer_diarization=""):
        """
        Update the diarization state with new data.
        
        Args:
            end_attributed_speaker: Timestamp until which speakers have been attributed
            buffer_diarization: Pending diarization text (optional)
        """
        async with self.lock:
            self.end_attributed_speaker = end_attributed_speaker
            if buffer_diarization:
                self.buffer_diarization = buffer_diarization
            
    async def add_dummy_token(self):
        """
        Add a dummy token to maintain display when there is no transcription.
        """
        async with self.lock:
            current_time = time() - self.beg_loop
            dummy_token = ASRToken(
                start=current_time,
                end=current_time + 1,
                text=".",
                speaker=-1,
                is_dummy=True
            )
            self.tokens.append(dummy_token)
            
    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state with remaining time calculations.
        
        Returns:
            Dict: Dictionary containing the current session state
        """
        async with self.lock:
            current_time = time()
            remaining_time_transcription = 0
            remaining_time_diarization = 0
            
            # Calculate remaining time for transcription buffer
            if self.end_buffer > 0:
                remaining_time_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 2))
                
            # Calculate remaining time for diarization
            remaining_time_diarization = max(0, round(
                max(self.end_buffer, self.tokens[-1].end if self.tokens else 0) - self.end_attributed_speaker, 2))
                
            return {
                "tokens": self.tokens.copy(),
                "buffer_transcription": self.buffer_transcription,
                "buffer_diarization": self.buffer_diarization,
                "end_buffer": self.end_buffer,
                "end_attributed_speaker": self.end_attributed_speaker,
                "sep": self.sep,
                "remaining_time_transcription": remaining_time_transcription,
                "remaining_time_diarization": remaining_time_diarization
            }
            
    async def reset(self):
        """Reset the session state."""
        async with self.lock:
            self.tokens = []
            self.buffer_transcription = ""
            self.buffer_diarization = ""
            self.end_buffer = 0
            self.end_attributed_speaker = 0
            self.full_transcription = ""
            self.beg_loop = time()
            self.last_response_content = ""


class SessionManager:
    """
    Manages multiple user sessions.
    Allows creating, retrieving, and deleting sessions.
    """
    
    def __init__(self):
        """Initialize the session manager."""
        self.sessions: Dict[str, SessionState] = {}
        self.lock = asyncio.Lock()
        
    async def create_session(self) -> Tuple[str, SessionState]:
        """
        Create a new session with a unique identifier.
        
        Returns:
            Tuple[str, SessionState]: (session_id, session)
        """
        session_id = str(uuid.uuid4())
        
        async with self.lock:
            self.sessions[session_id] = SessionState()
            
        return session_id, self.sessions[session_id]
        
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Retrieve an existing session by its identifier.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Optional[SessionState]: The requested session or None if it doesn't exist
        """
        async with self.lock:
            return self.sessions.get(session_id)
            
    async def get_or_create_session(self, session_id: str = None) -> Tuple[str, SessionState]:
        """
        Retrieve an existing session or create a new one.
        
        Args:
            session_id: Session identifier (optional)
            
        Returns:
            Tuple[str, SessionState]: (session_id, session)
        """
        if session_id is not None:
            session = await self.get_session(session_id)
            if session:
                return session_id, session
                
        return await self.create_session()
            
    async def remove_session(self, session_id: str) -> bool:
        """
        Remove a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if the session was removed, False otherwise
        """
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
            
    async def get_all_sessions(self) -> Dict[str, SessionState]:
        """
        Get all active sessions.
        
        Returns:
            Dict[str, SessionState]: Dictionary of active sessions
        """
        async with self.lock:
            return self.sessions.copy()
            
    async def get_session_count(self) -> int:
        """
        Count the number of active sessions.
        
        Returns:
            int: Number of active sessions
        """
        async with self.lock:
            return len(self.sessions)