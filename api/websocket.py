#!/usr/bin/env python3
# api/websocket.py - WebSocket handling for real-time audio transcription

import asyncio
import logging
import math
import time
import traceback
from datetime import timedelta
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from core.state import SessionManager, SessionState
from services.audio import AudioService, FFmpegAudioProcessor
from services.diarization import DiarizationService
from services.transcription import TranscriptionService


logger = logging.getLogger("api.websocket")


def format_time(seconds):
    """Format seconds as a human-readable timestamp."""
    return str(timedelta(seconds=int(seconds)))


class WebSocketHandler:
    """
    Handler for real-time transcription/diarization WebSocket connections.
    """
    
    def __init__(
        self, 
        session_manager: SessionManager,
        audio_service: AudioService,
        transcription_service: Optional[TranscriptionService] = None,
        diarization_service: Optional[DiarizationService] = None,
        config=None
    ):
        """
        Initialize the WebSocket handler.
        
        Args:
            session_manager: User session manager
            transcription_service: Transcription service (optional)
            diarization_service: Diarization service (optional)
            config: Handler configuration
        """
        self.session_manager = session_manager
        self.audio_service = audio_service
        self.transcription_service = transcription_service
        self.diarization_service = diarization_service
        self.config = config or {}
        
        # Default parameters
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.channels = self.config.get("channels", 1)
        self.min_chunk_size = self.config.get("min_chunk_size", 1.0)
        
        # Audio buffer size calculations
        self.samples_per_sec = self.sample_rate * int(self.min_chunk_size)
        self.bytes_per_sample = 2  # s16le = 2 bytes per sample
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = self.sample_rate * 5 * self.bytes_per_sample  # 5 seconds of audio
        
    async def handle_connection(self, websocket: WebSocket):
        """
        Handle an incoming WebSocket connection for audio transcription/diarization.
        
        Args:
            websocket: WebSocket connection to handle
        """
        await websocket.accept()
        
        logger.info("WebSocket connection opened") 
        
        # Create or retrieve a session for this user
        session_id, session = await self.session_manager.create_session()
        
        # Audio processing variables
        ffmpeg_processor = self.audio_service.create_processor()
        
        # Ensure services are available
        logger.debug(f"Available services: transcription={self.transcription_service is not None}, diarization={self.diarization_service is not None}")
        
        # Queues for communication between tasks
        transcription_queue = asyncio.Queue() if self.transcription_service is not None else None
        diarization_queue = asyncio.Queue() if self.diarization_service is not None else None

        # Create processors for this session
        online = None
        diarization_processor = None
        
        # Initialize transcription processor if service is available
        if self.transcription_service:
            try:
                logger.debug("Creating transcription processor")
                online = self.transcription_service.create_processor()
            except Exception as e:
                logger.error(f"Error creating transcription processor: {e}", exc_info=True)
                online = None
            
        # Initialize diarization processor if service is available
        if self.diarization_service:
            try:
                logger.debug("Creating diarization processor")
                diarization_processor = self.diarization_service.create_processor()
            except Exception as e:
                logger.error(f"Error creating diarization processor: {e}", exc_info=True)
                diarization_processor = None
        
        # List of asynchronous tasks
        tasks = []
        
        # Start processing tasks
        if online is not None and transcription_queue is not None:
            logger.debug("Starting transcription worker")
            tasks.append(asyncio.create_task(
                self._transcription_worker(session, transcription_queue, online)
            ))
            
        if diarization_processor is not None and diarization_queue is not None:
            logger.debug("Starting diarization worker")
            tasks.append(asyncio.create_task(
                self._diarization_worker(session, diarization_queue, diarization_processor)
            ))
            
        # Task to format and send results
        formatter_task = asyncio.create_task(
            self._results_formatter(session, websocket)
        )
        tasks.append(formatter_task)
        
        # Add task to read FFmpeg output
        ffmpeg_reader_task = asyncio.create_task(
            self._ffmpeg_stdout_reader(ffmpeg_processor, transcription_queue, diarization_queue)
        )
        tasks.append(ffmpeg_reader_task)
        
        try:
            # Start the FFmpeg decoder
            ffmpeg_process = await ffmpeg_processor.start_decoder()
            
            # Variable to track sampling time
            last_process_time = time.time()
            
            while True:
                # Receive audio chunks from client
                message = await websocket.receive_bytes()
                
                logger.debug(f"Received audio chunk: {len(message)} bytes")
                
                try:
                    # Send data to FFmpeg
                    if ffmpeg_processor.process:
                        ffmpeg_processor.process.stdin.write(message)
                        ffmpeg_processor.process.stdin.flush()
                    else:
                        await ffmpeg_processor.start_decoder()
                        ffmpeg_processor.process.stdin.write(message)
                        ffmpeg_processor.process.stdin.flush()
                    
                    # Calculate elapsed time since last processing
                    current_time = time.time()
                    elapsed_time = current_time - last_process_time
                    read_size = max(int(self.sample_rate * elapsed_time * self.bytes_per_sample), 4096)
                    last_process_time = current_time
                    
                    # Read and process audio data
                    if ffmpeg_processor.get_buffer_size() >= self.bytes_per_sec:
                        # Process data without sending a new message
                        pcm_array, restarted = await ffmpeg_processor.process_audio_chunk(
                            b'',  # No need to send data as it's already in stdin
                            read_size,
                            self.max_bytes_per_sec
                        )
                        
                        # If we have audio data, send it to processors
                        if len(pcm_array) > 0:
                            logger.debug(f"Processing audio: {len(pcm_array)} samples")
                            
                            if transcription_queue:
                                await transcription_queue.put(pcm_array.copy())
                            
                            if diarization_queue:
                                await diarization_queue.put(pcm_array.copy())
                    
                except (BrokenPipeError, AttributeError) as e:
                    logger.warning(f"Error writing to FFmpeg: {e}. Restarting...")
                    await ffmpeg_processor.start_decoder()
                except Exception as e:
                    logger.error(f"Error processing audio: {e}", exc_info=True)
                    
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected.")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        finally:
            # Clean up resources
            for task in tasks:
                task.cancel()
                
            try:
                # Wait for all tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Close the FFmpeg decoder
                await ffmpeg_processor.stop_decoder()
                
                # Close the diarization processor
                if diarization_processor:
                    diarization_processor.close()
                    
                # Remove the session
                await self.session_manager.remove_session(session_id)
                
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
                
            logger.info("WebSocket endpoint cleaned up.")
            
    async def _ffmpeg_stdout_reader(
        self, 
        ffmpeg_processor: FFmpegAudioProcessor,
        transcription_queue: Optional[asyncio.Queue] = None,
        diarization_queue: Optional[asyncio.Queue] = None
    ):
        """
        Reads the output from the FFmpeg process and distributes audio data.
        """        
        loop = asyncio.get_event_loop()
        beg = time.time()
        
        while True:
            try:
                # Calculate elapsed time to adjust read size
                elapsed_time = math.floor((time.time() - beg) * 10) / 10  # Rounded to 0.1 sec
                read_size = max(int(self.sample_rate * elapsed_time * self.bytes_per_sample), 4096)
                beg = time.time()

                # Read directly from FFmpeg process
                try:
                    raw_audio = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: ffmpeg_processor.process.stdout.read(read_size)
                        ),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg read timeout. Restarting...")
                    await ffmpeg_processor.start_decoder()
                    continue
                    
                if not raw_audio:
                    logger.debug("No audio data received from FFmpeg")
                    await asyncio.sleep(0.1)
                    continue
                    
                # Add to buffer
                ffmpeg_processor.pcm_buffer.extend(raw_audio)
                
                # Process data if buffer is large enough
                if len(ffmpeg_processor.pcm_buffer) >= self.bytes_per_sec:
                    max_size = self.max_bytes_per_sec
                    if len(ffmpeg_processor.pcm_buffer) > max_size:
                        logger.warning(
                            f"Audio buffer is too large: {len(ffmpeg_processor.pcm_buffer) / self.bytes_per_sec:.2f} seconds."
                        )
                    
                    # Convert to float32
                    pcm_array = (
                        np.frombuffer(ffmpeg_processor.pcm_buffer[:max_size], dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    
                    # Keep the rest of the buffer
                    ffmpeg_processor.pcm_buffer = ffmpeg_processor.pcm_buffer[max_size:]
                    
                    logger.debug(f"Processed PCM data: {len(pcm_array)} samples")
                    
                    # Send to queues
                    if len(pcm_array) > 0:
                        if transcription_queue:
                            await transcription_queue.put(pcm_array.copy())
                            
                        if diarization_queue:
                            await diarization_queue.put(pcm_array.copy())
                
            except Exception as e:
                logger.error(f"Error in ffmpeg_stdout_reader: {e}", exc_info=True)
                await asyncio.sleep(0.5)
        
    async def _transcription_worker(self, session: SessionState, pcm_queue: asyncio.Queue, online):
        """
        Process audio data for transcription.
        
        Args:
            session: User session state
            pcm_queue: PCM audio data queue
            online: Online ASR processor
        """
        full_transcription = ""
        sep = online.asr.sep
        
        logger.debug("Transcription worker started")
        
        while True:
            try:
                # Get audio data from the queue
                pcm_array = await pcm_queue.get()
                
                logger.debug(f"Processing audio: {len(online.audio_buffer) / online.SAMPLING_RATE:.2f}s")
                
                # Process transcription
                online.insert_audio_chunk(pcm_array)
                new_tokens = online.process_iter()
                
                if new_tokens:
                    full_transcription += sep.join([t.text for t in new_tokens])
                    
                _buffer = online.get_buffer()
                buffer = _buffer.text if hasattr(_buffer, 'text') else _buffer
                end_buffer = _buffer.end if hasattr(_buffer, 'end') and _buffer.end else (new_tokens[-1].end if new_tokens else 0)
                
                if buffer in full_transcription:
                    buffer = ""
                    
                await session.update_transcription(
                    new_tokens, buffer, end_buffer, full_transcription, sep)
                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
            finally:
                pcm_queue.task_done()
                
    async def _diarization_worker(self, session: SessionState, pcm_queue: asyncio.Queue, diarization_obj):
        """
        Process audio data for diarization.
        
        Args:
            session: User session state
            pcm_queue: PCM audio data queue
            diarization_obj: Diarization processor
        """
        logger.debug("Diarization worker started")
        
        buffer_diarization = ""
        
        while True:
            try:
                # Get audio data from the queue
                pcm_array = await pcm_queue.get()
                
                # Process diarization
                await diarization_obj.diarize(pcm_array)
                
                # Get current state
                state = await session.get_current_state()
                tokens = state["tokens"]
                end_attributed_speaker = state["end_attributed_speaker"]
                
                # Update speaker information
                new_end_attributed_speaker = diarization_obj.assign_speakers_to_tokens(
                    end_attributed_speaker, tokens)
                
                await session.update_diarization(new_end_attributed_speaker, buffer_diarization)
                
            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
            finally:
                pcm_queue.task_done()
                
    async def _results_formatter(self, session: SessionState, websocket: WebSocket):
        """
        Format and send transcription/diarization results to the client.
        
        Args:
            session: User session state
            websocket: WebSocket connection for sending results
        """
        logger.debug("Results formatter started")
        
        while True:
            try:
                # Get current state
                state = await session.get_current_state()
                tokens = state["tokens"]
                buffer_transcription = state["buffer_transcription"]
                buffer_diarization = state["buffer_diarization"]
                end_attributed_speaker = state["end_attributed_speaker"]
                remaining_time_transcription = state["remaining_time_transcription"]
                remaining_time_diarization = state["remaining_time_diarization"]
                sep = state["sep"]
                
                # If diarization is enabled but not transcription, add dummy tokens
                if (not tokens or tokens[-1].is_dummy) and self.diarization_service and not self.transcription_service:
                    await session.add_dummy_token()
                    await asyncio.sleep(0.5)
                    state = await session.get_current_state()
                    tokens = state["tokens"]
                
                # Process tokens to create response
                previous_speaker = -1
                lines = []
                last_end_diarized = 0
                undiarized_text = []
                
                for token in tokens:
                    speaker = token.speaker
                    if self.diarization_service:
                        if (speaker == -1 or speaker == 0) and token.end >= end_attributed_speaker:
                            undiarized_text.append(token.text)
                            continue
                        elif (speaker == -1 or speaker == 0) and token.end < end_attributed_speaker:
                            speaker = previous_speaker
                        if speaker not in [-1, 0]:
                            last_end_diarized = max(token.end, last_end_diarized)

                    if speaker != previous_speaker or not lines:
                        lines.append(
                            {
                                "speaker": speaker,
                                "text": token.text,
                                "beg": format_time(token.start),
                                "end": format_time(token.end),
                                "diff": round(token.end - last_end_diarized, 2)
                            }
                        )
                        previous_speaker = speaker
                    elif token.text:  # Only add if text isn't empty
                        lines[-1]["text"] += sep + token.text
                        lines[-1]["end"] = format_time(token.end)
                        lines[-1]["diff"] = round(token.end - last_end_diarized, 2)
                
                if undiarized_text:
                    combined_buffer_diarization = sep.join(undiarized_text)
                    if buffer_transcription:
                        combined_buffer_diarization += sep
                    await session.update_diarization(end_attributed_speaker, combined_buffer_diarization)
                    buffer_diarization = combined_buffer_diarization
                    
                if lines:
                    response = {
                        "lines": lines, 
                        "buffer_transcription": buffer_transcription,
                        "buffer_diarization": buffer_diarization,
                        "remaining_time_transcription": remaining_time_transcription,
                        "remaining_time_diarization": remaining_time_diarization
                    }
                else:
                    response = {
                        "lines": [{
                            "speaker": 1,
                            "text": "",
                            "beg": format_time(0),
                            "end": format_time(tokens[-1].end) if tokens else format_time(0),
                            "diff": 0
                        }],
                        "buffer_transcription": buffer_transcription,
                        "buffer_diarization": buffer_diarization,
                        "remaining_time_transcription": remaining_time_transcription,
                        "remaining_time_diarization": remaining_time_diarization
                    }
                
                response_content = ' '.join([str(line['speaker']) + ' ' + line["text"] for line in lines]) + ' | ' + buffer_transcription + ' | ' + buffer_diarization
                
                if response_content != session.last_response_content:
                    if lines or buffer_transcription or buffer_diarization:
                        await websocket.send_json(response)
                        session.last_response_content = response_content
                
                # Add a small delay to avoid overwhelming the client
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)  # Longer pause on error