# server/tts.py - Multi-backend TTS engine
import os
import time
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

log = logging.getLogger("v2v.tts")

def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"

class TTSEngine:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.backend = cfg.get("backend", "edge")  # Default to Edge TTS
        self.model_name = cfg.get("model", "edge-tts")
        self.out_sr = int(cfg.get("out_sr", 24000))
        self.speed = float(cfg.get("default_speed", 1.0))
        self.device = cfg.get("device") or _dev()
        
        # Enhanced settings for long paragraphs
        self.max_chunk_length = cfg.get("max_chunk_length", 800)
        self.chunk_overlap = cfg.get("chunk_overlap", 50) 
        self.synthesis_timeout = cfg.get("synthesis_timeout", 45)
        self.max_total_timeout = cfg.get("max_total_timeout", 180)
        self.retry_attempts = cfg.get("retry_attempts", 2)
        self.pause_between_chunks = cfg.get("pause_between_chunks", 0.3)

        log.info(f"TTS: init start (backend={self.backend}, device={self.device})")
        log.info(f"TTS: Enhanced settings - max_chunk: {self.max_chunk_length}, timeout: {self.synthesis_timeout}s")
        
        # Debug the configuration
        log.info(f"TTS Config: backend={cfg.get('backend', 'edge')}, model={cfg.get('model', 'edge-tts')}")

        t0 = time.time()
        
        if self.backend == "edge":
            log.info("TTS: Initializing Edge TTS backend...")
            self._init_edge_tts()
        elif self.backend == "coqui":
            log.info("TTS: Initializing Coqui TTS backend...")
            self._init_coqui_tts()
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

        log.info(f"TTS: initialization complete in {time.time()-t0:.2f}s")

    def _split_text(self, text: str, max_length: Optional[int] = None) -> List[str]:
        """
        Split long text into smaller chunks for TTS processing.
        Tries to split at sentence boundaries, then phrases, then words.
        """
        max_length = max_length or self.max_chunk_length
        
        if len(text) <= max_length:
            return [text]

        log.info(f"Splitting text of {len(text)} chars into chunks (max: {max_length})")
        chunks = []
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If this sentence alone exceeds max_length, split it further
            if len(sentence) > max_length:
                # If we have a current chunk, save it
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split long sentence by phrases/clauses
                phrases = re.split(r'(?<=[,;:])\s+', sentence)
                for phrase in phrases:
                    phrase = phrase.strip()
                    if not phrase:
                        continue
                        
                    if len(phrase) > max_length:
                        # Split by words as last resort
                        words = phrase.split()
                        word_chunk = ""
                        for word in words:
                            test_chunk = (word_chunk + " " + word).strip()
                            if len(test_chunk) <= max_length:
                                word_chunk = test_chunk
                            else:
                                if word_chunk:
                                    chunks.append(word_chunk)
                                word_chunk = word
                        if word_chunk:
                            chunks.append(word_chunk)
                    else:
                        test_chunk = (current_chunk + " " + phrase).strip()
                        if len(test_chunk) <= max_length:
                            current_chunk = test_chunk
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = phrase
            else:
                # Normal sentence processing
                test_chunk = (current_chunk + " " + sentence).strip()
                if len(test_chunk) <= max_length:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        log.info(f"Split text into {len(chunks)} chunks (avg: {sum(len(c) for c in chunks)//len(chunks) if chunks else 0} chars/chunk)")
        return chunks

    def _init_edge_tts(self):
        """Initialize Edge TTS backend."""
        try:
            import edge_tts
            self.edge_tts = edge_tts
            
            # Voice mapping for different languages
            self.voice_map = {
                "en": "en-US-AriaNeural",
                "ar": "ar-SA-ZariyahNeural", 
                "ne": "ne-NP-SagarNeural",
                "hi": "hi-IN-SwaraNeural",
                "es": "es-ES-ElviraNeural",
                "fr": "fr-FR-DeniseNeural",
                "de": "de-DE-KatjaNeural",
                "zh": "zh-CN-XiaoxiaoNeural",
                "ja": "ja-JP-NanamiNeural",
                "ko": "ko-KR-SunHiNeural",
            }
            
            log.info("Edge TTS backend initialized successfully")
            
            # Warmup test
            import asyncio
            try:
                asyncio.run(self._edge_warmup())
                log.info("Edge TTS warmup successful")
            except Exception as e:
                log.warning(f"Edge TTS warmup failed (but continuing): {e}")
                
        except ImportError:
            log.error("edge-tts not installed. Install with: pip install edge-tts")
            raise
        except Exception as e:
            log.exception("Edge TTS initialization failed")
            raise

    async def _edge_warmup(self):
        """Warmup Edge TTS with a simple synthesis."""
        communicate = self.edge_tts.Communicate("hello", "en-US-AriaNeural")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                break

    def _init_coqui_tts(self):
        """Initialize Coqui TTS backend (XTTS)."""
        try:
            from TTS.api import TTS
            
            # Disable numba JIT during init to avoid long compile pauses
            os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

            t1 = time.time()
            log.debug("TTS: constructing Coqui TTS()")
            self.tts = TTS(self.model_name)
            log.debug(f"TTS: constructed in {time.time()-t1:.2f}s")

            t2 = time.time()
            log.debug("TTS: moving model to device")
            self.tts = self.tts.to(self.device)
            log.debug(f"TTS: moved to {self.device} in {time.time()-t2:.2f}s")

            # Warmup synthesis
            t3 = time.time()
            log.info("TTS: warmup synthesis...")
            _ = self._coqui_synth("hello", "en", speaker_wav=None)
            log.info(f"TTS: WARMUP OK in {time.time()-t3:.2f}s")
            
        except Exception as e:
            log.exception("Coqui TTS initialization failed")
            raise

    def synth(self, text: str, lang: str, speaker_wav: Optional[str] = None, 
              speed: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using the configured backend.
        Handles long texts by splitting them into chunks with retry logic.
        
        Returns:
            Tuple[np.ndarray, int]: (audio_array, sample_rate)
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32), self.out_sr

        try:
            log.info(f"TTS: synthesizing '{text[:100]}...' ({len(text)} chars) using {self.backend} backend for lang={lang}")
            
            # Split long texts into chunks  
            text_chunks = self._split_text(text)
            
            if len(text_chunks) == 1:
                # Single chunk, process with retry
                return self._synthesize_with_retry(text, lang, speaker_wav, speed)
            else:
                # Multiple chunks, synthesize each and concatenate
                log.info(f"Processing {len(text_chunks)} text chunks for TTS")
                all_audio = []
                total_start_time = time.time()
                
                for i, chunk in enumerate(text_chunks):
                    chunk_start_time = time.time()
                    
                    # Check total timeout
                    elapsed_total = time.time() - total_start_time
                    if elapsed_total > self.max_total_timeout:
                        log.error(f"TTS total timeout exceeded ({elapsed_total:.1f}s > {self.max_total_timeout}s)")
                        break
                    
                    try:
                        log.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}: '{chunk[:50]}...' ({len(chunk)} chars)")
                        chunk_audio, sr = self._synthesize_with_retry(chunk, lang, speaker_wav, speed)
                        
                        if len(chunk_audio) > 0:
                            all_audio.append(chunk_audio)
                            
                        # Add pause between chunks (except for the last one)
                        if i < len(text_chunks) - 1:
                            pause_samples = int(self.pause_between_chunks * sr)
                            silence = np.zeros(pause_samples, dtype=np.float32)
                            all_audio.append(silence)
                            
                        chunk_time = time.time() - chunk_start_time
                        log.debug(f"Chunk {i+1} completed in {chunk_time:.1f}s")
                        
                    except Exception as e:
                        log.warning(f"Failed to synthesize chunk {i+1}: {e}")
                        # Add a brief silence for failed chunks to maintain timing
                        silence = np.zeros(int(0.5 * self.out_sr), dtype=np.float32)
                        all_audio.append(silence)
                        continue
                
                if not all_audio:
                    log.error("All TTS chunks failed")
                    return np.array([], dtype=np.float32), self.out_sr
                
                # Concatenate all audio chunks
                final_audio = np.concatenate(all_audio)
                total_time = time.time() - total_start_time
                log.info(f"TTS completed: {len(final_audio)} samples, {len(final_audio)/self.out_sr:.1f}s audio, {total_time:.1f}s processing")
                
                return final_audio, self.out_sr
                
        except Exception as e:
            log.exception(f"TTS synthesis failed for text: '{text[:50]}...' with backend: {self.backend}")
            # Return silence instead of crashing
            return np.array([], dtype=np.float32), self.out_sr

    def _synthesize_with_retry(self, text: str, lang: str, speaker_wav: Optional[str] = None, 
                              speed: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Synthesize a single chunk of text with minimal retry logic for fast failure.
        """
        last_error = None
        
        # For problematic languages, skip retries entirely to prevent hanging
        max_attempts = 1 if lang in ['ne', 'hi'] else (self.retry_attempts + 1)
        
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    log.info(f"TTS retry attempt {attempt}/{max_attempts-1} for {lang} chunk")
                    time.sleep(0.5 * attempt)  # Shorter backoff
                    
                return self._synthesize_single(text, lang, speaker_wav, speed)
                
            except Exception as e:
                last_error = e
                log.warning(f"TTS attempt {attempt + 1} failed for {lang}: {e}")
                
                # Don't retry on timeout errors at all
                if "timeout" in str(e).lower():
                    log.error(f"TTS timeout for {lang} - failing immediately")
                    break
                    
                # Don't retry for Indic languages that tend to hang
                if lang in ['ne', 'hi']:
                    log.error(f"TTS failed for {lang} - no retry for Indic languages")
                    break
                    
                if attempt >= max_attempts - 1:
                    break
        
        # All attempts failed
        log.error(f"TTS failed after {max_attempts} attempts for {lang}: {last_error}")
        raise last_error or Exception(f"TTS synthesis failed for {lang}")

    def _synthesize_single(self, text: str, lang: str, speaker_wav: Optional[str] = None, 
                          speed: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Synthesize a single chunk of text (internal method).
        """
        if self.backend == "edge":
            return self._edge_synth(text, lang, speed)
        elif self.backend == "coqui":
            return self._coqui_synth(text, lang, speaker_wav, speed)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _edge_synth(self, text: str, lang: str, speed: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Synthesize using Edge TTS with aggressive timeout handling for stability."""
        import asyncio
        
        voice = self.voice_map.get(lang, "en-US-AriaNeural")
        
        # Adjust speed if specified
        if speed and speed != 1.0:
            # Edge TTS speed format: +20% or -20%
            speed_percent = int((speed - 1.0) * 100)
            if speed_percent > 0:
                rate = f"+{speed_percent}%"
            else:
                rate = f"{speed_percent}%"
            
            # Use SSML for speed control
            text = f'<prosody rate="{rate}">{text}</prosody>'
        
        # Use more aggressive timeout for problematic texts
        timeout = self.synthesis_timeout
        
        # Even shorter timeout for long Nepali/Hindi texts that tend to hang
        if lang in ['ne', 'hi'] and len(text) > 100:
            timeout = min(timeout, 10.0)  # Maximum 10 seconds for long Indic texts
            log.warning(f"Using aggressive timeout {timeout}s for {lang} text ({len(text)} chars)")
        
        try:
            log.debug(f"Edge TTS synthesizing with timeout {timeout}s: '{text[:50]}...'")
            return asyncio.run(asyncio.wait_for(
                self._edge_synth_async(text, voice),
                timeout=timeout
            ))
        except asyncio.TimeoutError:
            log.error(f"Edge TTS timeout after {timeout}s for {lang} text: '{text[:50]}...' ({len(text)} chars)")
            # Return silence immediately instead of hanging
            return np.array([], dtype=np.float32), self.out_sr
        except Exception as e:
            log.error(f"Edge TTS error for {lang}: {e}")
            return np.array([], dtype=np.float32), self.out_sr

    async def _edge_synth_async(self, text: str, voice: str) -> Tuple[np.ndarray, int]:
        """Async Edge TTS synthesis."""
        communicate = self.edge_tts.Communicate(text, voice)
        audio_data = bytearray()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        
        if not audio_data:
            return np.array([], dtype=np.float32), self.out_sr
        
        # Convert MP3 bytes to audio array
        import io
        import soundfile as sf
        
        try:
            # Edge TTS returns MP3, convert to numpy array
            with io.BytesIO(audio_data) as audio_buffer:
                audio, sr = sf.read(audio_buffer)
                
                # Resample if needed
                if sr != self.out_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * self.out_sr / sr))
                
                return audio.astype(np.float32), self.out_sr
                
        except Exception as e:
            log.error(f"Failed to decode Edge TTS audio: {e}")
            # Return silence instead of crashing
            return np.array([], dtype=np.float32), self.out_sr

    def _coqui_synth(self, text: str, lang: str, speaker_wav: Optional[str] = None, 
                     speed: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Synthesize using Coqui TTS (XTTS) with timeout handling."""
        sp = speed or self.speed
        
        # XTTS synthesis with proper speaker handling
        tts_kwargs = {
            "text": text,
            "language": lang,
            "speed": sp,
        }
        
        # Add speaker parameter if provided, or use a default speaker ID
        if speaker_wav:
            tts_kwargs["speaker_wav"] = speaker_wav
        else:
            # For multi-speaker models, we need to provide either speaker_wav or speaker_id
            # Use a default speaker ID or the first available speaker
            if hasattr(self.tts, 'speakers') and self.tts.speakers:
                tts_kwargs["speaker"] = self.tts.speakers[0]
            elif hasattr(self.tts, 'speaker_manager') and self.tts.speaker_manager:
                speakers = list(self.tts.speaker_manager.speaker_names)
                if speakers:
                    tts_kwargs["speaker"] = speakers[0]
            # If no speakers available, try without speaker parameter
        
        # Use threading timeout for Coqui TTS
        import threading
        import queue
        
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def synthesis_worker():
            try:
                wav = self.tts.tts(**tts_kwargs)
                result_queue.put(wav)
            except ValueError as e:
                if "no `speaker` is provided" in str(e):
                    log.warning(f"Multi-speaker model needs speaker, trying with default...")
                    # Try with a generic speaker name or ID
                    tts_kwargs["speaker"] = "default"
                    try:
                        wav = self.tts.tts(**tts_kwargs)
                        result_queue.put(wav)
                    except:
                        # Last resort: try with speaker_id=0
                        tts_kwargs.pop("speaker", None)
                        tts_kwargs["speaker_id"] = 0
                        wav = self.tts.tts(**tts_kwargs)
                        result_queue.put(wav)
                else:
                    error_queue.put(e)
            except Exception as e:
                error_queue.put(e)
        
        # Start synthesis in a separate thread
        worker_thread = threading.Thread(target=synthesis_worker)
        worker_thread.daemon = True
        worker_thread.start()
        
        # Wait for result with configurable timeout
        worker_thread.join(timeout=self.synthesis_timeout)
        
        if worker_thread.is_alive():
            log.error(f"Coqui TTS timeout after {self.synthesis_timeout}s for text: '{text[:50]}...'")
            return np.array([], dtype=np.float32), self.out_sr
        
        # Check for errors
        if not error_queue.empty():
            error = error_queue.get()
            log.error(f"Coqui TTS error: {error}")
            return np.array([], dtype=np.float32), self.out_sr
        
        # Get result
        if result_queue.empty():
            log.error("Coqui TTS returned no result")
            return np.array([], dtype=np.float32), self.out_sr
        
        wav = result_queue.get()
        
        # Ensure numpy array
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        
        return wav.astype(np.float32), self.out_sr

    def synthesize(self, text: str, lang: str, speaker_wav: Optional[str], 
                   speed: Optional[float] = None, sample_rate: Optional[int] = None) -> bytes:
        """
        Legacy method for backward compatibility - returns WAV bytes.
        """
        wav, sr = self.synth(text, lang, speaker_wav, speed)
        
        # Convert to WAV bytes
        import io
        import soundfile as sf
        
        buf = io.BytesIO()
        sf.write(buf, wav, sample_rate or sr, subtype="PCM_16", format="WAV")
        return buf.getvalue()