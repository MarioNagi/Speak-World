# server/async_models.py - Complete version with streaming support
import asyncio
import concurrent.futures
from functools import partial
import logging
import gc
import os
import time

log = logging.getLogger("v2v.async_models")
MT_TIMEOUT_DEFAULT = float(os.environ.get("V2V_MT_TIMEOUT", "60"))

# Global thread pool for CPU-bound ML tasks
ML_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=3,  # Increased for better parallel processing
    thread_name_prefix="ml_worker"
)

def cleanup_executor():
    """Cleanup the executor on shutdown."""
    ML_EXECUTOR.shutdown(wait=True)

class AsyncASR:
    """Async wrapper for ASR model."""
    
    def __init__(self, asr_model):
        self.model = asr_model
    
    async def transcribe(self, audio_16k, lang=None, timeout=15.0):
        """Async ASR transcription with timeout and error handling."""
        try:
            func = partial(self.model.transcribe, audio_16k, lang=lang)
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(ML_EXECUTOR, func),
                timeout=timeout
            )
            return result if result and result.strip() else ""
        except asyncio.TimeoutError:
            raise Exception(f"ASR transcription timed out after {timeout}s")
        except Exception as e:
            log.exception("ASR transcription failed")
            raise Exception(f"ASR error: {str(e)}")

class AsyncMT:
    def __init__(self, mt_model):
        self.model = mt_model

    async def translate(self, text, src, tgt, timeout: float = MT_TIMEOUT_DEFAULT):
        if not text or not text.strip():
            return ""
        try:
            func = partial(self.model.translate, text, src=src, tgt=tgt)
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(ML_EXECUTOR, func),
                timeout=timeout
            )
            return result if result and result.strip() else text
        except asyncio.TimeoutError:
            raise Exception(f"Translation timed out after {timeout}s")
        except Exception as e:
            log.exception("Translation failed")
            raise Exception(f"MT error: {str(e)}")

class AsyncTTS:
    def __init__(self, tts_model):
        self.model = tts_model

    async def synthesize(self, text, lang, speaker_wav=None, timeout=None):
        """
        Async TTS synthesis with smart timeout based on text length.
        """
        if not text or not text.strip():
            import numpy as np
            return np.array([], dtype=np.float32), 24000

        # Calculate dynamic timeout based on text length
        char_count = len(text)
        
        if timeout is None:
            if char_count > 500:
                timeout = 60.0  # 1 minute for very long texts
            elif char_count > 200:
                timeout = 30.0  # 30 seconds for medium texts
            elif char_count > 100:
                timeout = 15.0  # 15 seconds for short texts
            else:
                timeout = 10.0  # 10 seconds for very short texts

        log.info(f"TTS synthesizing {char_count} chars with {timeout}s timeout")

        try:
            func = partial(self.model.synth, text, lang, speaker_wav=speaker_wav)
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(ML_EXECUTOR, func),
                timeout=timeout
            )
            
            if result is None or (hasattr(result, '__len__') and len(result) == 0):
                log.warning(f"TTS returned empty result for {lang}")
                import numpy as np
                return np.array([], dtype=np.float32), 24000
            
            return result
            
        except asyncio.TimeoutError:
            log.error(f"TTS synthesis timed out after {timeout:.1f}s for {char_count} chars")
            # Return silence instead of failing completely
            import numpy as np
            return np.array([], dtype=np.float32), 24000
        except Exception as e:
            log.exception(f"TTS synthesis failed: {e}")
            import numpy as np
            return np.array([], dtype=np.float32), 24000

class AsyncModelManager:
    """Manager for all async model wrappers with streaming support."""
    
    def __init__(self, asr_model, mt_model, tts_model):
        self.asr = AsyncASR(asr_model)
        self.mt = AsyncMT(mt_model)
        self.tts = AsyncTTS(tts_model)
    
    async def process_segment(self, audio_segment, src_codes, tgt_codes, speaker_wav=None):
        """
        Standard process_segment method for backward compatibility.
        """
        return await self.process_segment_streaming(
            audio_segment, src_codes, tgt_codes, speaker_wav, ws=None
        )
    
    async def process_segment_streaming(self, audio_segment, src_codes, tgt_codes, 
                                      speaker_wav=None, ws=None):
        """
        Process segment with streaming results to prevent WebSocket timeout.
        Sends intermediate results (ASR, MT) before TTS completes.
        """
        result = {
            'asr_text': '',
            'mt_text': '', 
            'audio': None,
            'sample_rate': 24000,
            'success': False,
            'errors': []
        }
        
        try:
            # Step 1: ASR
            start_time = time.time()
            try:
                asr_text = await self.asr.transcribe(
                    audio_segment, 
                    lang=src_codes.get("whisper")
                )
                
                if not asr_text or not asr_text.strip():
                    result['errors'].append("Empty ASR result")
                    return result
                    
                result['asr_text'] = asr_text.strip()
                asr_time = time.time() - start_time
                log.info(f"ASR completed in {asr_time:.1f}s: '{asr_text[:100]}...'")
                
                # Send ASR result immediately if WebSocket provided
                if ws:
                    import json
                    await ws.send_text(json.dumps({
                        "type": "asr", 
                        "text": result['asr_text']
                    }))
                
            except Exception as e:
                result['errors'].append(f"ASR failed: {str(e)}")
                return result

            # Step 2: MT
            mt_start = time.time()
            try:
                mt_text = await self.mt.translate(
                    result['asr_text'],
                    src=src_codes.get("mt", "en"),
                    tgt=tgt_codes.get("mt", "en")
                )
                
                result['mt_text'] = mt_text.strip() if mt_text else result['asr_text']
                mt_time = time.time() - mt_start
                log.info(f"MT completed in {mt_time:.1f}s: '{result['mt_text'][:100]}...'")
                
                # Send MT result immediately if WebSocket provided
                if ws:
                    import json
                    await ws.send_text(json.dumps({
                        "type": "mt", 
                        "text": result['mt_text']
                    }))
                
            except Exception as e:
                log.warning(f"MT failed, using original text: {e}")
                result['mt_text'] = result['asr_text']
                
                if ws:
                    import json
                    await ws.send_text(json.dumps({
                        "type": "mt", 
                        "text": result['mt_text']
                    }))

            # Step 3: TTS (potentially long operation)
            tts_start = time.time()
            try:
                # Notify that TTS is starting (for long texts)
                if len(result['mt_text']) > 200 and ws:
                    import json
                    await ws.send_text(json.dumps({
                        "type": "info",
                        "message": f"Generating speech for {len(result['mt_text'])} characters..."
                    }))
                
                audio, sr = await self.tts.synthesize(
                    result['mt_text'],
                    lang=tgt_codes.get("tts", "en"),
                    speaker_wav=speaker_wav
                )
                
                result['audio'] = audio
                result['sample_rate'] = sr
                result['success'] = True
                
                tts_time = time.time() - tts_start
                total_time = time.time() - start_time
                
                if len(audio) > 0:
                    audio_duration = len(audio) / sr
                    log.info(f"TTS completed in {tts_time:.1f}s: {audio_duration:.1f}s audio (total: {total_time:.1f}s)")
                else:
                    log.warning(f"TTS completed but returned empty audio (took {tts_time:.1f}s)")
                    result['errors'].append("TTS returned empty audio")
                    
            except Exception as e:
                error_msg = f"TTS failed: {str(e)}"
                log.error(error_msg)
                result['errors'].append(error_msg)
                result['success'] = True  # Still success since we have text
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            result['errors'].append(f"Pipeline error: {str(e)}")
            log.exception("Error in streaming segment processing")
        
        return result
    
    async def process_text_translation(self, text, src_codes, tgt_codes, speaker_wav=None):
        """Process text-only translation (skip ASR step)."""
        result = {
            'asr_text': text,
            'mt_text': '', 
            'audio': None,
            'sample_rate': 24000,
            'success': False,
            'errors': []
        }
        
        try:
            # MT step
            mt_text = await self.mt.translate(
                text,
                src=src_codes.get("mt", "en"),
                tgt=tgt_codes.get("mt", "en")
            )
            
            result['mt_text'] = mt_text or text
            
            # TTS step
            audio, sr = await self.tts.synthesize(
                result['mt_text'],
                lang=tgt_codes.get("tts", "en"),
                speaker_wav=speaker_wav
            )
            
            result['audio'] = audio
            result['sample_rate'] = sr
            result['success'] = True
            
            gc.collect()
            
        except Exception as e:
            result['errors'].append(str(e))
            log.exception("Error in text translation processing")
        
        return result