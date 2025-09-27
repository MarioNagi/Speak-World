# server/app.py - Fixed complete version
from __future__ import annotations
import os
import sys
import traceback
import json
import torch
import yaml
import asyncio
import logging
import atexit
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from pydantic import BaseModel

# Local imports
from language_utils import validate_language_pair, get_supported_languages, get_language_families, is_bidirectional_supported
from async_models import AsyncModelManager, cleanup_executor
from utils.audio import validate_audio_segment, normalize_audio, pcm16_bytes_to_float32_mono, resample, float32_to_pcm16_bytes
from vad import VADChunker
from asr import ASR
from mt import MT
from dual_mt import DualMTManager
from tts import TTSEngine
from voice_bank import VoiceBank
from monitoring import metrics, check_model_health, log_system_resources

startup_time = time.time()

# ---------------- Logging setup ----------------
LOG_LEVEL = os.environ.get("V2V_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("v2v")

# ---------------- Config load ----------------
def load_config() -> dict:
    tried = []
    for p in (Path("server/config.yaml"), Path("./config.yaml")):
        tried.append(str(p))
        if p.exists():
            log.info(f"Loading config from {p.resolve()}")
            with p.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            log.debug(f"Config content: {json.dumps(cfg, indent=2)}")
            return cfg
    raise FileNotFoundError(f"config.yaml not found. Tried: {tried}")

try:
    CFG = load_config()
except Exception as e:
    log.exception("Failed to load configuration")
    raise

def get_vad_cfg(cfg: dict) -> dict:
    if "vad" in cfg:
        return cfg["vad"]
    s = cfg.get("stream", {})
    return {
        "sample_rate": s.get("in_sr", 16000),
        "frame_ms": 20,
        "aggressiveness": s.get("vad_aggressiveness", 2),
        "end_silence_ms": s.get("end_silence_ms", 500),
    }

ASR_CFG = CFG.get("asr", {})
MT_CFG = CFG.get("mt", {})
TTS_CFG = CFG.get("tts", {})
VAD_CFG = get_vad_cfg(CFG)

# ---------------- FastAPI ----------------
app = FastAPI(title="Real-time V2V Translator", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Mount static files and templates
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    log.warning("Static directory not found, creating it")
    Path("static").mkdir(exist_ok=True)
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except:
        pass

templates = Jinja2Templates(directory="templates")

# ---------------- Pydantic Models ----------------
class Health(BaseModel):
    status: str = "ok"

class LanguageInfo(BaseModel):
    key: str
    display: str
    family: str
    preferred_mt: str

class TranslationSession(BaseModel):
    session_id: str
    lang1: str
    lang1_display: str
    lang2: str
    lang2_display: str
    mt_model: str
    is_bidirectional: bool
    created_at: float

class SessionRequest(BaseModel):
    lang1: str
    lang2: str
    speaker_id: Optional[str] = None

# ---------------- Model Initialization ----------------
def _init_asr() -> ASR:
    log.info("Initializing ASR...")
    try:
        m = ASR(
            model_name=ASR_CFG.get("model", "small"),
            device=ASR_CFG.get("device", "cpu"),
            compute_type=ASR_CFG.get("compute_type", "int8"),
        )
        log.info("ASR ready")
        return m
    except Exception as e:
        log.exception("ASR init failed")
        raise

def _init_mt() -> DualMTManager:
    log.info("=" * 50)
    log.info("STARTING DUAL MT SYSTEM INITIALIZATION")
    log.info("=" * 50)
    try:
        mt_config = {
            "model": MT_CFG.get("model", "facebook/m2m100_418M"),
            "max_new_tokens": int(MT_CFG.get("max_new_tokens", 160)),
            "device": MT_CFG.get("device"),
        }
        log.info(f"MT config: {mt_config}")
        log.info("Loading both IndicTrans2 (Indic languages) and M2M100 (other languages)...")
        log.info("First-time download may take several minutes, subsequent runs will be fast")

        dual_mt = DualMTManager(mt_config)
        
        # Log which models are available
        info = dual_mt.get_model_info()
        log.info(f"✓ Dual MT system initialized: {info['total_models']} models loaded")
        if info['indictrans_available']:
            log.info("  ✅ IndicTrans2: Ready for English ↔ Indic translations")
        if info['m2m100_available']:
            log.info("  ✅ M2M100: Ready for non-Indic language pairs (English-German, etc.)")
        
        return dual_mt
    except Exception as e:
        log.error("✗ MT initialization FAILED")
        log.error(f"Error: {e}")
        traceback.print_exc()
        raise

def _init_tts() -> TTSEngine:
    log.info("Initializing TTS...")
    try:
        engine = TTSEngine(TTS_CFG if isinstance(TTS_CFG, dict) else {})
        log.info("TTS ready")
        return engine
    except Exception as e:
        log.exception("TTS init failed")
        raise

def _init_vad() -> VADChunker:
    log.info(f"Initializing VAD with {VAD_CFG} ...")
    try:
        v = VADChunker(
            sample_rate=int(VAD_CFG.get("sample_rate", 16000)),
            frame_ms=int(VAD_CFG.get("frame_ms", 20)),
            aggressiveness=int(VAD_CFG.get("aggressiveness", 2)),
            end_silence_ms=int(VAD_CFG.get("end_silence_ms", 500)),
        )
        log.info("VAD ready")
        return v
    except Exception as e:
        log.exception("VAD init failed")
        raise

# Enhanced AsyncModelManager with text translation support
class EnhancedAsyncModelManager(AsyncModelManager):
    """Enhanced manager with text-only translation support."""
    
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
            # Skip ASR, go straight to MT
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
            
            import gc
            gc.collect()
            
        except Exception as e:
            result['errors'].append(str(e))
            log.exception("Error in text translation processing")
        
        return result

# ---------------- Model Loading ----------------
log.info("🚀 Starting V2V Translation Server...")
log.info(f"Python: {sys.version}")
log.info(f"PyTorch: {torch.__version__}")

try:
    log.info("Phase 1/5: Initializing ASR...")
    asr = _init_asr()
    
    log.info("Phase 2/5: Initializing MT...")
    mt = _init_mt()
    
    log.info("Phase 3/5: Initializing TTS...")
    tts = _init_tts()
    
    log.info("Phase 4/5: Initializing Voice Bank...")
    voices = VoiceBank("voices")
    log.info(f"Voice bank ready at {Path('voices').resolve()}")
    
    log.info("Phase 5/5: Creating Enhanced Async Model Manager...")
    async_models = EnhancedAsyncModelManager(asr, mt, tts)
    log.info("Enhanced async model manager initialized")
    
    log.info("=" * 50)
    log.info("🎉 ALL MODELS LOADED SUCCESSFULLY!")
    log.info("Server is ready to accept connections")
    log.info("=" * 50)
    
except Exception as e:
    log.error("=" * 50)
    log.error("💥 CRITICAL STARTUP FAILURE")
    log.error(f"Error: {e}")
    log.error("=" * 50)
    traceback.print_exc()
    sys.exit(1)

# Register cleanup
atexit.register(cleanup_executor)

# Session management
active_sessions = {}

# ---------------- Routes ----------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main translation interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return Health()

@app.get("/languages/supported", response_model=List[LanguageInfo])
def get_languages():
    """Get all supported languages with metadata."""
    return get_supported_languages()

@app.get("/languages/families")
def get_families():
    """Get languages grouped by family for UI organization."""
    return get_language_families()

@app.post("/session/create")
def create_session(request: SessionRequest):
    """Create a new bidirectional translation session."""
    try:
        # Validate language pair
        lang1_codes, lang2_codes, mt_model = validate_language_pair(request.lang1, request.lang2)
        
        # Check bidirectional support
        bidirectional = is_bidirectional_supported(request.lang1, request.lang2)
        
        # Create session
        session_id = str(uuid.uuid4())
        session = TranslationSession(
            session_id=session_id,
            lang1=request.lang1,
            lang1_display=lang1_codes["display"],
            lang2=request.lang2,
            lang2_display=lang2_codes["display"],
            mt_model=mt_model,
            is_bidirectional=bidirectional,
            created_at=time.time()
        )
        
        # Store session
        active_sessions[session_id] = {
            "session": session,
            "lang1_codes": lang1_codes,
            "lang2_codes": lang2_codes,
            "speaker_id": request.speaker_id,
        }
        
        # Clean old sessions (keep last 100)
        if len(active_sessions) > 100:
            oldest_sessions = sorted(active_sessions.items(), key=lambda x: x[1]["session"].created_at)
            for old_session_id, _ in oldest_sessions[:-100]:
                del active_sessions[old_session_id]
        
        return {
            "session": session.dict(),
            "lang1_codes": lang1_codes,
            "lang2_codes": lang2_codes,
        }
        
    except Exception as e:
        return {"error": str(e)}, 400

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Get session information."""
    if session_id not in active_sessions:
        return {"error": "Session not found"}, 404
    
    session_data = active_sessions[session_id]
    return {
        "session": session_data["session"].dict(),
        "lang1_codes": session_data["lang1_codes"],
        "lang2_codes": session_data["lang2_codes"],
    }

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session."""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": "Session deleted"}
    return {"error": "Session not found"}, 404

@app.get("/session/list")
def list_sessions():
    """List all active sessions."""
    return {
        "sessions": [
            {
                "session_id": session_id,
                "lang1": data["session"].lang1_display,
                "lang2": data["session"].lang2_display,
                "mt_model": data["session"].mt_model,
                "created_at": data["session"].created_at,
            }
            for session_id, data in active_sessions.items()
        ]
    }

@app.websocket("/ws/stream")
async def ws_stream_legacy(websocket: WebSocket):
    """Legacy WebSocket endpoint with streaming support for long audio."""
    await websocket.accept()
    client_id = f"client_{int(time.time() * 1000)}"
    log.info(f"Legacy WS {client_id} connected")
    
    vad = None
    speaker_wav = None
    src_codes = {"key": "en", "mt": "en", "tts": "en", "display": "English"}
    tgt_codes = {"key": "ne", "mt": "ne", "tts": "ne", "display": "Nepali"}
    
    try:
        # Wait for initial message with longer timeout
        msg = await asyncio.wait_for(websocket.receive(), timeout=30.0)
        
        if "text" in msg:
            try:
                hello = json.loads(msg["text"])
                src_lang = hello.get("src", "en")
                tgt_lang = hello.get("tgt", "ne") 
                speaker_id = hello.get("speaker")
                
                # Validate language pair and get codes
                try:
                    src_codes, tgt_codes, mt_model = validate_language_pair(src_lang, tgt_lang)
                    speaker_wav = voices.resolve(speaker_id) if speaker_id else None
                    
                    log.info(f"{client_id} session: {src_codes['display']} -> {tgt_codes['display']} using {mt_model}")
                    
                    # Send ready message
                    await websocket.send_text(json.dumps({
                        "type": "ready",
                        "tgt": tgt_codes["display"],
                        "speaker": speaker_id or "neutral"
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": f"Language pair validation failed: {str(e)}"
                    }))
                    return
                    
            except json.JSONDecodeError:
                log.error("Invalid JSON in initial message")
                return
        
        # Initialize VAD
        vad = _init_vad()
        log.info("VAD ready for streaming mode")
        
        # Main processing loop with heartbeat
        last_heartbeat = time.time()
        heartbeat_interval = 30.0  # Send heartbeat every 30 seconds
        
        while True:
            try:
                # Non-blocking receive with timeout
                try:
                    msg = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if we need to send heartbeat
                    current_time = time.time()
                    if current_time - last_heartbeat > heartbeat_interval:
                        try:
                            await websocket.send_text(json.dumps({"type": "heartbeat"}))
                            last_heartbeat = current_time
                        except:
                            log.info(f"Legacy WS {client_id} heartbeat failed - connection closed")
                            break
                    continue
                
                if "bytes" in msg:
                    # Audio frame received
                    frame = msg["bytes"]
                    seg = vad.add_frame(frame)
                    
                    if seg is not None:
                        # Process complete audio segment with streaming results
                        log.info(f"VAD: Processing segment of {len(seg)} bytes")
                        
                        # Convert PCM16 bytes to float32 mono
                        a = pcm16_bytes_to_float32_mono(seg)
                        a16k = resample(a, int(VAD_CFG.get("sample_rate", 16000)), 16000)
                        
                        if not validate_audio_segment(a16k):
                            continue
                        
                        a16k = normalize_audio(a16k)
                        
                        # Use streaming processing to prevent timeout
                        try:
                            result = await async_models.process_segment_streaming(
                                a16k, src_codes, tgt_codes, speaker_wav, websocket
                            )
                            
                            # Send final audio if available
                            if result.get('audio') is not None and len(result['audio']) > 0:
                                try:
                                    # Send audio metadata first
                                    await websocket.send_text(json.dumps({
                                        "type": "audio",
                                        "sr": result['sample_rate'],
                                        "duration": len(result['audio']) / result['sample_rate']
                                    }))
                                    
                                    # Send audio data
                                    audio_bytes = float32_to_pcm16_bytes(result['audio'])
                                    await websocket.send_bytes(audio_bytes)
                                    log.info(f"Sent {len(audio_bytes)} bytes of audio")
                                    
                                except Exception as e:
                                    log.error(f"Failed to send audio: {e}")
                                    await websocket.send_text(json.dumps({
                                        "type": "error",
                                        "error": "Audio transmission failed"
                                    }))
                            elif result.get('errors'):
                                # Send errors if any
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "errors": result['errors']
                                }))
                                
                        except Exception as e:
                            log.exception(f"Audio processing failed: {e}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "error": f"Processing failed: {str(e)}"
                            }))
                        
                        # Update heartbeat after processing
                        last_heartbeat = time.time()
                        
                elif "text" in msg:
                    data = msg["text"]
                    if data == "flush":
                        log.info(f"{client_id} flush requested")
                        seg = vad.flush()
                        if seg:
                            # Process flushed segment
                            a = pcm16_bytes_to_float32_mono(seg)
                            a16k = resample(a, int(VAD_CFG.get("sample_rate", 16000)), 16000)
                            
                            if validate_audio_segment(a16k):
                                a16k = normalize_audio(a16k)
                                result = await async_models.process_segment_streaming(
                                    a16k, src_codes, tgt_codes, speaker_wav, websocket
                                )
                                
                                # Send final audio
                                if result.get('audio') is not None and len(result['audio']) > 0:
                                    await websocket.send_text(json.dumps({
                                        "type": "audio", 
                                        "sr": result['sample_rate']
                                    }))
                                    await websocket.send_bytes(float32_to_pcm16_bytes(result['audio']))
                    else:
                        # Handle other text messages
                        try:
                            command = json.loads(data)
                            if command.get("type") == "ping":
                                await websocket.send_text(json.dumps({"type": "pong"}))
                                last_heartbeat = time.time()
                        except:
                            log.debug(f"Non-JSON text message: {data[:50]}")
                            
            except WebSocketDisconnect:
                log.info(f"Legacy WS {client_id} received disconnect")
                break
            except RuntimeError as e:
                if "Cannot call" in str(e) and "disconnect" in str(e):
                    log.info(f"Legacy WS {client_id} already disconnected")
                    break
                else:
                    log.exception(f"Runtime error in WebSocket processing: {e}")
                    break
            except Exception as e:
                log.exception(f"Error in WebSocket processing: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": "Processing error occurred"
                    }))
                except:
                    log.info(f"Could not send error message, connection likely closed")
                    break
                    
    except WebSocketDisconnect:
        log.info(f"Legacy WS {client_id} disconnected normally")
    except asyncio.TimeoutError:
        log.info(f"Legacy WS {client_id} init timeout")
    except Exception as e:
        log.exception(f"Legacy WS {client_id} error: {e}")
    finally:
        log.info(f"Legacy WS {client_id} cleanup complete")


@app.websocket("/ws/translate/{session_id}")
async def ws_translate(ws: WebSocket, session_id: str):
    """Enhanced WebSocket for bidirectional translation with streaming support."""
    await ws.accept()
    client_id = f"client_{int(time.time() * 1000)}"
    log.info(f"WS {client_id} connected for session {session_id}")
    
    if session_id not in active_sessions:
        await ws.close(code=1008, reason="Session not found")
        return
    
    session_data = active_sessions[session_id]
    session = session_data["session"]
    lang1_codes = session_data["lang1_codes"]
    lang2_codes = session_data["lang2_codes"]
    speaker_id = session_data.get("speaker_id")
    
    vad = None
    speaker_wav = None
    
    try:
        # Initialize VAD
        vad = _init_vad()
        
        # Resolve speaker
        speaker_wav = voices.resolve(speaker_id) if speaker_id else None
        
        # Send ready message
        await ws.send_text(json.dumps({
            "type": "ready",
            "session_id": session_id,
            "lang1": lang1_codes["display"],
            "lang2": lang2_codes["display"],
            "mt_model": session.mt_model,
            "speaker": speaker_id or "neutral"
        }))
        
        while True:
            try:
                msg = await ws.receive()
                
                # Check for disconnect message
                if msg["type"] == "websocket.disconnect":
                    log.info(f"WS {client_id} received disconnect message")
                    break
                
                if "bytes" in msg:
                    log.debug(f"WS {client_id} received binary data: {len(msg['bytes'])} bytes")
                    frame = msg["bytes"]
                    seg = vad.add_frame(frame)
                    
                    if seg is not None:
                        log.info(f"WS {client_id} VAD segment complete: {len(seg)} bytes")
                        await ws.send_text(json.dumps({
                            "type": "info",
                            "message": "Processing audio segment..."
                        }))
                        
                elif "text" in msg:
                    data = msg["text"]
                    
                    try:
                        command = json.loads(data)
                        command_type = command.get("type")
                        
                        if command_type == "translate":
                            direction = command.get("direction")
                            text_input = command.get("text")
                            
                            if direction == "lang1_to_lang2":
                                src_codes, tgt_codes = lang1_codes, lang2_codes
                            elif direction == "lang2_to_lang1":
                                src_codes, tgt_codes = lang2_codes, lang1_codes
                            else:
                                await ws.send_text(json.dumps({
                                    "type": "error",
                                    "error": "Invalid direction"
                                }))
                                continue
                            
                            # Process translation
                            if text_input:
                                # Text-only translation
                                result = await async_models.process_text_translation(
                                    text_input, src_codes, tgt_codes, speaker_wav
                                )
                            else:
                                # Audio translation with streaming
                                seg = vad.flush()
                                if not seg:
                                    continue
                                    
                                a = pcm16_bytes_to_float32_mono(seg)
                                a16k = resample(a, int(VAD_CFG.get("sample_rate", 16000)), 16000)
                                
                                if not validate_audio_segment(a16k):
                                    continue
                                
                                a16k = normalize_audio(a16k)
                                
                                # Use streaming approach for long audio
                                result = await async_models.process_segment_streaming(
                                    a16k, src_codes, tgt_codes, speaker_wav, ws
                                )
                            
                            # Send final response
                            response = {
                                "type": "translation_result", 
                                "direction": direction,
                                "asr_text": result.get('asr_text', ''),
                                "translated_text": result.get('mt_text', ''),
                                "audio_sr": result.get('sample_rate', 24000)
                            }
                            
                            if result.get('errors'):
                                response["warnings"] = result['errors']
                            
                            await ws.send_text(json.dumps(response))
                            
                            # Send audio if available
                            if result.get('audio') is not None and len(result['audio']) > 0:
                                try:
                                    await ws.send_bytes(float32_to_pcm16_bytes(result['audio']))
                                except Exception as audio_error:
                                    log.error(f"WS {client_id} failed to send audio: {audio_error}")
                        
                    except json.JSONDecodeError:
                        if data == "flush":
                            seg = vad.flush() if vad else None
                            # Handle flush
                            
            except WebSocketDisconnect:
                log.info(f"WS {client_id} disconnected normally")
                break
            except Exception as e:
                log.exception(f"WS {client_id} error: {e}")
                break
    
    except WebSocketDisconnect:
        log.info(f"WS {client_id} disconnected normally")
    except Exception as e:
        log.exception(f"WS {client_id} error: {e}")
    finally:
        log.info(f"WS {client_id} cleanup complete")

@app.websocket("/ws/translate/{session_id}")
async def ws_translate(ws: WebSocket, session_id: str):
    """Enhanced WebSocket for bidirectional translation."""
    await ws.accept()
    client_id = f"client_{int(time.time() * 1000)}"
    log.info(f"WS {client_id} connected for session {session_id}")
    
    if session_id not in active_sessions:
        await ws.close(code=1008, reason="Session not found")
        return
    
    session_data = active_sessions[session_id]
    session = session_data["session"]
    lang1_codes = session_data["lang1_codes"]
    lang2_codes = session_data["lang2_codes"]
    speaker_id = session_data.get("speaker_id")
    
    vad = None
    speaker_wav = None
    
    try:
        # Initialize VAD
        vad = _init_vad()
        
        # Resolve speaker
        speaker_wav = voices.resolve(speaker_id) if speaker_id else None
        
        # Send ready message
        await ws.send_text(json.dumps({
            "type": "ready",
            "session_id": session_id,
            "lang1": lang1_codes["display"],
            "lang2": lang2_codes["display"],
            "mt_model": session.mt_model,
            "speaker": speaker_id or "neutral"
        }))
        
        while True:
            try:
                log.debug(f"WS {client_id} waiting for message...")
                msg = await ws.receive()
                
                # Check for disconnect message
                if msg["type"] == "websocket.disconnect":
                    log.info(f"WS {client_id} received disconnect message")
                    break
                    
                log.debug(f"WS {client_id} received message type: {type(msg)}")
                
                if "bytes" in msg:
                    log.info(f"WS {client_id} received binary data: {len(msg['bytes'])} bytes")
                    frame = msg["bytes"]
                    seg = vad.add_frame(frame)
                    
                    if seg is not None:
                        log.info(f"WS {client_id} VAD segment complete: {len(seg)} bytes")
                        await ws.send_text(json.dumps({
                            "type": "info",
                            "message": "Processing audio segment..."
                        }))
                        
                elif "text" in msg:
                    data = msg["text"]
                    log.info(f"WS {client_id} received text message: {data[:100]}...")
                    
                    try:
                        command = json.loads(data)
                        command_type = command.get("type")
                        log.info(f"WS {client_id} command type: {command_type}")
                        
                        if command_type == "translate":
                            direction = command.get("direction")
                            text_input = command.get("text")
                            log.info(f"WS {client_id} translate command: direction={direction}, text={text_input}")
                            
                            if direction == "lang1_to_lang2":
                                src_codes, tgt_codes = lang1_codes, lang2_codes
                            elif direction == "lang2_to_lang1":
                                src_codes, tgt_codes = lang2_codes, lang1_codes
                            else:
                                log.error(f"WS {client_id} invalid direction: {direction}")
                                await ws.send_text(json.dumps({
                                    "type": "error",
                                    "error": "Invalid direction"
                                }))
                                continue
                            
                            # Process translation
                            if text_input:
                                result = await async_models.process_text_translation(
                                    text_input, src_codes, tgt_codes, speaker_wav
                                )
                            else:
                                seg = vad.flush()
                                if not seg:
                                    continue
                                    
                                a = pcm16_bytes_to_float32_mono(seg)
                                a16k = resample(a, int(VAD_CFG.get("sample_rate", 16000)), 16000)
                                
                                if not validate_audio_segment(a16k):
                                    continue
                                
                                a16k = normalize_audio(a16k)
                                
                                # Process with timeout to prevent WebSocket hanging
                                log.info(f"WS {client_id} processing audio segment for {direction}")
                                processing_timeout = 30.0  # 30 second max for entire processing pipeline
                                
                                try:
                                    result = await asyncio.wait_for(
                                        async_models.process_segment(a16k, src_codes, tgt_codes, speaker_wav),
                                        timeout=processing_timeout
                                    )
                                except asyncio.TimeoutError:
                                    log.error(f"WS {client_id} processing timeout after {processing_timeout}s")
                                    # Send partial result with text-only
                                    result = {
                                        'success': True,
                                        'asr_text': 'Processing timeout - please try shorter messages',
                                        'mt_text': 'Processing timeout - please try shorter messages',
                                        'audio': None,
                                        'sample_rate': 24000,
                                        'errors': ['Processing timeout - TTS unavailable for long texts']
                                    }
                            
                            # Send results - Always send text results even if TTS fails
                            response = {
                                "type": "translation_result", 
                                "direction": direction,
                                "asr_text": result.get('asr_text', ''),
                                "translated_text": result.get('mt_text', ''),
                                "audio_sr": result.get('sample_rate', 24000)
                            }
                            
                            # Add warning if there were errors but we still got text
                            if result.get('errors') and result.get('success'):
                                response["warnings"] = result['errors']
                                log.warning(f"WS {client_id} translation had warnings: {result['errors']}")
                            
                            await ws.send_text(json.dumps(response))
                            
                            # Send audio only if available
                            if result.get('audio') is not None and len(result['audio']) > 0:
                                try:
                                    await ws.send_bytes(float32_to_pcm16_bytes(result['audio']))
                                    log.debug(f"WS {client_id} sent {len(result['audio'])} audio samples")
                                except Exception as audio_error:
                                    log.error(f"WS {client_id} failed to send audio: {audio_error}")
                                    # Send notification about audio failure
                                    await ws.send_text(json.dumps({
                                        "type": "info",
                                        "message": "Translation complete, but audio playback failed"
                                    }))
                            else:
                                # Notify if no audio was generated
                                if result.get('errors'):
                                    await ws.send_text(json.dumps({
                                        "type": "info", 
                                        "message": "Translation complete (text only - TTS unavailable)"
                                    }))
                            
                            # If completely failed, still send error
                            if not result.get('success'):
                                await ws.send_text(json.dumps({
                                    "type": "error",
                                    "errors": result.get('errors', ['Unknown processing error']),
                                    "direction": direction
                                }))
                        
                    except json.JSONDecodeError:
                        if data == "flush":
                            log.info(f"WS {client_id} received flush command")
                            seg = vad.flush() if vad else None
                            # Handle flush
                        else:
                            log.error(f"WS {client_id} invalid JSON: {data[:50]}...")
                            
            except WebSocketDisconnect:
                log.info(f"WS {client_id} disconnected normally")
                break
            except Exception as e:
                log.exception(f"WS {client_id} error: {e}")
                try:
                    await ws.close(code=1011, reason=str(e))
                except:
                    pass
                break
    
    except WebSocketDisconnect:
        log.info(f"WS {client_id} disconnected normally")
    except Exception as e:
        log.exception(f"WS {client_id} error: {e}")
        try:
            await ws.close(code=1011, reason=str(e))
        except:
            pass
    finally:
        log.info(f"WS {client_id} cleanup complete")

# Voice enrollment
@app.post("/voices/enroll")
async def enroll_voice(file: UploadFile = File(...), name: str = Form(default=None)):
    tmp_dir = Path("voices")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"tmp_{file.filename}"
    log.info(f"Uploading voice to {tmp}")
    try:
        with open(tmp, "wb") as f:
            f.write(await file.read())
        sid = voices.enroll(str(tmp), name)
        log.info(f"Enrolled speaker_id={sid}")
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return {"speaker_id": sid}

# Additional endpoints
@app.get("/health/models")
def health_models():
    try:
        return {
            "status": "ok",
            "models": {
                "asr": "loaded",
                "mt": "loaded", 
                "tts": "loaded",
                "voices": len(voices._speakers) if hasattr(voices, '_speakers') else 0
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/metrics")
def get_metrics(window_minutes: int = 60):
    return metrics.get_stats(window_minutes)

@app.get("/health/detailed")
def detailed_health():
    try:
        model_health = check_model_health(asr, mt, tts)
        voice_stats = voices.get_stats()
        processing_stats = metrics.get_stats(window_minutes=10)
        
        return {
            "status": "ok",
            "timestamp": time.time(),
            "models": model_health,
            "voice_bank": voice_stats,
            "recent_performance": {
                "operations_last_10min": processing_stats["total_operations"],
                "success_rate_percent": processing_stats["overall"]["success_rate"],
                "avg_processing_time_ms": processing_stats["overall"]["avg_duration_ms"]
            },
            "system": {
                "active_sessions": len(active_sessions),
                "uptime_seconds": time.time() - startup_time
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/voices/list")
def list_voices():
    return {"voices": voices.list_speakers()}

@app.delete("/voices/{speaker_id}")
def delete_voice(speaker_id: str):
    success = voices.remove_speaker(speaker_id)
    if success:
        return {"message": f"Speaker '{speaker_id}' removed successfully"}
    else:
        return {"error": f"Failed to remove speaker '{speaker_id}'"}, 400

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )