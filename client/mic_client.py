import argparse, asyncio, json, sys
import numpy as np
import sounddevice as sd
import websockets
import soundfile as sf
from pathlib import Path

def pcm16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def load_audio_file(file_path, target_sr=16000):
    """Load audio file and resample to target sample rate."""
    try:
        audio, sr = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            # Simple resampling (for more accurate resampling, use librosa)
            from scipy import signal
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)
        
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

async def run(url, tgt, speaker, audio_file=None, in_sr=16000, out_sr_hint=24000):
    """Connect to server websocket. Try a few common ports if the provided url fails.

    Inputs:
    - url: websocket URL (e.g. ws://localhost:8080/ws/stream)
    - tgt: target language
    - speaker: speaker id or 'none'

    Behavior: attempts the exact url, then replaces the port with 8000, 8080, 8081 as fallbacks.
    """
    # Prepare list of candidate URLs to try
    candidates = [url]
    try:
        # attempt to extract host and path and try common ports
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        host = parsed.hostname or '180.94.16.204'
        path = parsed.path or '/ws/stream'
        scheme = parsed.scheme or 'ws'
        for p in (8000, 8080, 8081):
            cand = urlunparse((scheme, f"{host}:{p}", path, '', '', ''))
            if cand not in candidates:
                candidates.append(cand)
    except Exception:
        pass

    ws = None
    last_exc = None
    for attempt_url in candidates:
        try:
            print(f"Attempting to connect to {attempt_url} ...")
            ws = await websockets.connect(attempt_url, max_size=None, ping_interval=20, ping_timeout=10)
            break
        except Exception as e:
            print(f"Failed to connect to {attempt_url}: {e}")
            last_exc = e

    if ws is None:
        # Re-raise the last exception to bubble up a clear error
        raise ConnectionError(f"Could not connect to server. Last error: {last_exc}")

    # Use explicit try/finally to manage the websocket lifetime. Some versions of
    # the websockets library return a protocol object that isn't an async context manager.
    try:
        hello = {"src": "en", "tgt": tgt, "speaker": None if speaker == "none" else speaker}
        await ws.send(json.dumps(hello))
        print("Connected, streaming mic → server…  (Ctrl+C to stop)")

        # Check input mode and audio device availability
        use_file_input = audio_file is not None
        audio_data = None
        
        if use_file_input:
            print(f"Loading audio file: {audio_file}")
            audio_data = load_audio_file(audio_file, in_sr)
            if audio_data is None:
                raise ValueError(f"Failed to load audio file: {audio_file}")
            print(f"Loaded {len(audio_data)/in_sr:.2f} seconds of audio")
            audio_available = True  # We have audio data even without devices
        else:
            # Check if audio devices are available for microphone input
            try:
                sd.query_devices()
                in_stream = sd.InputStream(channels=1, samplerate=in_sr, blocksize=int(in_sr*0.02))
                audio_available = True
                print("Using microphone input")
            except sd.PortAudioError:
                print("Warning: No audio devices found. Cannot use microphone.")
                audio_available = False
                in_stream = None

        # Try to set up output stream for audio playback (optional)
        try:
            out_stream = sd.OutputStream(channels=1, samplerate=out_sr_hint, blocksize=int(out_sr_hint*0.02))
            output_available = True
        except sd.PortAudioError:
            print("Warning: No output devices found. Audio playback disabled.")
            out_stream = None
            output_available = False

        if audio_available and not use_file_input and in_stream:
            in_stream.start()
        if output_available and out_stream:
            out_stream.start()

        async def recv_loop():
            nonlocal out_stream, out_sr_hint, output_available
            while True:
                try:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        if output_available and out_stream:
                            buf = np.frombuffer(msg, dtype=np.int16).astype(np.float32)/32768.0
                            out_stream.write(buf.reshape(-1,1))
                    else:
                        try:
                            data = json.loads(msg)
                            if data.get("type") == "audio":
                                sr = int(data.get("sr", out_sr_hint))
                                if sr != out_sr_hint and output_available and out_stream:
                                    print(f"[server] switching output SR to {sr}")
                                    out_stream.stop(); out_stream.close()
                                    out_sr_hint = sr
                                    out_stream = sd.OutputStream(channels=1, samplerate=out_sr_hint, blocksize=int(out_sr_hint*0.02))
                                    out_stream.start()
                            elif data.get("type") == "asr":
                                print(f"[ASR] {data['text']}")
                            elif data.get("type") == "mt":
                                print(f"[MT ] {data['text']}")
                            elif data.get("type") == "ready":
                                print(f"Server ready: tgt={data['tgt']} voice={data['speaker']}")
                            elif data.get("type") == "error":
                                print(f"[ERROR] {data}")
                        except Exception as e:
                            print(f"Error parsing message: {e}")
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by server")
                    break
                except Exception as e:
                    print(f"Error in receive loop: {e}")
                    break

        async def send_loop():
            if use_file_input and audio_data is not None:
                # Send file audio in chunks
                chunk_size = int(in_sr * 0.02)  # 20ms chunks
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    if len(chunk) < chunk_size:
                        # Pad last chunk with zeros
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                    try:
                        await ws.send(pcm16(chunk).tobytes())
                        await asyncio.sleep(0.02)  # Real-time playback
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection lost during file transmission")
                        break
                print("Finished sending audio file")
                # Keep connection alive after file is sent
                while True:
                    await asyncio.sleep(1.0)
            else:
                # Microphone input mode
                while True:
                    try:
                        if audio_available and in_stream:
                            blk = in_stream.read(int(in_sr*0.02))[0][:,0]
                            await ws.send(pcm16(blk).tobytes())
                        else:
                            # In text-only mode, simulate silence
                            await asyncio.sleep(0.02)
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection lost during microphone streaming")
                        break
                    except Exception as e:
                        print(f"Error in send loop: {e}")
                        await asyncio.sleep(0.1)  # Brief pause before retry

        try:
            await asyncio.gather(recv_loop(), send_loop())
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection was closed")
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            print("Session ended")
    finally:
        try:
            if ws and not ws.closed:
                await ws.close()
        except Exception:
            pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Voice-to-Voice Translation Client")
    ap.add_argument("--url", default="ws://localhost:9000/ws/stream", help="WebSocket server URL") 
    ap.add_argument("--tgt", default="ne", help="target language code (e.g., ne, ar, hi, en)")
    ap.add_argument("--speaker", default="none", help="speaker_id from enrollment or 'none'")
    ap.add_argument("--file", "-f", help="audio file path (if not provided, uses microphone)")
    args = ap.parse_args()
    
    if args.file and not Path(args.file).exists():
        print(f"Error: Audio file not found: {args.file}")
        sys.exit(1)
    
    try:
        asyncio.run(run(args.url, args.tgt, args.speaker, args.file))
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
