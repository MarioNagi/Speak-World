Speak World — Real-timeThis project is a near real-time voice-to-voice translation system with a Web UI and a mic client. It runs locally or in Docker and combines: an ASR component (speech→text), a Dual-MT layer (preferentially uses IndicTrans2 for Indic languages and M2M100 for other pairs), a TTS engine (text→speech), a VAD for chunking microphone audio, and a small web UI for interactive sessions.

**Performance**: End-to-end latency is approximately 6-7 seconds from speech input to translated audio output.

For decades, universal languages like Esperanto promised a single tongue for everyone — but they never truly took off.
With AI, we don't need a single universal language anymore. You can speak your own language and still be understood worldwide.

Speak World makes that possible, in near real-time.h-to-speech translation
=================================================

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/YOUR_REPO_NAME)](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/LICENSE)

> ⭐ **Star this repo** if you find it useful!

Table of Contents
-----------------
- [Overview](#overview)
- [Demo video](#demo-video)
- [Quick highlights](#quick-highlights)
- [Repository layout](#repository-layout)
- [Supported endpoints](#supported-endpoints)
- [Audio format and expectations](#audio-format-and-expectations)
- [Quickstart — Local](#quickstart---local-recommended-for-development)
- [Quickstart — Docker](#quickstart---docker-containerized)
- [Troubleshooting tips](#troubleshooting-tips)
- [Development notes](#development-notes)
- [Future improvements](#future-improvements)
- [Contributing](#contributing)
- [How to add a language](#how-to-add-a-language)
- [License](#license)
- [Contact / Demo](#contact--demo)

Overview
--------
This project is a near real-time voice-to-voice translation system with a Web UI and a mic client. It runs locally or in Docker and combines: an ASR component (speech→text), a Dual-MT layer (preferentially uses IndicTrans2 for Indic languages and M2M100 for other pairs), a TTS engine (text→speech), a VAD for chunking microphone audio, and a small web UI for interactive sessions.

For decades, universal languages like Esperanto promised a single tongue for everyone — but they never truly took off.
With AI, we don’t need a single universal language anymore. You can speak your own language and still be understood worldwide.

Speak World makes that possible, in near real-time.

Demo video
----------
Play the short demo to see two-directional translation and example UI flows.

 


https://github.com/user-attachments/assets/e8fd0658-dc4b-4a07-ba64-26bc2ac738ff


 

Quick highlights
- Near Real-time, bidirectional voice translation sessions
- Web UI for browser-based interaction and microphone capture
- Mic client CLI for automated or local testing
- Dual MT: chooses the best model per language pair
- Streaming WebSocket endpoints for audio and text exchange

Why this matters / Use cases
---------------------------
- Ease cross-border communication: travelers can speak to locals in their native language and get near-instant translations back, making navigation and daily interactions simpler.
- Bridge under-served languages: this project focuses on including under-served languages (for example, Bihari languages) that often receive less tooling and research attention. The Dual-MT strategy lets you plug in specialized models for those languages so communities are better represented.
- Privacy-first demos & self-hosting: the system can run on a dedicated server so audio and transcripts remain under your control.


Repository layout
-----------------
- client/
  - mic_client.py      # CLI mic client that streams audio to server
  - requirements.txt
- server/
  - app.py             # FastAPI server with WebSocket endpoints
  - app-server.py      # (alternate server entry; project contains variants)
  - async_models.py    # Async wrappers for ASR/MT/TTS
  - mt.py              # MT helpers and model loader
  - dual_mt.py         # Manages IndicTrans2 + M2M100
  - asr.py             # ASR wrapper
  - tts.py             # TTS engine
  - vad.py             # Voice Activity Detector / chunker
  - voice_bank.py      # Speaker enrollment & voice bank
  - templates/         # Web UI templates (index.html)
  - static/            # Static assets/css/js
  - requirements.txt
- utils/               # Audio helpers and small utilities
- voices/              # Enrollment storage (audio files)
- docker-compose.yml
- run_script.py        # Helper to start the server

Supported endpoints
-------------------
- HTTP
  - GET  /                → Web UI
  - GET  /languages/supported
  - POST /session/create  → Create a translation session (returns session_id)
- WebSocket
  - /ws/stream            → Legacy streaming endpoint (mic_client uses this)
  - /ws/translate/{id}    → Enhanced bidirectional translation endpoint (web UI)

Audio format and expectations
-----------------------------
- Client → Server (binary frames): 16 kHz, PCM16 (signed 16-bit little-endian) frames.
- Server transforms incoming PCM16 bytes to float32, resamples/normalizes as needed, applies VAD and pipeline (ASR→MT→TTS).
- Server → Client (audio responses): typically 24 kHz float32 PCM (or WAV-like binary). The web UI includes logic that expects raw PCM16 from server or decodes ArrayBuffer payloads into an AudioBuffer.

Quickstart — Local (recommended for development)
------------------------------------------------
1. Create Python virtual environments and install dependencies:

```bash
# Server
python -m venv .venv-server
.venv-server\Scripts\activate
pip install -r server/requirements.txt

# Client (optional)
python -m venv .venv-client
.venv-client\Scripts\activate
pip install -r client/requirements.txt
```

2. Start the server (development):

```bash
# From repository root
python server/run_script.py
# or (alternative)
uvicorn server.app:app --host 0.0.0.0 --port 8000 --log-level info
```

3. Open the Web UI in a browser:

- http://localhost:8000/

Create a session (UI) and click the mic button to record. The UI will capture microphone audio, resample to 16 kHz, convert to PCM16 and stream it over WebSocket to the server.

4. CLI mic client (quick test):

```bash
# Example: stream microphone to server  (replace url/target as needed)
python client/mic_client.py --url ws://localhost:8000/ws/stream --tgt de
# To stream an audio file instead of the mic:
python client/mic_client.py --file samples/test_en.wav --url ws://localhost:8000/ws/stream --tgt de
```

Quickstart — Docker (containerized)
----------------------------------
1. Build and run with docker-compose:

```bash
docker-compose up --build
```

2. The container will expose ports according to `docker-compose.yml`. Open the web UI using the configured host/port.

Troubleshooting tips
--------------------
- Browser microphone permissions: ensure your browser allows microphone access for localhost.
- If the web UI shows "Microphone access required", check the browser console for errors and server logs for WebSocket connections.
- If no audio appears on the server, confirm the web client is sending binary frames and `ws.binaryType` is set to `arraybuffer` (the UI debug log prints what is sent).
- If translations come back in the wrong language, try specifying explicit target language when using the mic client or verify the session language pair in the web UI.

Development notes
-----------------
- `async_models.py` wraps blocking ML calls in a ThreadPoolExecutor to avoid blocking FastAPI's event loop.
- The server provides both a legacy `/ws/stream` endpoint (mic_client compatibility) and `/ws/translate/{session_id}` for the web UI session flow.
- Model loading can be slow on first run (downloading weights). Check logs for progress.

Future improvements
-------------------
Planned improvements and ideas to increase security, UX, and model quality:

1) Harden the app / authentication
 - Issue short-lived JWTs from `/session/create`; pass them as `?token=...` to the WS URL and verify on connect.
 - CORS allow-list trusted origins; rate-limit session creation & audio frames per minute.
 - Consider mTLS or API keys for trusted backends; TLS termination for all external endpoints.
 - Enforce per-session quotas/timeouts; deny oversized frames; sanitize text payloads.
 - Structured logging + audit trail; export Prometheus metrics (latency, real-time factor, TTS time).

2) Mobile apps
 - Wrap the same WS protocol in React Native or Flutter.
 - Offer an HTTP upload fallback for devices that restrict WS mic streaming.
 - Consider Opus encoding (20 ms frames) for lower bandwidth on cellular; keep PCM as default on Wi-Fi.

3) Quality & UX
 - Auto voice selection by target language (e.g., Egyptian voice for ar-EG).
 - Personal dictionary & phrasebook to bias MT.
 - VAD tuning for noisy streets; AGC/denoise toggle; input level meter.
 - Offline mode for on-device ASR/MT/TTS (edge devices) when feasible.

4) Modeling
 - Add beam search / num_beams > 1 presets for high-importance lines; keep greedy for low latency.
 - Increase max_new_tokens for longer sentences; guard with timeouts.
 - Plug-in architecture so each language pair can pick: Faster-Whisper size, IndicTrans vs NLLB, and TTS voice.

5) Optimization
 - Profile end-to-end latency and optimize hot paths (ASR/TTS model loading, audio IO, serialization).
 - Use batching and quantized models where appropriate; explore faster runtimes (ONNX, CTranslate2 optimizations).
 - Consider async pipelining to overlap ASR, MT and TTS work when safe to reduce total wall-clock time.

Contributing
------------
- Add new voices to `voices/` via the API endpoint `/voices/enroll` or extend `voice_bank.py`.
- To add language support, update `language_utils.py` and provide corresponding MT/TTS mappings.

How to add a language
---------------------
This mini-guide shows the typical steps to add support for a new language pair and tie it into the pipeline.

1. Add language metadata
 - Edit `server/language_utils.py` and add a new entry in the language configuration for the new language key. Include fields for `display`, `whisper` (ASR language code), `mt` (MT code), and `tts` (TTS code) if needed.

2. Register MT/TTS models
 - Update `server/mt.py` or `server/dual_mt.py` mappings so the new language pair maps to the preferred MT backend (IndicTrans2, M2M100, etc.).
 - Add a TTS voice mapping in `server/tts.py` (or the TTS config) for the target language and test the voice locally.

3. Update frontend/templating
 - Ensure `server/templates/index.html` (or any client UI) exposes the new language key in language dropdowns. The server-side templates are populated from `language_utils`.

4. Add test audio / sample phrases
 - Add short audio samples to `video-demo/` or a `samples/` folder and run `client/mic_client.py --file samples/test_<lang>.wav` to validate the ASR→MT→TTS roundtrip.

5. Validate and tune
 - Run a few sessions and evaluate ASR accuracy, MT adequacy, and TTS naturalness. Consider adding phrasebook entries or small fine-tuning if quality is low.

6. Document and contribute
 - Update README (or a language support document) describing any special model weights, tokenizers, or credentials needed. Submit a PR with the language entry and tests.

License
-------
MIT License — feel free to reuse and adapt with attribution.

Contact / Demo
---------------
If you'd like help testing a demo or deploying the project, reply here with your environment details (OS, GPU availability) and I can provide instructions.
 
