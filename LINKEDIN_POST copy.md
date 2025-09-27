Announcing my latest project: real-time voice-to-voice translation

I built a local, privacy-first voice-to-voice translator that captures speech from a microphone, runs streaming ASR, translates the text with a dual MT strategy (IndicTrans2 for Indic pairs + M2M100 otherwise), and returns synthesized speech — all in near real-time.

Why this matters
- Real-time speech translation removes language barriers in conversations.
- Local-first design keeps audio and transcripts private.
- Dual MT approach lets us use the best model for each language pair.

Tech highlights
- FastAPI + WebSockets for low-latency streaming
- VAD-based chunking, streaming ASR, MT, and TTS pipeline
- Browser UI captures mic audio and streams 16 kHz PCM16 frames to the server
- CLI mic client for automated testing and demos

Try it
- Clone the repo: git clone <repo>
- Start the server: python server/run_script.py
- Open the web UI, create a session, click the mic, and speak. The system will translate your speech and play the synthesized audio back.

Open source — contributions welcome
- See README.md for setup instructions. If you want to try a demo or need help running this locally, DM me!

#speech #opensource #machinelearning #nlp #audiotech #translation #fastapi #python