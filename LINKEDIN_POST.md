Announcing my latest project: Speak World — real-time voice-to-voice translation

For decades, universal languages like Esperanto promised a single tongue for everyone — but they never truly took off.
With AI, we don’t need a single universal language anymore. You can speak your own language and still be understood worldwide.

Speak World makes that possible, in near real-time.

I built a local, privacy-first voice-to-voice translator that captures speech from a microphone, runs streaming ASR, translates the text with a dual MT strategy (IndicTrans2 for Indic pairs + M2M100 otherwise), and returns synthesized speech — all in near real-time.

I recorded a short demo showing two-way translation in action (link in comments). I tested the system on a GPU instance (H200 via Vast.ai) for faster model loads and used an SSH tunnel to forward the web UI during live demos.

Why this matters
- Real-time speech translation removes language barriers in everyday conversations.
- This project focuses on including under-served languages (for example, Bihari languages) — communities that often receive less tooling and research attention. I hope this helps make translation tech more equitable and inclusive.

Tech highlights
- FastAPI + WebSockets for low-latency streaming
- VAD-based chunking, streaming ASR, MT, and TTS pipeline
- Browser UI captures mic audio and streams 16 kHz PCM16 frames to the server
- CLI mic client for automated testing and demos

Try it
- Clone the repo: git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME
- Start the server: python server/run_script.py
- Open the web UI, create a session, click the mic, and speak. The system will translate your speech and play the synthesized audio back.

Open source — contributions welcome
- See README.md for setup instructions and deployment notes (I include a short note about testing on Vast.ai). If you want to try a demo or need help running this locally, DM me!
- ⭐ Star the repo: https://github.com/YOUR_USERNAME/YOUR_REPO_NAME

#ai #artificialintelligence #machinelearning #deeplearning #nlp #python #fastapi #opensource #speechrecognition #voicetechnology #translation #realtime #innovation #tech #programming #developer #linguistics #multilingual #accessibility #techforgood #buildinpublic #democratizingai #Biharilanguages #inclusivetech #digitaldivide