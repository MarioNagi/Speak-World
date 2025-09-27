import numpy as np
from faster_whisper import WhisperModel

class ASR:
    def __init__(self, model_name="large-v3", device="auto", compute_type="float16"):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_16k: np.ndarray, lang=None):
        segments, info = self.model.transcribe(audio_16k, language=lang, vad_filter=False, beam_size=1)
        texts = [seg.text.strip() for seg in segments]
        return " ".join(t for t in texts if t)
