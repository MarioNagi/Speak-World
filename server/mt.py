# server/mt.py - Fixed for AI4Bharat models
from __future__ import annotations
import importlib, re, os
from typing import List, Optional
import torch
import logging

log = logging.getLogger("v2v.mt")

# FLORES-200 codes (extend as needed)
FLORES = {
    "en": "eng_Latn",
    "hi": "hin_Deva", 
    "ne": "npi_Deva",   # Nepali
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu", 
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "as": "asm_Beng",   # Assamese
    "or": "ory_Orya",   # Odia
    "ml": "mal_Mlym",   # Malayalam
    "kn": "kan_Knda",   # Kannada
    "sa": "san_Deva",   # Sanskrit
}

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _split_sents(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r"([\.!\?।॥])", text)  # Added Devanagari punctuation
    if len(parts) == 1:
        return [text.strip()] if text.strip() else []
    out, cur = [], []
    for p in parts:
        if p is None: 
            continue
        cur.append(p)
        if p in (".", "!", "?", "।", "॥"):  # Include Devanagari punctuation
            s = "".join(cur).strip()
            if s: 
                out.append(s)
            cur = []
    tail = "".join(cur).strip()
    if tail: 
        out.append(tail)
    return out

class MT:
    def __init__(self, model_name="facebook/m2m100_418M", max_new_tokens=160, device: Optional[str]=None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or _device()
        
        log.info(f"Initializing MT model: {model_name} on {self.device}")

        tr = importlib.import_module("transformers")
        
        if "ai4bharat/indictrans2" in model_name.lower():
            # ---------- IndicTrans2 path ----------
            self.kind = "indictrans2"
            log.info("Loading IndicTrans2 model...")
            
            try:
                # Try without local_files_only first
                self.tokenizer = tr.AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    # Remove local_files_only=True to allow downloading
                )
                self.model = tr.AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    # Remove local_files_only=True to allow downloading
                ).to(self.device).eval()
                
            except Exception as e:
                log.exception("Failed to load IndicTrans2 model")
                raise RuntimeError(f"IndicTrans2 model loading failed: {e}")
                
        else:
            # ---------- M2M100 path ----------
            self.kind = "m2m100"
            log.info("Loading M2M100 model...")
            
            try:
                # Check for required dependencies
                importlib.import_module("sentencepiece")
                
                tok_cls = getattr(tr, "M2M100TokenizerFast", None) or getattr(tr, "M2M100Tokenizer", None)
                if tok_cls is None:
                    raise RuntimeError("Transformers missing M2M100Tokenizer{Fast}. pip install 'transformers>=4.52.1,<5'")
                    
                model_cls = getattr(tr, "M2M100ForConditionalGeneration", None)
                if model_cls is None:
                    raise RuntimeError("Transformers missing M2M100ForConditionalGeneration.")
                
                self.tokenizer = tok_cls.from_pretrained(model_name)
                self.model = model_cls.from_pretrained(model_name).to(self.device).eval()
                
            except Exception as e:
                log.exception("Failed to load M2M100 model")
                raise RuntimeError(f"M2M100 model loading failed: {e}")
        
        log.info(f"MT model loaded successfully: {self.kind}")

    def translate(self, text: str, src: str, tgt: str) -> str:
        """Translate text from source to target language."""
        if not text or not text.strip():
            return ""

        log.debug(f"Translating: '{text[:50]}...' from {src} to {tgt}")

        if self.kind == "indictrans2":
            return self._translate_indictrans2(text, src, tgt)
        else:
            return self._translate_m2m100(text, src, tgt)

    def _translate_indictrans2(self, text: str, src: str, tgt: str) -> str:
        """Translate using IndicTrans2 model."""
        # Validate language codes
        if src not in FLORES or tgt not in FLORES:
            available = list(FLORES.keys())
            raise ValueError(f"IndicTrans2 expects FLORES languages. Got src={src}, tgt={tgt}. "
                           f"Available: {available}")
        
        src_tag, tgt_tag = FLORES[src], FLORES[tgt]
        sents = _split_sents(text) or [text.strip()]
        
        # Build the required "<SRC> <TGT> <sentence>" format
        inputs = [f"{src_tag} {tgt_tag} {s}" for s in sents]
        
        try:
            tok = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=256,  # Prevent very long inputs
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            with torch.no_grad():
                gen = self.model.generate(
                    **tok,
                    use_cache=True,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,  # Faster inference
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            outs = self.tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # Clean up extra spaces
            outs = [o.replace("  ", " ").strip() for o in outs]
            result = " ".join([o for o in outs if o]).strip()
            
            log.debug(f"IndicTrans2 result: '{result[:50]}...'")
            return result
            
        except Exception as e:
            log.exception("IndicTrans2 translation failed")
            return text  # Fallback to original text

    def _translate_m2m100(self, text: str, src: str, tgt: str) -> str:
        """Translate using M2M100 model."""
        sents = _split_sents(text) or [text.strip()]
        tok = self.tokenizer
        tok.src_lang = src
        outs = []
        
        try:
            for s in sents:
                enc = tok(s, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                with torch.no_grad():
                    gen = self.model.generate(
                        **enc,
                        forced_bos_token_id=tok.get_lang_id(tgt),
                        max_new_tokens=self.max_new_tokens,
                        num_beams=1,
                        do_sample=False,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                    )
                outs.append(tok.batch_decode(gen, skip_special_tokens=True)[0].strip())
            
            result = " ".join([o for o in outs if o]).strip()
            log.debug(f"M2M100 result: '{result[:50]}...'")
            return result
            
        except Exception as e:
            log.exception("M2M100 translation failed")
            return text  # Fallback to original text