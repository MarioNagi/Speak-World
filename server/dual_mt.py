# server/dual_mt.py - Dual MT model manager for optimal translation
import logging
from typing import Optional, Dict, Any
from mt import MT
from language_utils import determine_best_mt_model

log = logging.getLogger("v2v.dual_mt")

class DualMTManager:
    """
    Manages both IndicTrans2 and M2M100 models and chooses the best one for each language pair.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.indictrans_model: Optional[MT] = None
        self.m2m100_model: Optional[MT] = None
        self.models_loaded = {"indictrans": False, "m2m100": False}
        
        # Try to load both models
        self._load_models()
        
    def _load_models(self):
        """Load both MT models."""
        
        # Load IndicTrans2 model
        try:
            log.info("Loading IndicTrans2 model...")
            indictrans_cfg = self.cfg.copy()
            indictrans_cfg["model"] = "ai4bharat/indictrans2-en-indic-1B"
            
            self.indictrans_model = MT(
                model_name=indictrans_cfg["model"],
                max_new_tokens=indictrans_cfg.get("max_new_tokens", 160),
                device=indictrans_cfg.get("device")
            )
            self.models_loaded["indictrans"] = True
            log.info("✅ IndicTrans2 model loaded successfully")
            
        except Exception as e:
            log.warning(f"❌ Failed to load IndicTrans2: {e}")
            log.info("Will use M2M100 for all translations")
        
        # Load M2M100 model
        try:
            log.info("Loading M2M100 model...")
            m2m100_cfg = self.cfg.copy()
            m2m100_cfg["model"] = "facebook/m2m100_418M"
            
            self.m2m100_model = MT(
                model_name=m2m100_cfg["model"],
                max_new_tokens=m2m100_cfg.get("max_new_tokens", 160),
                device=m2m100_cfg.get("device")
            )
            self.models_loaded["m2m100"] = True
            log.info("✅ M2M100 model loaded successfully")
            
        except Exception as e:
            log.error(f"❌ Failed to load M2M100: {e}")
            if not self.models_loaded["indictrans"]:
                raise RuntimeError("Both MT models failed to load!")
        
        # Log final status
        loaded_models = [k for k, v in self.models_loaded.items() if v]
        log.info(f"MT models available: {loaded_models}")
    
    def translate(self, text: str, src: str, tgt: str) -> str:
        """
        Translate text using the optimal model for the language pair.
        
        Args:
            text: Text to translate
            src: Source language code
            tgt: Target language code
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        
        # Determine best model for this language pair
        best_model = determine_best_mt_model(src, tgt)
        log.debug(f"Translation {src}->{tgt}: using {best_model} model")
        
        # Try preferred model first, fallback to available model
        if best_model == "indictrans" and self.models_loaded["indictrans"]:
            try:
                # Convert language codes to IndicTrans2 format
                src_code = self._convert_to_indictrans_code(src)
                tgt_code = self._convert_to_indictrans_code(tgt)
                return self.indictrans_model.translate(text, src_code, tgt_code)
            except Exception as e:
                log.warning(f"IndicTrans2 failed for {src}->{tgt}: {e}")
                if self.models_loaded["m2m100"]:
                    log.info("Falling back to M2M100")
                    return self.m2m100_model.translate(text, src, tgt)
                else:
                    return text
        
        elif best_model == "m2m100" and self.models_loaded["m2m100"]:
            try:
                return self.m2m100_model.translate(text, src, tgt)
            except Exception as e:
                log.warning(f"M2M100 failed for {src}->{tgt}: {e}")
                if self.models_loaded["indictrans"]:
                    log.info("Falling back to IndicTrans2")
                    src_code = self._convert_to_indictrans_code(src)
                    tgt_code = self._convert_to_indictrans_code(tgt)
                    return self.indictrans_model.translate(text, src_code, tgt_code)
                else:
                    return text
        
        # Fallback to any available model
        elif self.models_loaded["indictrans"]:
            try:
                src_code = self._convert_to_indictrans_code(src)
                tgt_code = self._convert_to_indictrans_code(tgt)
                return self.indictrans_model.translate(text, src_code, tgt_code)
            except:
                return text
                
        elif self.models_loaded["m2m100"]:
            try:
                return self.m2m100_model.translate(text, src, tgt)
            except:
                return text
        
        else:
            log.error("No MT models available!")
            return text
    
    def _convert_to_indictrans_code(self, lang_code: str) -> str:
        """Convert language code to IndicTrans2 format."""
        # Map common codes to IndicTrans2/FLORES format
        mapping = {
            "en": "en",
            "hi": "hi", 
            "ne": "ne",
            "bn": "bn",
            "ta": "ta",
            "te": "te",
            "mr": "mr",
            "gu": "gu",
            "pa": "pa",
            "ur": "ur",
            "as": "as",
            "or": "or", 
            "ml": "ml",
            "kn": "kn",
            "sa": "sa"
        }
        return mapping.get(lang_code.lower(), lang_code)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "models_loaded": self.models_loaded,
            "indictrans_available": self.models_loaded["indictrans"],
            "m2m100_available": self.models_loaded["m2m100"],
            "total_models": sum(self.models_loaded.values())
        }