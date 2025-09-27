# server/language_utils.py - Enhanced with intelligent model selection
"""
Enhanced language utilities with intelligent model selection.
"""
import logging
from typing import Dict, Tuple, List

log = logging.getLogger("v2v.language_utils")

# Language configurations with model preferences
LANGUAGE_CONFIG = {
    # Indic languages - best with IndicTrans2
    "english": {
        "codes": {"whisper": "en", "indictrans": "en", "m2m100": "en", "tts": "en"},
        "display": "English",
        "family": "indo_european",
        "preferred_mt": "indictrans",  # Works well with both models
    },
    "nepali": {
        "codes": {"whisper": "ne", "indictrans": "ne", "m2m100": "ne", "tts": "ne"},
        "display": "नेपाली (Nepali)",
        "family": "indic",
        "preferred_mt": "indictrans",
    },
    "hindi": {
        "codes": {"whisper": "hi", "indictrans": "hi", "m2m100": "hi", "tts": "hi"},
        "display": "हिन्दी (Hindi)",
        "family": "indic", 
        "preferred_mt": "indictrans",
    },
    "bengali": {
        "codes": {"whisper": "bn", "indictrans": "bn", "m2m100": "bn", "tts": "bn"},
        "display": "বাংলা (Bengali)",
        "family": "indic",
        "preferred_mt": "indictrans",
    },
    "tamil": {
        "codes": {"whisper": "ta", "indictrans": "ta", "m2m100": "ta", "tts": "ta"},
        "display": "தமிழ் (Tamil)",
        "family": "indic",
        "preferred_mt": "indictrans",
    },
    "gujarati": {
        "codes": {"whisper": "gu", "indictrans": "gu", "m2m100": "gu", "tts": "gu"},
        "display": "ગુજરાતી (Gujarati)",
        "family": "indic",
        "preferred_mt": "indictrans",
    },
    "marathi": {
        "codes": {"whisper": "mr", "indictrans": "mr", "m2m100": "mr", "tts": "mr"},
        "display": "मराठी (Marathi)",
        "family": "indic",
        "preferred_mt": "indictrans",
    },
    "punjabi": {
        "codes": {"whisper": "pa", "indictrans": "pa", "m2m100": "pa", "tts": "pa"},
        "display": "ਪੰਜਾਬੀ (Punjabi)",
        "family": "indic",
        "preferred_mt": "indictrans",
    },
    "urdu": {
        "codes": {"whisper": "ur", "indictrans": "ur", "m2m100": "ur", "tts": "ur"},
        "display": "اردو (Urdu)",
        "family": "indic",
        "preferred_mt": "indictrans",
    },
    
    # Non-Indic languages - better with M2M100
    "arabic": {
        "codes": {"whisper": "ar", "indictrans": "ar", "m2m100": "ar", "tts": "ar"},
        "display": "العربية (Arabic)",
        "family": "semitic",
        "preferred_mt": "m2m100",
    },
    "spanish": {
        "codes": {"whisper": "es", "indictrans": "es", "m2m100": "es", "tts": "es"},
        "display": "Español (Spanish)",
        "family": "romance",
        "preferred_mt": "m2m100",
    },
    "french": {
        "codes": {"whisper": "fr", "indictrans": "fr", "m2m100": "fr", "tts": "fr"},
        "display": "Français (French)",
        "family": "romance",
        "preferred_mt": "m2m100",
    },
    "german": {
        "codes": {"whisper": "de", "indictrans": "de", "m2m100": "de", "tts": "de"},
        "display": "Deutsch (German)",
        "family": "germanic",
        "preferred_mt": "m2m100",
    },
    "chinese": {
        "codes": {"whisper": "zh", "indictrans": "zh", "m2m100": "zh", "tts": "zh"},
        "display": "中文 (Chinese)",
        "family": "sino_tibetan",
        "preferred_mt": "m2m100",
    },
    "japanese": {
        "codes": {"whisper": "ja", "indictrans": "ja", "m2m100": "ja", "tts": "ja"},
        "display": "日本語 (Japanese)",
        "family": "japonic",
        "preferred_mt": "m2m100",
    },
    "korean": {
        "codes": {"whisper": "ko", "indictrans": "ko", "m2m100": "ko", "tts": "ko"},
        "display": "한국어 (Korean)",
        "family": "koreanic",
        "preferred_mt": "m2m100",
    },
    "portuguese": {
        "codes": {"whisper": "pt", "indictrans": "pt", "m2m100": "pt", "tts": "pt"},
        "display": "Português (Portuguese)",
        "family": "romance",
        "preferred_mt": "m2m100",
    },
    "russian": {
        "codes": {"whisper": "ru", "indictrans": "ru", "m2m100": "ru", "tts": "ru"},
        "display": "Русский (Russian)",
        "family": "slavic",
        "preferred_mt": "m2m100",
    },
    "italian": {
        "codes": {"whisper": "it", "indictrans": "it", "m2m100": "it", "tts": "it"},
        "display": "Italiano (Italian)",
        "family": "romance",
        "preferred_mt": "m2m100",
    },
}

def determine_best_mt_model(src_lang: str, tgt_lang: str) -> str:
    """
    Determine the best MT model for a language pair.
    
    Args:
        src_lang: Source language key
        tgt_lang: Target language key
        
    Returns:
        str: "indictrans" or "m2m100"
    """
    src_config = LANGUAGE_CONFIG.get(src_lang.lower(), {})
    tgt_config = LANGUAGE_CONFIG.get(tgt_lang.lower(), {})
    
    src_family = src_config.get("family", "unknown")
    tgt_family = tgt_config.get("family", "unknown")
    src_preferred = src_config.get("preferred_mt", "m2m100")
    tgt_preferred = tgt_config.get("preferred_mt", "m2m100")
    
    # Logic for model selection:
    # 1. If both languages prefer IndicTrans2, use it
    # 2. If one is Indic and the other is English, use IndicTrans2
    # 3. Otherwise use M2M100
    
    if src_preferred == "indictrans" and tgt_preferred == "indictrans":
        return "indictrans"
    elif (src_family == "indic" and tgt_lang.lower() == "english") or \
         (src_lang.lower() == "english" and tgt_family == "indic"):
        return "indictrans"
    else:
        return "m2m100"

def normalize_language_code(lang_input: str, mt_model: str = "auto") -> Dict:
    """
    Enhanced language normalization with model-specific codes.
    
    Args:
        lang_input: Language input (name or code)
        mt_model: "indictrans", "m2m100", or "auto"
    
    Returns:
        Dict with language codes for each model
    """
    if not lang_input:
        raise ValueError("Language input cannot be empty")
        
    if lang_input.lower() == "auto":
        return {
            "whisper": None,
            "mt": "en",
            "tts": "en",
            "display": "Auto-detect",
            "family": "auto",
            "preferred_mt": "m2m100"
        }
    
    # Normalize input
    lang_key = lang_input.lower().strip()
    
    # Find language config
    lang_config = None
    
    # Try direct match first
    if lang_key in LANGUAGE_CONFIG:
        lang_config = LANGUAGE_CONFIG[lang_key]
    else:
        # Try code matching
        for name, config in LANGUAGE_CONFIG.items():
            codes = config["codes"]
            if lang_key in codes.values():
                lang_config = config
                break
    
    if not lang_config:
        log.warning(f"Unknown language '{lang_input}', using fallback")
        return {
            "whisper": lang_key,
            "mt": lang_key,
            "tts": lang_key,
            "display": lang_key.title(),
            "family": "unknown",
            "preferred_mt": "m2m100"
        }
    
    # Select MT code based on model preference
    if mt_model == "indictrans":
        mt_code = lang_config["codes"]["indictrans"]
    elif mt_model == "m2m100":
        mt_code = lang_config["codes"]["m2m100"]
    else:  # auto
        mt_code = lang_config["codes"][lang_config["preferred_mt"]]
    
    return {
        "whisper": lang_config["codes"]["whisper"],
        "mt": mt_code,
        "tts": lang_config["codes"]["tts"],
        "display": lang_config["display"],
        "family": lang_config["family"],
        "preferred_mt": lang_config["preferred_mt"]
    }

def validate_language_pair(src_lang: str, tgt_lang: str) -> Tuple[Dict, Dict, str]:
    """
    Validate and normalize a language pair with optimal model selection.
    
    Returns:
        Tuple: (src_codes, tgt_codes, recommended_mt_model)
    """
    try:
        # Determine best MT model for this pair
        best_mt_model = determine_best_mt_model(src_lang, tgt_lang)
        
        # Normalize with the selected model
        src_codes = normalize_language_code(src_lang, best_mt_model)
        tgt_codes = normalize_language_code(tgt_lang, best_mt_model)
        
        log.info(f"Language pair: {src_codes['display']} -> {tgt_codes['display']}")
        log.info(f"Recommended MT model: {best_mt_model}")
        
        return src_codes, tgt_codes, best_mt_model
        
    except Exception as e:
        raise ValueError(f"Invalid language pair: src={src_lang}, tgt={tgt_lang}: {e}")

def get_supported_languages() -> List[Dict]:
    """Get list of supported languages with metadata."""
    return [
        {
            "key": key,
            "display": config["display"],
            "family": config["family"],
            "preferred_mt": config["preferred_mt"]
        }
        for key, config in LANGUAGE_CONFIG.items()
    ]

def get_language_families() -> Dict[str, List[str]]:
    """Group languages by family for UI organization."""
    families = {}
    for key, config in LANGUAGE_CONFIG.items():
        family = config["family"]
        if family not in families:
            families[family] = []
        families[family].append({
            "key": key,
            "display": config["display"]
        })
    return families

def is_bidirectional_supported(lang1: str, lang2: str) -> bool:
    """Check if bidirectional translation is supported for a language pair."""
    try:
        # Try both directions
        validate_language_pair(lang1, lang2)
        validate_language_pair(lang2, lang1)
        return True
    except:
        return False