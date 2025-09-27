# server/voice_bank.py
"""
Voice bank for managing speaker voices and enrollments.
"""
import os
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict
import hashlib
import time

log = logging.getLogger("v2v.voice_bank")

class VoiceBank:
    """Manages enrolled speaker voices."""
    
    def __init__(self, voices_dir: str = "voices"):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.voices_dir / "voices.json"
        self._speakers = self._load_metadata()
        
        log.info(f"VoiceBank initialized with {len(self._speakers)} voices")
    
    def _load_metadata(self) -> Dict:
        """Load voice metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                log.error(f"Failed to load voice metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save voice metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._speakers, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save voice metadata: {e}")
    
    def _generate_speaker_id(self, audio_path: str, name: Optional[str] = None) -> str:
        """Generate unique speaker ID from audio file."""
        # Use file content hash + timestamp for uniqueness
        with open(audio_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        timestamp = str(int(time.time()))[-6:]  # Last 6 digits
        
        if name:
            # Clean name for use in ID
            clean_name = ''.join(c for c in name if c.isalnum() or c in '-_').lower()[:10]
            return f"{clean_name}_{file_hash}_{timestamp}"
        else:
            return f"voice_{file_hash}_{timestamp}"
    
    def enroll(self, audio_path: str, name: Optional[str] = None) -> str:
        """
        Enroll a new speaker voice.
        
        Args:
            audio_path: Path to audio file containing speaker's voice
            name: Optional human-readable name for the speaker
            
        Returns:
            str: Unique speaker ID
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Generate unique speaker ID
        speaker_id = self._generate_speaker_id(audio_path, name)
        
        # Copy audio file to voices directory
        audio_ext = Path(audio_path).suffix.lower()
        dest_path = self.voices_dir / f"{speaker_id}{audio_ext}"
        
        try:
            shutil.copy2(audio_path, dest_path)
            
            # Store metadata
            self._speakers[speaker_id] = {
                "name": name or speaker_id,
                "audio_file": dest_path.name,
                "enrolled_at": time.time(),
                "file_size": dest_path.stat().st_size
            }
            
            self._save_metadata()
            log.info(f"Enrolled speaker '{speaker_id}' from {audio_path}")
            return speaker_id
            
        except Exception as e:
            log.error(f"Failed to enroll speaker: {e}")
            # Cleanup on failure
            if dest_path.exists():
                dest_path.unlink()
            raise
    
    def resolve(self, speaker_id: Optional[str]) -> Optional[str]:
        """
        Resolve speaker ID to audio file path.
        
        Args:
            speaker_id: Speaker ID to resolve (None for default/neutral voice)
            
        Returns:
            str: Path to speaker audio file, or None for neutral voice
        """
        if not speaker_id:
            return None
        
        if speaker_id not in self._speakers:
            log.warning(f"Speaker ID '{speaker_id}' not found")
            return None
        
        audio_file = self._speakers[speaker_id]["audio_file"]
        full_path = self.voices_dir / audio_file
        
        if not full_path.exists():
            log.error(f"Speaker audio file missing: {full_path}")
            return None
        
        return str(full_path)
    
    def list_speakers(self) -> Dict:
        """Get list of all enrolled speakers."""
        return {
            speaker_id: {
                "name": data["name"],
                "enrolled_at": data["enrolled_at"]
            }
            for speaker_id, data in self._speakers.items()
        }
    
    def remove_speaker(self, speaker_id: str) -> bool:
        """Remove an enrolled speaker."""
        if speaker_id not in self._speakers:
            log.warning(f"Speaker ID '{speaker_id}' not found for removal")
            return False
        
        try:
            # Remove audio file
            audio_file = self._speakers[speaker_id]["audio_file"]
            audio_path = self.voices_dir / audio_file
            if audio_path.exists():
                audio_path.unlink()
            
            # Remove from metadata
            del self._speakers[speaker_id]
            self._save_metadata()
            
            log.info(f"Removed speaker '{speaker_id}'")
            return True
            
        except Exception as e:
            log.error(f"Failed to remove speaker '{speaker_id}': {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get voice bank statistics."""
        total_size = sum(
            (self.voices_dir / data["audio_file"]).stat().st_size
            for data in self._speakers.values()
            if (self.voices_dir / data["audio_file"]).exists()
        )
        
        return {
            "total_speakers": len(self._speakers),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "voices_dir": str(self.voices_dir.absolute())
        }