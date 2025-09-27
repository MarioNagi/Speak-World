# server/monitoring.py
"""
Monitoring and metrics collection for V2V translation system.
"""
import time
import logging
from collections import defaultdict, deque
from typing import Dict, List
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

log = logging.getLogger("v2v.monitoring")

@dataclass
class ProcessingMetrics:
    """Metrics for a single processing operation."""
    timestamp: float
    operation: str  # "asr", "mt", "tts", "full_pipeline"
    duration_ms: float
    success: bool
    error_type: str = None
    audio_duration_ms: float = None
    text_length: int = None

class MetricsCollector:
    """Collects and aggregates processing metrics."""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self._metrics = deque(maxlen=max_history)
        self._error_counts = defaultdict(int)
        self._session_count = 0
        self._lock = threading.Lock()
    
    def record_processing(self, operation: str, duration_ms: float, success: bool, 
                         error_type: str = None, audio_duration_ms: float = None, 
                         text_length: int = None):
        """Record a processing operation metric."""
        metric = ProcessingMetrics(
            timestamp=time.time(),
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            audio_duration_ms=audio_duration_ms,
            text_length=text_length
        )
        
        with self._lock:
            self._metrics.append(metric)
            if error_type:
                self._error_counts[f"{operation}_{error_type}"] += 1
    
    def record_session_start(self):
        """Record a new session starting."""
        with self._lock:
            self._session_count += 1
    
    def get_stats(self, window_minutes: int = 60) -> Dict:
        """Get aggregated statistics for the specified time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self._lock:
            recent_metrics = [m for m in self._metrics if m.timestamp > cutoff_time]
            
            stats = {
                "window_minutes": window_minutes,
                "total_operations": len(recent_metrics),
                "total_sessions": self._session_count,
                "operations": defaultdict(lambda: {"count": 0, "success": 0, "avg_duration_ms": 0, "errors": {}}),
                "overall": {
                    "success_rate": 0.0,
                    "avg_duration_ms": 0.0,
                    "total_errors": 0
                }
            }
            
            if not recent_metrics:
                return stats
            
            # Aggregate by operation type
            for metric in recent_metrics:
                op_stats = stats["operations"][metric.operation]
                op_stats["count"] += 1
                
                if metric.success:
                    op_stats["success"] += 1
                else:
                    error_key = metric.error_type or "unknown"
                    op_stats["errors"][error_key] = op_stats["errors"].get(error_key, 0) + 1
                
                # Update running average duration
                prev_avg = op_stats["avg_duration_ms"]
                prev_count = op_stats["count"] - 1
                op_stats["avg_duration_ms"] = (prev_avg * prev_count + metric.duration_ms) / op_stats["count"]
            
            # Calculate overall stats
            total_success = sum(op["success"] for op in stats["operations"].values())
            total_ops = len(recent_metrics)
            total_duration = sum(m.duration_ms for m in recent_metrics)
            total_errors = sum(sum(op["errors"].values()) for op in stats["operations"].values())
            
            stats["overall"] = {
                "success_rate": (total_success / total_ops * 100) if total_ops > 0 else 0.0,
                "avg_duration_ms": total_duration / total_ops if total_ops > 0 else 0.0,
                "total_errors": total_errors
            }
            
            return dict(stats)  # Convert defaultdict to regular dict
    
    def get_error_summary(self) -> Dict:
        """Get summary of all recorded errors."""
        with self._lock:
            return dict(self._error_counts)

# Global metrics collector instance
metrics = MetricsCollector()

class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, operation: str, audio_duration_ms: float = None, text_length: int = None):
        self.operation = operation
        self.audio_duration_ms = audio_duration_ms
        self.text_length = text_length
        self.start_time = None
        self.success = False
        self.error_type = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            self.success = True
        else:
            self.success = False
            self.error_type = exc_type.__name__ if exc_type else "unknown"
        
        metrics.record_processing(
            operation=self.operation,
            duration_ms=duration_ms,
            success=self.success,
            error_type=self.error_type,
            audio_duration_ms=self.audio_duration_ms,
            text_length=self.text_length
        )
        
        if self.success:
            log.debug(f"{self.operation} completed in {duration_ms:.1f}ms")
        else:
            log.error(f"{self.operation} failed in {duration_ms:.1f}ms: {self.error_type}")

def log_system_resources():
    """Log current system resource usage."""
    try:
        import psutil
        process = psutil.Process()
        
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        log.info(f"System resources - CPU: {cpu_percent}%, Memory: {memory_info.rss / 1024 / 1024:.1f}MB")
        
        # Log GPU memory if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                log.info(f"GPU {i} - Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
        except ImportError:
            pass  # GPU monitoring not available
            
    except ImportError:
        pass  # psutil not available

# Health check functions
def check_model_health(asr_model, mt_model, tts_model) -> Dict[str, str]:
    """Check health of all models."""
    health = {}
    
    try:
        # Test ASR with dummy data
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = asr_model.transcribe(dummy_audio, lang="en")
        health["asr"] = "healthy"
    except Exception as e:
        health["asr"] = f"unhealthy: {str(e)}"
    
    try:
        # Test MT with dummy text
        result = mt_model.translate("Hello", src="en", tgt="es")
        health["mt"] = "healthy"
    except Exception as e:
        health["mt"] = f"unhealthy: {str(e)}"
    
    try:
        # Test TTS with dummy text
        wav, sr = tts_model.synth("Hello", lang="en", speaker_wav=None)
        health["tts"] = "healthy"
    except Exception as e:
        health["tts"] = f"unhealthy: {str(e)}"
    
    return health
