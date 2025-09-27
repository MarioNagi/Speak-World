#!/usr/bin/env python3
# run.py - Startup script for V2V translation server
import os
import sys
import logging
import uvicorn
from pathlib import Path

# Add server directory to path so imports work
server_dir = Path(__file__).parent / "server"
sys.path.insert(0, str(server_dir))

def setup_logging():
    """Setup logging before importing the app."""
    log_level = os.environ.get("V2V_LOG_LEVEL", "INFO").upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

def main():
    """Main startup function."""
    setup_logging()
    log = logging.getLogger("v2v.startup")
    
    # Configuration
    host = os.environ.get("V2V_HOST", "0.0.0.0")
    port = int(os.environ.get("V2V_PORT", "8000"))
    workers = int(os.environ.get("V2V_WORKERS", "1"))  # Keep at 1 for ML models
    reload = os.environ.get("V2V_RELOAD", "false").lower() == "true"
    
    log.info(f"Starting V2V Translation Server on {host}:{port}")
    log.info(f"Workers: {workers}, Reload: {reload}")
    
    # Check if config exists
    config_paths = [Path("server/config.yaml"), Path("./config.yaml")]
    config_found = any(p.exists() for p in config_paths)
    
    if not config_found:
        log.error("config.yaml not found! Please create it first.")
        log.error(f"Looked in: {[str(p) for p in config_paths]}")
        sys.exit(1)
    
    # Create necessary directories
    Path("voices").mkdir(exist_ok=True)
    
    try:
        # Import app after path setup
        from app import app
        
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=log_level.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        log.info("Server stopped by user")
    except Exception as e:
        log.exception(f"Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
