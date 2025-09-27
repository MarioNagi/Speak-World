FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# Torch with CUDA 12.1 wheels
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cu121 \    torch==2.3.1+cu121 torchaudio==2.3.1+cu121

WORKDIR /workspace
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Note: requirements.txt now includes jinja2, python-multipart and aiofiles
# used by the web UI (templating and file uploads). If you have extra
# environment-specific packages (e.g. OS-level libs) add them here.

COPY . .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
