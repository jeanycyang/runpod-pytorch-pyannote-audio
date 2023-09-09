FROM runpod/pytorch:3.10-2.0.1-120-devel

RUN apt-get update -y && \
    git clone --depth=1 https://github.com/pyannote/pyannote-audio.git /workspace/pyannote-audio && cd /workspace/pyannote-audio && \
    pip install -e . && \
    pip install gdown && apt-get install -y vim

COPY sd.py /workspace/pyannote-audio

WORKDIR /workspace/pyannote-audio
