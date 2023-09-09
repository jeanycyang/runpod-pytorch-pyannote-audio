FROM runpod/pytorch:3.10-2.0.1-118-runtime

RUN apt-get update -y && \
    pip install https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip && \
    pip install gdown && apt-get install -y vim

COPY sd.py /workspace/

WORKDIR /workspace/
