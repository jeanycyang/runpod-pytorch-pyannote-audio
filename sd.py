import os
import sys
import logging
from collections import defaultdict
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from pyannote.audio import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    logger = logging.getLogger('speaker_diarization')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: ", device)

    hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in environment variables.")

    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    sr = 44100 # sample rate

    if not input_path or not output_dir:
        raise ValueError("input_path and output_dir must be given.")

    try:
        audio, sr = librosa.load(input_path, sr=sr, mono=True)
    except Exception as e:
        logger.error(f"Failed to read {input_path}: {e}")
        raise e

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token=hf_token)
    if pipeline is None:
        raise ValueError("Failed to load pipeline")

    logger.info(f"Processing {input_path}. This may take a while...")
    diarization = pipeline(
        input_path
    )

    logger.info(f"Found {len(diarization)} tracks, writing to {output_dir}")
    speaker_count = defaultdict(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    for segment, track, speaker in tqdm(
        list(diarization.itertracks(yield_label=True)), desc=f"Writing {input_path}"
    ):
        if segment.end - segment.start < 1:
            continue
        speaker_count[speaker] += 1
        audio_cut = audio[int(segment.start * sr) : int(segment.end * sr)]
        sf.write(
            (output_dir / f"{speaker}_{speaker_count[speaker]}.wav"),
            audio_cut,
            sr,
        )

    logger.info(f"Speaker count: {speaker_count}")

if __name__ == '__main__':
    main()
