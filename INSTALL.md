# WhisperLiveKit Installation Guide

> **⚠️ WORK IN PROGRESS**  
> This documentation is currently under development. The installation instructions primarily focus on macOS as it's the only platform available for testing. Users on other operating systems may need to adapt certain steps.

WhisperLiveKit is a real-time speech transcription and speaker diarization toolkit built on Whisper models with a FastAPI backend. This guide will help you set up and run the application.

## System Requirements

- Python 3.9 or higher
- FFmpeg (for audio processing)
- NVIDIA GPU (optional, for faster-whisper backend)
- Apple Silicon (recommended for mlx-whisper backend)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/jeanjerome/WhisperLiveKit.git
cd WhisperLiveKit
```

### 2. Set Up Environment

Using conda (recommended):

```bash
conda create -n whisperlivekit python=3.10 -y
conda activate whisperlivekit
```

### 3. Install System Dependencies

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

### 4. Install Python Dependencies

Core requirements:
```bash
pip install fastapi ffmpeg-python pydantic-settings uvicorn websockets
pip install librosa soundfile
```

Install at least one Whisper backend:
```bash
# For Apple Silicon (recommended for M1/M2/M3 Macs)
pip install mlx-whisper

# OR standard Whisper
pip install whisper

# OR faster-whisper (NVIDIA GPU)
pip install faster-whisper
```

Optional dependencies for advanced features:
```bash
# Voice Activity Controller (VAC) - prevents hallucinations
pip install torch

# Buffer trimming with sentence segmentation
pip install mosestokenizer wtpsplit
```

### 5. Diarization Setup (Optional)

If you want to use speaker diarization, you'll need to set up access to pyannote.audio models:

1. Create an account on [Hugging Face](https://huggingface.co)
2. Visit [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization) and [pyannote/segmentation](https://huggingface.co/pyannote/segmentation) and accept the license terms
3. Log in via the CLI:

```bash
huggingface-cli login
```

For macOS users with Apple Silicon, you may need additional setup:
```bash
xcode-select --install
brew install gcc llvm meson

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export LDFLAGS="-L$(brew --prefix llvm)/lib"
export CPPFLAGS="-I$(brew --prefix llvm)/include"
```

```bash
# Speaker diarization
pip install diart
```

## Running the Application

The application is started using the `main.py` script with various configuration options:

```bash
python main.py --host 127.0.0.1 --port 8000 --transcription True --diarization True \
  --model medium --model_cache_dir .model_cache_dir --lan fr \
  --task transcribe --backend mlx-whisper
```

### Common Configuration Options

- `--host`: Server host address (default: localhost)
- `--port`: Server port (default: 8000)
- `--transcription`: Enable speech transcription (default: True)
- `--diarization`: Enable speaker diarization (default: False)
- `--model`: Whisper model size (tiny, small, medium, large, etc.)
- `--model_cache_dir`: Directory for cached models
- `--lan`: Source language code (e.g., en, fr, de) or 'auto' for detection
- `--task`: Either 'transcribe' or 'translate'
- `--backend`: Choose between 'faster-whisper', 'whisper_timestamped', 'mlx-whisper', or 'openai-api'

### Accessing the Application

Once running, access the application in your browser:

```
http://127.0.0.1:8000
```

## Troubleshooting

- **VAD/VAC Issues**: Some combinations of voice activity detection options may not work with all backends. If experiencing issues, try disabling these features.
- **Diarization Not Working**: Ensure you've properly set up access to pyannote models through Hugging Face.
- **Performance Issues**: If transcription is slow, try a smaller model or a different backend.
- **CPU/Memory Usage**: For resource-constrained systems, use 'tiny' or 'small' models and avoid enabling both diarization and transcription simultaneously.

## License

This project is licensed under the MIT License - see the LICENSE file for details.