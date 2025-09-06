# transcribe

## Installation

### Linux

```shell
sudo apt install ffmpeg

# this fails may need to do:
sudo apt update
sudo apt install ffmpeg --fix-missing
```

### Mac

```shell
brew install ffmpeg
```

Now install requirements in a virtualenv

```shell
virtualenv -p python3 venv/transcribe
source venv/transcribe/bin/activate
python3 -m pip install -r requirements.txt
```

### Hugging Face Token

for the `transcribe_pyannote.py` script, `pyannote.audio` diarization requires access to pretrained models on HuggingFace.

Get a free token at https://huggingface.co/settings/tokens and then make it read only access, then search for pyannote/speaker-diarization model and accept the license.

## Usage

```shell

```


