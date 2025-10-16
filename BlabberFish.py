import argparse
import json
import sys
import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment

def get_device():
    """Determine the device (CPU or CUDA) and log the selection."""
    if os.environ.get('FORCE_CPU', '0') == '1':
        device = torch.device("cpu")
        print("DEBUG: Environment variable FORCE_CPU=1 found. Using CPU.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("DEBUG: CUDA found. Using GPU with memory-safe settings.")
    else:
        device = torch.device("cpu")
        print("DEBUG: CUDA not available. Using CPU.")
    return device

def run_diarization(audio_path: str, pyannote_path: str, device: torch.device):
    """Run Pyannote Diarization (Who spoke when)."""
    print(f"\n--- 1. Running Pyannote Diarization on {audio_path} ---")
    
    if device.type == 'cuda' and os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING', '0') != '1':
        print("WARNING: Consider running with PYTORCH_NO_CUDA_MEMORY_CACHING=1 for stability.")

    try:
        pipeline = Pipeline.from_pretrained(
            pyannote_path,
            use_auth_token=False
        ).to(device)
        
        diarization = pipeline(audio_path)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"FATAL: Diarization failed. Error: {e}")
        sys.exit(1)
        
    print("Diarization complete.")
    return diarization

def run_transcription(audio_path: str, diarization, whisper_model_name: str, device: torch.device):
    """Run Whisper Transcription, segment by Pyannote's output."""
    print(f"\n--- 2. Running Whisper Transcription (Model: {whisper_model_name}) ---")
    
    try:
        whisper_model = whisper.load_model(whisper_model_name, device=device)
    except Exception as e:
        print(f"FATAL: Whisper model failed to load. Error: {e}")
        sys.exit(1)
        
    audio_full = whisper.load_audio(audio_path)
    final_conversation = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        
        start_sample = int(turn.start * whisper.audio.SAMPLE_RATE)
        end_sample = int(turn.end * whisper.audio.SAMPLE_RATE)
        
        segment_audio = audio_full[start_sample:end_sample]
        
        try:
            result = whisper_model.transcribe(segment_audio, device=device)
            text = result["text"].strip()
        except Exception as e:
            text = f"[Transcription Error: {e}]"
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        if text: 
            final_conversation.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "text": text
            })

    print("Transcription and segment merging complete.")
    return final_conversation

def save_conversation(conversation, out_path):
    """Save the final conversation to JSONL."""
    print(f"\n--- 3. Saving Final Conversation to {out_path} ---")
    with open(out_path, "w", encoding="utf-8") as out_f:
        for record in conversation:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Final conversation saved.")


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline: Diarization (Pyannote) + Transcription (Whisper).")
    parser.add_argument("--mp3", type=str, required=True, help="Path to the input audio file (.mp3 or .wav).")
    parser.add_argument("--pyannote-path", type=str, required=True, help="Path to the local Pyannote repository directory (e.g., './blabberfish').")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size (tiny, base, small, medium, large).")
    parser.add_argument("--out", default="conversation.jsonl", help="Output JSONL file path for the conversation.")

    args = parser.parse_args()
    
    device = get_device()
    
    diarization_output = run_diarization(args.mp3, args.pyannote_path, device)
    
    final_conversation = run_transcription(args.mp3, diarization_output, args.whisper_model, device)
    
    save_conversation(final_conversation, args.out)

if __name__ == "__main__":
    main()
