#!/usr/bin/env python3

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
import whisper

def extract_zip(zip_path, extract_to):
    """Extract all mp3 files from a zip archive into a temporary directory."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    return [str(p) for p in Path(extract_to).rglob("*.mp3")]

def transcribe_mp3(mp3_file, model):
    """Transcribe a single mp3 file using Whisper (local)."""
    print(f"  ▶ Loading {mp3_file}...")
    result = model.transcribe(mp3_file)
    return result["text"]

def process_files(mp3_files, model, out_path):
    """Process and transcribe a list of mp3 files, writing JSONL output."""
    if not mp3_files:
        print("No MP3 files found.")
        return

    with open(out_path, "w", encoding="utf-8") as out_f:
        for mp3_file in mp3_files:
            print(f"Transcribing {mp3_file}...")
            try:
                text = transcribe_mp3(mp3_file, model)
                record = {
                    "file": os.path.basename(mp3_file),
                    "transcription": text
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"❌ Error transcribing {mp3_file}: {e}")

    print(f"✅ Transcriptions saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP3(s) into JSONL format using Whisper locally.")
    parser.add_argument("--zip", help="Path to a ZIP file containing MP3s")
    parser.add_argument("--mp3", help="Path to a single MP3 file")
    parser.add_argument("--out", default="transcriptions.jsonl", help="Output JSONL file")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    args = parser.parse_args()

    if not args.zip and not args.mp3:
        parser.error("You must provide either --zip or --mp3")

    # Load Whisper model
    print(f"Loading Whisper model: {args.model} ...")
    model = whisper.load_model(args.model)

    if args.zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp3_files = extract_zip(args.zip, tmpdir)
            process_files(mp3_files, model, args.out)
    elif args.mp3:
        process_files([args.mp3], model, args.out)

if __name__ == "__main__":
    main()
