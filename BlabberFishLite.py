#!/usr/bin/env python3

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio.*list_audio_backends.*deprecated.*"
)
import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import requests
from halo import Halo

WHISPER_API_BASE = "https://api-inference.huggingface.co/models/openai/whisper-"
PYANNOTE_API_URL = "https://api-inference.huggingface.co/models/pyannote/speaker-diarization"

SPINNER_TYPE = "pong"
SPINNER_COLOR = "red"


def extract_zip(zip_path, extract_to):
    """Extract all mp3/mp4 files from a zip archive into a temporary directory."""
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    return [
        str(p)
        for p in Path(extract_to).rglob("*")
        if p.suffix.lower() in (".mp3", ".mp4")
    ]


def convert_to_wav(input_file, tmpdir):
    """Convert media to wav using ffmpeg (necessary if API requires a specific format)."""
    filename = os.path.basename(input_file)
    output_file = os.path.join(
        tmpdir, os.path.splitext(filename)[0] + ".wav"
    )
    
    spinner = Halo(
        text=f"Converting {filename} to WAV (16kHz, mono)...", 
        spinner=SPINNER_TYPE, 
        color=SPINNER_COLOR
    ).start()
    
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1", output_file],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        spinner.succeed("Conversion complete.")
        return output_file
    except subprocess.CalledProcessError as e:
        spinner.fail(f"FFmpeg conversion failed for {filename}.")
        raise e


def transcribe_via_api(audio_file, whisper_model_id, token):
    """Transcribe audio file using Hugging Face Inference API."""
    
    # Map local model name to Hugging Face Model ID
    model_suffix = whisper_model_id if 'large' in whisper_model_id.lower() else whisper_model_id + '.en'
    api_url = WHISPER_API_BASE + model_suffix
    headers = {"Authorization": f"Bearer {token}"}
    
    spinner = Halo(
        text=f"Transcribing {os.path.basename(audio_file)} via HF API ({model_suffix})...", 
        spinner=SPINNER_TYPE, 
        color=SPINNER_COLOR
    ).start()
    
    try:
        with open(audio_file, "rb") as f:
            data = f.read()
        
        response = requests.post(
            api_url, 
            headers=headers, 
            data=data,
            params={
                "return_timestamps": "word"
            }
        )
        response.raise_for_status()
        
        result = response.json()
        
        if isinstance(result, dict) and "chunks" in result:
             segments = [
                {"text": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1]}
                for c in result["chunks"]
            ]
        elif isinstance(result, str):

            spinner.fail("HF ASR API did not return segments with timestamps. Diarization alignment is impossible.")
            raise ValueError("ASR API must return segments/timestamps for diarization.")
        else:
            segments = []
            
        spinner.succeed("Transcription complete.")
        return segments
        
    except requests.exceptions.HTTPError as e:
        spinner.fail(f"HF API (ASR) failed for {os.path.basename(audio_file)}. HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        raise e
    except Exception as e:
        spinner.fail(f"Transcription failed for {os.path.basename(audio_file)}. Details: {e}")
        raise e


def diarize_via_api(audio_file, token):
    """Run speaker diarization using Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "audio/wav"}
    
    spinner = Halo(
        text=f"Running speaker diarization on {os.path.basename(audio_file)} via HF API...", 
        spinner=SPINNER_TYPE, 
        color=SPINNER_COLOR
    ).start()
    
    try:
        with open(audio_file, "rb") as f:
            data = f.read()
          
        response = requests.post(
            PYANNOTE_API_URL, 
            headers=headers, 
            data=data,
            params={
                "task": "speaker-diarization"
            }
        )
        response.raise_for_status()
        
        speaker_segments = response.json()
        
        spinner.succeed("Diarization complete.")
        return speaker_segments
        
    except requests.exceptions.HTTPError as e:
        spinner.fail(f"HF API (Diarization) failed for {os.path.basename(audio_file)}. HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        raise e
    except Exception as e:
        spinner.fail(f"Diarization failed for {os.path.basename(audio_file)}. Details: {e}")
        raise e


def align_transcription_with_diarization(transcript_segments, speaker_segments):
    """Align Whisper transcript segments with diarized speaker segments (same logic)."""
    aligned = []
    for t in transcript_segments:
        t_start, t_end, text = t["start"], t["end"], t["text"].strip()
        best_speaker = None
        best_overlap = 0.0

        for s in speaker_segments:
            overlap = max(0, min(t_end, s["end"]) - max(t_start, s["start"]))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = s["speaker"]

        aligned.append(
            {
                "speaker": best_speaker if best_speaker else "unknown",
                "start": t_start,
                "end": t_end,
                "text": text,
            }
        )
    return merge_adjacent_segments(aligned)


def merge_adjacent_segments(segments):
    """Merge consecutive transcript segments by the same speaker (same logic)."""
    if not segments:
        return []
    
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"]:
            last["text"] += " " + seg["text"]
            last["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def write_markdown(jsonl_records, md_path, split_md=False):
    """Write transcripts as Markdown (same logic)."""

    now_utc = datetime.now(timezone.utc)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if split_md:
        for record in jsonl_records:
            filename = record["file"]
            conversation = record["conversation"]
            out_path = os.path.join(
                os.path.dirname(md_path) or ".",
                f"{os.path.splitext(filename)[0]}.md",
            )
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write(f"# {filename}\n\n")
                out_f.write(f"## Conversation {timestamp}\n\n")
                for entry in conversation:
                    out_f.write(f"**{entry['speaker']}**: {entry['text']}\n\n")
            print(f"Markdown transcript saved to {out_path}")
    else:
        with open(md_path, "w", encoding="utf-8") as out_f:
            for record in jsonl_records:
                filename = record["file"]
                conversation = record["conversation"]
                out_f.write(f"# {filename}\n\n")
                out_f.write(f"## Conversation {timestamp}\n\n")
                for entry in conversation:
                    out_f.write(f"**{entry['speaker']}**: {entry['text']}\n\n")
                out_f.write("---\n\n")
        print(f"Combined Markdown transcript saved to {md_path}")


def process_files(files, whisper_model_id, pyannote_token, out_path, split_md):
    """Process and transcribe a list of mp3/mp4 files, writing JSONL + Markdown output."""
    if not files:
        print("No media files found.")
        return

    records = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(out_path, "w", encoding="utf-8") as out_f:
            for input_file in files:
                print(f"\nProcessing file: {os.path.basename(input_file)}")
                audio_for_api = input_file
                
                try:
                    if Path(input_file).suffix.lower() == ".mp4" or Path(input_file).suffix.lower() == ".mp3":
                        audio_for_api = convert_to_wav(input_file, tmpdir)
                    

                    transcript_segments = transcribe_via_api(
                        audio_for_api, whisper_model_id, pyannote_token
                    )
                    
                    speaker_segments = diarize_via_api(
                        audio_for_api, pyannote_token
                    )
                    
                    conversation = align_transcription_with_diarization(
                        transcript_segments, speaker_segments
                    )

                    record = {"file": os.path.basename(input_file), "conversation": conversation}
                    records.append(record)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"File processing complete: {os.path.basename(input_file)}")

                except subprocess.CalledProcessError:
                    print(f"Error: FFmpeg required but failed for {os.path.basename(input_file)}. Check installation.")
                except Exception as e:
                    print(f"Error processing {os.path.basename(input_file)}. Details: {e}")

    md_path = os.path.splitext(out_path)[0] + ".md"
    write_markdown(records, md_path, split_md)
    print(f"\nProcess complete. JSONL saved to {out_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe media files with Whisper and Pyannote diarization (Hugging Face API offloaded)."
    )
    parser.add_argument("--zip", help="Path to a ZIP file containing MP3/MP4s")
    parser.add_argument("--mp3", help="Path to a single MP3 file")
    parser.add_argument("--mp4", help="Path to a single MP4 file")
    parser.add_argument("--out", default="transcriptions.jsonl", help="Output JSONL file")
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size (tiny, base, small, medium, large). Default: base. Used to construct the HF API model ID.",
    )
    parser.add_argument(
        "--hf-token", 
        required=True,
        help="HuggingFace API access token. Required."
    )
    parser.add_argument(
        "--split-md",
        action="store_true",
        help="Save one Markdown file per media file (default is combined output).",
    )
    args = parser.parse_args()

    if not args.zip and not args.mp3 and not args.mp4:
        parser.error("Must provide one of: --zip, --mp3, or --mp4.")
    
    print("\n--- Hugging Face API Mode ---")
    print("Local Pyannote and Whisper libraries skipped.")
    print(f"Using Whisper model ID: {'openai/whisper-' + args.whisper_model + ('.en' if 'large' not in args.whisper_model else '')}")
    print(f"Using Pyannote API Endpoint: {PYANNOTE_API_URL}")
    print("---------------------------\n")

    files = []
    if args.zip:
        zip_spinner = Halo(
            text=f"Extracting ZIP file {args.zip}...", 
            spinner=SPINNER_TYPE, 
            color=SPINNER_COLOR
        ).start()
        with tempfile.TemporaryDirectory() as tmpdir:
            files = extract_zip(args.zip, tmpdir)
            zip_spinner.succeed(f"Extracted {len(files)} media file(s).")
            process_files(files, args.whisper_model, args.hf_token, args.out, args.split_md)
    elif args.mp3:
        files = [args.mp3]
        process_files(files, args.whisper_model, args.hf_token, args.out, args.split_md)
    elif args.mp4:
        files = [args.mp4]
        process_files(files, args.whisper_model, args.hf_token, args.out, args.split_md)


if __name__ == "__main__":
    main()
