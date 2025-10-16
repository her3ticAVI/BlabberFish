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
import whisper
from pyannote.audio import Pipeline
from halo import Halo 

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
    """Convert mp3/mp4 to wav using ffmpeg (for pyannote compatibility)."""
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


def transcribe_with_whisper(audio_file, model):
    """Transcribe audio file using Whisper (local). Returns structured segments."""
    spinner = Halo(
        text=f"Transcribing {os.path.basename(audio_file)} using Whisper...", 
        spinner=SPINNER_TYPE, 
        color=SPINNER_COLOR
    ).start()
    
    try:
        result = model.transcribe(audio_file, verbose=False) 
        spinner.succeed("Transcription complete.")
        return result.get("segments", [])
    except Exception as e:
        spinner.fail(f"Transcription failed for {os.path.basename(audio_file)}.")
        raise e


def diarize_with_pyannote(audio_file, diarization_pipeline):
    """Run speaker diarization on audio file. Returns speaker segments."""
    spinner = Halo(
        text=f"Running speaker diarization on {os.path.basename(audio_file)} using Pyannote...", 
        spinner=SPINNER_TYPE, 
        color=SPINNER_COLOR
    ).start()
    
    try:
        diarization = diarization_pipeline(audio_file)
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append(
                {"start": turn.start, "end": turn.end, "speaker": speaker}
            )
        spinner.succeed("Diarization complete.")
        return speaker_segments
    except Exception as e:
        spinner.fail(f"Diarization failed for {os.path.basename(audio_file)}.")
        raise e


def align_transcription_with_diarization(transcript_segments, speaker_segments):
    """Align Whisper transcript segments with diarized speaker segments."""
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
    """Merge consecutive transcript segments by the same speaker."""
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
    """Write transcripts as Markdown (one file or multiple if split_md=True)."""
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


def process_files(files, whisper_model, diarization_pipeline, out_path, split_md):
    """Process and transcribe a list of mp3/mp4 files, writing JSONL + Markdown output."""
    if not files:
        print("No media files found.")
        return

    records = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(out_path, "w", encoding="utf-8") as out_f:
            for input_file in files:
                print(f"\nProcessing file: {os.path.basename(input_file)}")
                try:
                    
                    if Path(input_file).suffix.lower() == ".mp4":
                        audio_file = convert_to_wav(input_file, tmpdir)
                    else:
                        audio_file = input_file

                    transcript_segments = transcribe_with_whisper(audio_file, whisper_model)
                    
                    speaker_segments = diarize_with_pyannote(audio_file, diarization_pipeline)
                    
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
        description="Transcribe media files with Whisper and Pyannote diarization."
    )
    parser.add_argument("--zip", help="Path to a ZIP file containing MP3/MP4s")
    parser.add_argument("--mp3", help="Path to a single MP3 file")
    parser.add_argument("--mp4", help="Path to a single MP4 file")
    parser.add_argument("--out", default="transcriptions.jsonl", help="Output JSONL file")
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size (tiny, base, small, medium, large). Default: base.",
    )
    parser.add_argument(
        "--pyannote-token", 
        help="HuggingFace access token for pyannote models. Required."
    )
    parser.add_argument(
        "--split-md",
        action="store_true",
        help="Save one Markdown file per media file (default is combined output).",
    )
    args = parser.parse_args()

    if not args.zip and not args.mp3 and not args.mp4:
        parser.error("Must provide one of: --zip, --mp3, or --mp4.")
    
    if not args.pyannote_token:
        parser.error("Must provide --pyannote-token.")

    model_spinner = Halo(
        text=f"Loading Whisper model: {args.whisper_model}...", 
        spinner=SPINNER_TYPE, 
        color=SPINNER_COLOR
    ).start()
    try:
        whisper_model = whisper.load_model(args.whisper_model)
        model_spinner.succeed(f"Whisper model loaded: {args.whisper_model}.")
    except Exception as e:
        model_spinner.fail(f"Failed to load Whisper model. Details: {e}")
        return

    diarization_spinner = Halo(
        text="Loading Pyannote diarization pipeline...", 
        spinner=SPINNER_TYPE, 
        color=SPINNER_COLOR
    ).start()
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", use_auth_token=args.pyannote_token
        )
        diarization_spinner.succeed("Pyannote pipeline loaded.")
    except Exception as e:
        diarization_spinner.fail(f"Failed to load Pyannote pipeline. Check token/network. Details: {e}")
        return

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
            process_files(files, whisper_model, diarization_pipeline, args.out, args.split_md)
    elif args.mp3:
        files = [args.mp3]
        process_files(files, whisper_model, diarization_pipeline, args.out, args.split_md)
    elif args.mp4:
        files = [args.mp4]
        process_files(files, whisper_model, diarization_pipeline, args.out, args.split_md)


if __name__ == "__main__":
    main()
