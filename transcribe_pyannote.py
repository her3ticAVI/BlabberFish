import argparse
import json
import sys
import torch
from pyannote.audio import Pipeline

def get_diarization(audio_path: str, pyannote_path: str):
    """
    Applies speaker diarization, prioritizing CPU if the GPU (CUDA) is problematic.
    This helps mitigate the std::length_error often caused by GPU memory fragmentation.
    """
    
    if torch.cuda.is_available():
        # Prefer CPU to bypass fragmentation issue, but if using the GPU,
        # the command line flag 'PYTORCH_NO_CUDA_MEMORY_CACHING=1' should handle it.
        device = torch.device("cpu")
        print(f"DEBUG: CUDA found, but **forcing device to {device}** to avoid std::length_error.")
    else:
        device = torch.device("cpu")
        print(f"DEBUG: CUDA not available. Using device: {device}")

    try:
        pipeline = Pipeline.from_pretrained(
            pyannote_path,
            use_auth_token=False
        ).to(device)
        
    except Exception as e:
        print(f"\n--- ERROR LOADING MODEL ---")
        print(f"Error: {e}")
        print("Suggestion: Ensure 'pyannote/speaker-diarization-3.1' is accepted on Hugging Face and 'git lfs pull' was run in your local pyannote-path.")
        sys.exit(1)

    print(f"\nStarting diarization for {audio_path} on device {device}...")
    try:
        diarization = pipeline(audio_path)
    except Exception as e:
        print(f"\n--- ERROR DURING DIARIZATION ---")
        print(f"The C++ error 'std::length_error: vector::reserve' might be happening here.")
        print(f"Suggestion: Re-run the script with the environment variable:")
        print(f"   PYTORCH_NO_CUDA_MEMORY_CACHING=1 python {sys.argv[0]} ...")
        print(f"Underlying Python Error: {e}")
        sys.exit(1)
        
    print("Diarization complete.")
    
    return diarization


def save_diarization_to_jsonl(diarization, output_path: str):
    """
    Converts the pyannote Annotation object to a JSONL format and saves it.
    """
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    with open(output_path, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Results successfully saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Pyannote Diarization from a local repository.")
    parser.add_argument("--audio", dest='mp3', type=str, required=True, help="Path to the input audio file (.wav or .mp3).")
    parser.add_argument("--pyannote-path", type=str, required=True, help="Path to the local pyannote repository directory (e.g., './blabberfish').")
    parser.add_argument("--out", type=str, required=True, help="Path to save the diarization output (.json or .jsonl).")

    args = parser.parse_args()

    diarization_result = get_diarization(args.mp3, args.pyannote_path)

    save_diarization_to_jsonl(diarization_result, args.out)
