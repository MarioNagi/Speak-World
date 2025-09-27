#!/usr/bin/env python3
"""
Simple subtitle generator using faster-whisper.
Usage:
    python tools/generate_subtitles.py input_video.mp4 output.srt --model small

Requirements:
- ffmpeg (installed on PATH)
- faster-whisper installed in the Python env

This script extracts audio with ffmpeg, runs faster-whisper to get timestamps
and writes a basic SRT file.
"""
import argparse
import os
import subprocess
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except Exception as e:
    raise SystemExit("faster-whisper is required: pip install faster-whisper")


def extract_audio(input_path: Path, out_wav: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_wav),
    ]
    subprocess.check_call(cmd)


def format_timestamp(seconds: float) -> str:
    ms = int(seconds * 1000)
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            text = seg.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def main():
    p = argparse.ArgumentParser()
    # allow either positional or flag-style input/output
    p.add_argument("input", nargs="?", help="Input video/audio file")
    p.add_argument("output", nargs="?", help="Output SRT file or directory")
    p.add_argument("-i", "--input", dest="in_flag", help="Input video/audio file (optional flag)")
    p.add_argument("-o", "--output", dest="out_flag", help="Output SRT file or directory (optional flag)")
    p.add_argument("--model", default="small", help="Whisper model size (tiny, base, small, medium, large)")
    p.add_argument("--device", default="cpu", help="Device to run model on (cpu or cuda)")
    args = p.parse_args()

    def _sanitize(s: str) -> str:
        if s is None:
            return None
        # strip surrounding quotes that sometimes appear on Windows command lines
        return s.strip().strip('"').strip("'")

    # choose flag over positional if provided
    in_arg = _sanitize(args.in_flag or args.input)
    out_arg = _sanitize(args.out_flag or args.output)

    if not in_arg:
        raise SystemExit("Error: input file is required. Example:\n  python tools/generate_subtitles.py input.mp4 output.srt")

    inp = Path(in_arg).expanduser()
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    # determine output path
    if out_arg:
        out = Path(out_arg).expanduser()
        # if user passed an existing directory, write <input_stem>.srt into it
        if out.exists() and out.is_dir():
            out = out / f"{inp.stem}.srt"
        # if output has no suffix, assume .srt
        if out.suffix == "":
            out = out.with_suffix(".srt")
    else:
        # default output next to input with .srt
        out = inp.with_suffix('.srt')

    # tmp wav will be placed next to output if it has a filename, else next to input
    if out.name:
        tmp_wav = out.with_suffix('.wav')
    else:
        tmp_wav = inp.with_suffix('.wav')

    print(f"Input: {inp}")
    print(f"Output: {out}")
    print(f"Temporary WAV: {tmp_wav}")

    print(f"Extracting audio to {tmp_wav}...")
    try:
        extract_audio(inp, tmp_wav)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ffmpeg failed: {e}")

    print(f"Loading Whisper model ({args.model}) on {args.device}...")
    model = WhisperModel(args.model, device=args.device)

    print("Transcribing...")
    try:
        segments, info = model.transcribe(str(tmp_wav), beam_size=5, language="en")
        # faster-whisper may return a generator for segments; convert to list so we can
        # measure length and iterate multiple times.
        segments = list(segments)
    except Exception as e:
        raise SystemExit(f"Transcription failed: {e}")

    print(f"Writing {len(segments)} segments to {out}...")
    write_srt(segments, out)

    print("Done")


if __name__ == "__main__":
    main()
