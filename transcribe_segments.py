# -*- coding: utf-8 -*-
"""本地转写，输出**带时间戳**的分段（asr_server 的 HTTP 接口不返回 segments）。

用 faster-whisper 转写；模型名由 WHISPER_MODEL 指定，默认 small。

用法：
    python transcribe_segments.py 音频.m4a --out out.json --lang zh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import av
import numpy as np
from faster_whisper import WhisperModel

MODEL_DIR = os.environ.get("WHISPER_MODEL", "small")


def decode(path: str) -> np.ndarray:
    """自己解码成 16 kHz 单声道，绕开 faster_whisper 与新版 PyAV 的 metadata_errors 不兼容。"""
    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", default="")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--words", action="store_true", help="输出词级时间戳（用于判断是否踩拍）")
    ap.add_argument("--no-vad", action="store_true", help="关掉 VAD（有音乐时 VAD 会把整段砍掉）")
    args = ap.parse_args()

    src = Path(args.audio)
    out = Path(args.out) if args.out else src.with_suffix(".segments.json")

    t0 = time.time()
    model = WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    audio = decode(str(src))
    print(f"decoded {len(audio) / 16000:.1f}s audio in {time.time() - t1:.1f}s", flush=True)
    segments, info = model.transcribe(
        audio,
        language=args.lang,
        beam_size=args.beam,
        vad_filter=not args.no_vad,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
        word_timestamps=args.words,
    )
    rows = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        rows.append({"start": round(s.start, 2), "end": round(s.end, 2),
                     "dur": round(s.end - s.start, 2), "text": text,
                     "words": [{"w": w.word, "t": round(w.start, 2)}
                               for w in (getattr(s, "words", None) or [])]})
    elapsed = time.time() - t1

    out.write_text(json.dumps({
        "audio": str(src),
        "duration": round(float(getattr(info, "duration", 0.0)), 2),
        "language": getattr(info, "language", args.lang),
        "elapsed_sec": round(elapsed, 1),
        "segments": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"segments: {len(rows)}, audio {getattr(info, 'duration', 0):.1f}s, took {elapsed:.1f}s")
    print(f"out: {out}")
    for r in rows[:5]:
        print(f"  [{r['start']:7.2f}-{r['end']:7.2f}] {r['text'][:40]}")


if __name__ == "__main__":
    main()
