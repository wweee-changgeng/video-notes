# -*- coding: utf-8 -*-
"""人声/音乐分段 + 时间轴对齐。

两件事：
1. 用 ASR 的时间戳反推「谁在说话」，中间的空白就是纯音乐段；只在音乐段上算 BPM，
   不会被语音重音带歪。
2. 把画面切点和音频事件（说话起止、鼓点）对齐，看剪辑落在拍子上还是在说话间隙。

用法：
    python audio_align.py 视频.mp4 segments.json shots.json --out align.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_audio(path: Path):
    """视频容器 librosa 读不了，先用 ffmpeg 抽成 wav。"""
    if path.suffix.lower() in AUDIO_EXT:
        return librosa.load(str(path), mono=True, sr=22050)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run([ff, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "22050", tmp],
                   capture_output=True, timeout=600)
    y, sr = librosa.load(tmp, mono=True, sr=22050)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return y, sr


def speech_regions(segs: list, dur: float, pad: float = 0.3) -> list[tuple[float, float]]:
    out = []
    for s in segs:
        a = max(0.0, float(s["start"]) - pad)
        b = min(dur, float(s["end"]) + pad)
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def gaps(regions: list[tuple[float, float]], dur: float, min_len: float = 1.5) -> list[tuple[float, float]]:
    out, cur = [], 0.0
    for a, b in regions:
        if a - cur >= min_len:
            out.append((cur, a))
        cur = max(cur, b)
    if dur - cur >= min_len:
        out.append((cur, dur))
    return out


def bpm_of(y: np.ndarray, sr: int, a: float, b: float) -> float:
    seg = y[int(a * sr):int(b * sr)]
    if len(seg) < sr:
        return 0.0
    env = librosa.onset.onset_strength(y=seg, sr=sr)
    t, _ = librosa.beat.beat_track(onset_envelope=env, sr=sr)
    return round(float(np.atleast_1d(t)[0]), 1)


def mix_balance(y: np.ndarray, sr: int, speech: list, dur: float) -> dict:
    """人声/背景比：说话段响度减说话间隙响度，近似人声电平。

    正常混音人声比伴奏高 6~10 dB；3 dB 以内算喧宾夺主；没间隙就明说算不了。
    """
    small_gaps = gaps(speech, dur, 0.12)
    total_gap = sum(b - a for a, b in small_gaps)
    if not small_gaps or total_gap < 0.5 or not speech:
        return {"ok": False, "reason": "几乎没有说话间隙，估不了"}

    def rms(a, b):
        seg = y[int(a * sr):int(b * sr)]
        return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0

    sp = float(np.mean([rms(a, b) for a, b in speech])) or 1e-9
    bg = float(np.mean([rms(a, b) for a, b in small_gaps])) or 1e-9
    ratio = 20 * np.log10(sp / bg)
    verdict = ("几乎没有BGM" if ratio >= 20 else "正常" if ratio >= 6 else ("人声偏小" if ratio >= 3 else "喧宾夺主"))
    return {"ok": True, "ratio_db": round(ratio, 1),
            "speech_rms": round(sp, 4), "bg_rms": round(bg, 4),
            "gap_seconds": round(total_gap, 1), "verdict": verdict}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("segments")
    ap.add_argument("shots")
    ap.add_argument("--out", default="")
    ap.add_argument("--tol", type=float, default=0.12, help="切点与音频事件相差多少秒内算同步")
    args = ap.parse_args()

    video = Path(args.video)
    seg_data = json.loads(Path(args.segments).read_text(encoding="utf-8"))
    shot_data = json.loads(Path(args.shots).read_text(encoding="utf-8"))
    segs = seg_data.get("segments") or seg_data.get("lines") or []

    y, sr = load_audio(video)
    dur = len(y) / sr

    speech = speech_regions(segs, dur)
    music = gaps(speech, dur, 1.5)
    mix = mix_balance(y, sr, speech, dur)
    if mix.get("ok"):
        print(f"人声/背景 {mix['ratio_db']:+.1f} dB -> {mix['verdict']}"
              f"（间隙共 {mix['gap_seconds']}s）")
    else:
        print(f"人声/背景：{mix['reason']}")

    talk_time = sum(b - a for a, b in speech)
    print(f"{video.name}: {dur:.1f}s | 说话 {talk_time:.0f}s ({talk_time/dur:.0%}) | 音乐/静音 {len(music)} 段")
    regions = []
    for i, (a, b) in enumerate(music, 1):
        bpm = bpm_of(y, sr, a, b)
        rms = float(np.mean(librosa.feature.rms(y=y[int(a * sr):int(b * sr)])[0]))
        regions.append({"n": i, "start": round(a, 1), "end": round(b, 1),
                        "dur": round(b - a, 1), "bpm": bpm, "rms": round(rms, 3)})
        if b - a >= 3:
            print(f"  音乐段 {i}: {a:6.1f}-{b:6.1f}s ({b-a:5.1f}s) bpm={bpm} rms={rms:.3f}")

    # 音频事件：说话起止 + 音乐段内部鼓点
    events = []
    for a, b in speech:
        events.append((a, "speech_in"))
        events.append((b, "speech_out"))
    for r in regions:
        if r["dur"] < 3 or not r["bpm"]:
            continue
        start = int(r["start"] * sr)
        seg = y[start:int(r["end"] * sr)]
        env = librosa.onset.onset_strength(y=seg, sr=sr)
        for t in librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="time"):
            events.append((r["start"] + t, "beat"))

    cuts = [s["start"] for s in shot_data.get("seconds", []) if s.get("kind") == "shot"]
    matched, detail = 0, []
    for t in cuts:
        if not events:
            break
        dt, kind = min(((abs(t - e[0]), e[1]) for e in events), key=lambda x: x[0])
        ok = dt <= args.tol
        matched += ok
        detail.append({"cut": round(t, 2), "nearest": round(dt, 2), "kind": kind, "synced": ok})

    print(f"画面切点 {len(cuts)} 个，其中 {matched} 个 (±{args.tol}s) 落在音频事件上"
          f"（{matched/max(1,len(cuts)):.0%}）")
    kinds = {}
    for d in detail:
        if d["synced"]:
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    print(f"  同步的切点里：{kinds}")

    out = Path(args.out) if args.out else video.with_suffix(".align.json")
    out.write_text(json.dumps({
        "video": video.name, "duration": round(dur, 1),
        "speech_ratio": round(talk_time / dur, 3),
        "mix": mix,
        "music_regions": regions,
        "cuts": len(cuts), "synced_cuts": matched,
        "sync_ratio": round(matched / max(1, len(cuts)), 3),
        "detail": detail,
    }, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"out: {out}")


if __name__ == "__main__":
    main()
