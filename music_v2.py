# -*- coding: utf-8 -*-
"""离线拆歌 v2：按正确顺序——鼓点 → 调性 → 和弦。

流程
----
1. 分帧 + STFT。
2. 鼓点：kick 取 40-150 Hz、snare 取 200-2000 Hz 的能量突增，各自出峰值时刻；
   节拍周期用鼓点包络的自相关定（不再用全频带，避免被旋律/人声带偏）。
3. 调性：chroma 全局平均 → Krumhansl 模板相关。
4. 和弦：每 1 秒取一次 chroma，与 24 个三和弦模板匹配。粒度比调性细。
5. 段落：按和弦序列的变化点切，而不是拿 chroma 强行聚类。

用法：
    python music_v2.py 歌.mp3 --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def band_onset(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (freqs >= lo) & (freqs < hi)
    e = S[m].sum(axis=0)
    d = np.diff(e, prepend=e[0])
    env = np.maximum(d, 0)
    mx = env.max() or 1.0
    return env / mx


def key_of(chroma_mean: np.ndarray) -> tuple[str, float]:
    best, score = "", -1.0
    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        for prof, mode in ((MAJOR, "maj"), (MINOR, "min")):
            c = float(np.corrcoef(rolled, prof)[0, 1])
            if c > score:
                score, best = c, f"{NOTES[i]}:{mode}"
    return best, round(score, 3)


# 24 个三和弦模板（根音 × 大小三）
def chord_templates():
    out = []
    for root in range(12):
        for quality, ivs in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
            v = np.zeros(12)
            for iv in ivs:
                v[(root + iv) % 12] = 1.0
            out.append((f"{NOTES[root]}:{quality}", v / np.linalg.norm(v)))
    return out


TEMPLATES = chord_templates()


def estimate_chords(chroma: np.ndarray, sr: int, hop: int, seconds: float = 2.0):
    """每 seconds 秒判定一次和弦，再做一次 3 点平滑（避免逐秒乱跳）。"""
    per = max(1, int(seconds * sr / hop))
    rows = []
    for i in range(0, chroma.shape[1], per):
        block = chroma[:, i:i + per].mean(axis=1)
        n = np.linalg.norm(block)
        if n < 1e-6:
            continue
        v = block / n
        best, score = "-", 0.0
        for name, tpl in TEMPLATES:
            c = float(np.dot(v, tpl))
            if c > score:
                score, best = c, name
        if score < 0.55:      # 太不像就不硬给
            best = "?"
        rows.append({"t": round(i * hop / sr, 1), "chord": best, "score": round(score, 2)})

    # 平滑：每个窗口看邻居，取出现最多的和声（但不跨越 ? 段）
    smoothed = [dict(r) for r in rows]
    for j in range(1, len(rows) - 1):
        trio = [rows[j - 1]["chord"], rows[j]["chord"], rows[j + 1]["chord"]]
        if "?" not in trio:
            counts = {c: trio.count(c) for c in set(trio)}
            smoothed[j]["chord"] = max(counts, key=counts.get)
    return smoothed


def merge_runs(rows):
    """把连续相同的和弦合并成段。"""
    out = []
    for r in rows:
        if out and out[-1]["chord"] == r["chord"]:
            out[-1]["end"] = round(r["t"] + 1.0, 1)
        else:
            out.append({"start": r["t"], "end": round(r["t"] + 1.0, 1),
                        "chord": r["chord"], "score": r["score"]})
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = Path(args.audio)
    out = Path(args.out) if args.out else src.with_suffix(".v2.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(src), mono=True, sr=22050)
    dur = len(y) / sr
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    kick = band_onset(S, freqs, 40, 150)
    snare = band_onset(S, freqs, 200, 2000)
    drums = np.maximum(kick, 0.6 * snare)

    # 1) BPM 从鼓点包络定
    tempo, beats = librosa.beat.beat_track(onset_envelope=drums, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])

    # 2) 鼓点时刻（分开数 kick 和 snare）
    t_kick = librosa.onset.onset_detect(onset_envelope=kick, sr=sr, hop_length=hop, units="time",
                                        delta=0.15, wait=4)
    t_snare = librosa.onset.onset_detect(onset_envelope=snare, sr=sr, hop_length=hop, units="time",
                                         delta=0.15, wait=4)

    # 3) 调性
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    key, key_score = key_of(chroma.mean(axis=1))

    # 4) 和弦（每秒一次）
    raw_chords = estimate_chords(chroma, sr, hop, 1.0)
    chords = merge_runs(raw_chords)

    # 5) 段落：和弦序列的变化点，加上 20 秒的最小间隔
    sections, last = [], 0.0
    for c in chords:
        if c["start"] - last >= 20 and c["chord"] != "?":
            sections.append({"start": round(last, 1), "end": round(c["start"], 1)})
            last = c["start"]
    sections.append({"start": round(last, 1), "end": round(dur, 1)})

    result = {
        "file": src.name, "duration": round(dur, 1),
        "tempo_bpm": round(tempo, 1),
        "beats": int(len(beats)),
        "kick_onsets": int(len(t_kick)),
        "snare_onsets": int(len(t_snare)),
        "key": key, "key_confidence": key_score,
        "chord_count": len(chords),
        "chords": chords[:120],
        "sections": sections,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{src.name}: {dur:.1f}s")
    print(f"  BPM {tempo:.1f} | 拍 {len(beats)} | kick {len(t_kick)} 次 | snare {len(t_snare)} 次")
    print(f"  调性 {key}（相关 {key_score}）")
    print(f"  和弦 {len(chords)} 段（原始 {len(raw_chords)} 秒）")
    head = "  ".join(f"{c['start']:.0f}s {c['chord']}" for c in chords[:14])
    print(f"  前 14 段：{head}")
    print(f"out: {out}")


if __name__ == "__main__":
    main()
