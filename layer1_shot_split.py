# -*- coding: utf-8 -*-
"""第 1 层：用 HSV 直方图做镜头切分，挑出「稳定镜头的中点帧」。

思路
----
1. 按固定采样率（默认 4 fps）读帧，每帧算 3D HSV 直方图（H30 × S16 × V16，归一化）。
2. 相邻采样帧的直方图距离（Bhattacharyya，0~1）超过阈值 → 记为切点。
3. 切点之间是一段「状态」；段时长短于「全片段时长中位数 / 5」的判为过渡（叠化、闪白、
   快剪），跳过，不取帧。
4. 每个稳定段取中点（画面停稳处）保存 PNG，并输出 JSON 清单。

用法
----
    python layer1_shot_split.py 视频.mp4 --out 输出目录 --sample-fps 4 --threshold 0.40
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np


def hsv_hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [30, 16, 16], [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h)
    return h


def scan(video: Path, sample_fps: float):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    times, hists = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            times.append(idx / fps)
            hists.append(hsv_hist(frame))
        idx += 1
    cap.release()
    return fps, times, hists


def split(times, hists, threshold: float, min_gap: float = 0.0):
    """阈值切分 + 最小间隔：避免第一人称/运动镜头把同一次移动切成几十段。"""
    if not hists:
        return []
    cuts = [0]
    last_t = times[0] if times else 0.0
    for i in range(1, len(hists)):
        d = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)
        if d > threshold and (times[i] - last_t) >= min_gap:
            cuts.append(i)
            last_t = times[i]
    cuts.append(len(hists))
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        segs.append({"start": times[a], "end": times[min(b, len(times) - 1)], "i0": a, "i1": b - 1})
    return segs


def grab_frame(video: Path, t: float, out: Path) -> bool:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cap.release()
    return ok


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="")
    ap.add_argument("--sample-fps", type=float, default=4.0)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--min-gap", type=float, default=0.0, help="两次切点之间至少隔多少秒")
    ap.add_argument("--keep-ratio", type=float, default=5.0,
                    help="段时长短于 中位数/该值 的判为过渡，跳过")
    args = ap.parse_args()

    video = Path(args.video)
    out_dir = Path(args.out) if args.out else video.with_name(video.stem + "_shots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"scanning {video.name} ...", flush=True)
    fps, times, hists = scan(video, args.sample_fps)
    print(f"  fps={fps:.2f}, sampled frames={len(hists)}", flush=True)

    segs = split(times, hists, args.threshold, args.min_gap)
    durs = [s["end"] - s["start"] for s in segs]
    if not durs:
        print("没有切出任何段")
        return
    med = statistics.median(durs)
    short = med / args.keep_ratio
    print(f"  segments={len(segs)}, median={med:.2f}s, transition<{short:.2f}s", flush=True)

    rows = []
    kept = 0
    for n, s in enumerate(segs, 1):
        dur = s["end"] - s["start"]
        kind = "transition" if dur < short else "shot"
        mid = (s["start"] + s["end"]) / 2
        item = {"n": n, "start": round(s["start"], 2), "end": round(s["end"], 2),
                "dur": round(dur, 2), "mid": round(mid, 2), "kind": kind}
        if kind == "shot":
            name = out_dir / f"shot_{n:03d}_{mid:07.2f}s.jpg"
            if grab_frame(video, mid, name):
                item["frame"] = name.name
                kept += 1
        rows.append(item)

    (out_dir / "shots.json").write_text(
        json.dumps({"video": str(video), "fps": fps, "sample_fps": args.sample_fps,
                    "threshold": args.threshold, "median_dur": round(med, 2),
                    "transition_below": round(short, 2), "seconds": rows},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")

    total = rows[-1]["end"] if rows else 0
    shots = [r for r in rows if r["kind"] == "shot"]
    print(f"done: {total:.1f}s video -> {len(shots)} shots kept ({kept} frames), "
          f"{len(rows) - len(shots)} transitions skipped", flush=True)
    print(f"out: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
