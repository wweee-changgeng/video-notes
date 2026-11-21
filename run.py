# -*- coding: utf-8 -*-
"""一条命令跑完：视频（链接或本地文件）→ 带图笔记。

用法：
    python run.py 视频.mp4
    python run.py "https://www.bilibili.com/video/BVxxxx/" --out out/名字
    python run.py 视频.mp4 --no-ocr     # 跳过 OCR，直接用语音转写

依赖：见 requirements.txt；另外要 ffmpeg（下载 B 站视频/抽音轨用）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable


def call(args: list) -> None:
    print("\n$", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True)


def probe(video: Path) -> dict:
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    step = max(1, int(round(fps / 4)))
    prev, ds, idx = None, [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [30, 16, 16], [0, 180, 0, 256, 0, 256])
            cv2.normalize(h, h)
            if prev is not None:
                ds.append(cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA))
            prev = h
        idx += 1
    cap.release()
    d = np.array(ds)
    # 阈值按这条片子自己的分布定：p98 附近，避开运动画面的误切
    thr = round(float(np.percentile(d, 98)) * 0.95, 2) if len(d) else 0.5
    return {"duration": n / fps, "p50": float(np.percentile(d, 50)) if len(d) else 0,
            "p98": float(np.percentile(d, 98)) if len(d) else 0,
            "threshold": max(0.3, min(0.8, thr))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="本地视频，或 B 站链接")
    ap.add_argument("--out", default="", help="输出目录，默认在视频旁边建 <名字>_notes")
    ap.add_argument("--no-ocr", action="store_true", help="不做硬字幕 OCR，直接语音转写")
    ap.add_argument("--frames", type=int, default=3, help="每章取几张帧送给模型")
    ap.add_argument("--ffmpeg", default="", help="ffmpeg 路径（下载/抽音轨需要）")
    args = ap.parse_args()

    src = args.source
    out = Path(args.out) if args.out else None
    if not src.lower().startswith("http"):
        src = Path(src)

    work = out or (src.with_name(src.stem + "_notes") if isinstance(src, Path) else Path("notes"))
    work.mkdir(parents=True, exist_ok=True)

    # 1) 下载
    if isinstance(src, str):
        print("[1/7] 下载")
        tgt = work / "%(id)s.%(ext)s"
        cmd = [PY, "-m", "yt_dlp", "-f", "bv*[height<=720]+ba/b[height<=720]",
               "--merge-output-format", "mp4", "-o", str(tgt), src]
        if args.ffmpeg:
            cmd += ["--ffmpeg-location", args.ffmpeg]
        call(cmd)
        vids = list(work.glob("*.mp4")) + list(work.glob("*.mkv"))
        if not vids:
            raise SystemExit("下载没出文件")
        video = vids[0]
    else:
        video = src

    # 2) 摸分布、定阈值、切镜头
    print("[2/7] 测分布 + 切镜头")
    info = probe(video)
    print(f"  时长 {info['duration']:.1f}s, p50={info['p50']:.3f}, p98={info['p98']:.3f}, "
          f"阈值 {info['threshold']}")
    call([PY, HERE / "layer1_shot_split.py", video, "--out", work / "shots",
          "--threshold", info["threshold"], "--min-gap", 2.5])

    # 3) 字幕：先 OCR 硬字幕，抓不到再语音转写
    subs = work / "subs.json"
    if not args.no_ocr:
        print("[3/7] OCR 硬字幕")
        call([PY, HERE / "ocr_subtitles.py", video, "--out", subs])
        got = json.loads(subs.read_text(encoding="utf-8")).get("lines") or []
        if not got:
            print("  画面里没有字幕，改用语音转写")
            subs = work / "segments.json"
            call([PY, HERE / "transcribe_segments.py", video, "--out", subs])
    else:
        print("[3/7] 语音转写")
        subs = work / "segments.json"
        call([PY, HERE / "transcribe_segments.py", video, "--out", subs])

    # 4) 清洗字幕
    print("[4/7] 清洗字幕")
    clean = work / "subs_clean.json"
    call([PY, HERE / "polish_subs.py", subs, "--out", clean])

    # 5) 语义切章
    print("[5/7] 语义切章")
    chapters = work / "chapters.json"
    call([PY, HERE / "split_semantic.py", clean, "--out", chapters])

    # 6) 画面描述
    print("[6/7] 画面描述")
    desc = work / "desc.json"
    call([PY, HERE / "describe_chapters.py", chapters, clean, work / "shots",
          "--out", desc, "--frames", str(args.frames)])

    # 7) 拼笔记
    print("[7/7] 拼笔记")
    note = work / "视频笔记.md"
    call([PY, HERE / "assemble_note.py", chapters, work / "shots" / "shots.json", clean,
          "--out", note, "--per-chapter", str(args.frames), "--desc", desc,
          "--title", video.stem])

    print(f"\n完成：{note}")
    print(f"帧和图都在 {work / 'shots'}")


if __name__ == "__main__":
    main()
