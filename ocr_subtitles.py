# -*- coding: utf-8 -*-
"""从视频画面里 OCR 硬字幕（烧录在画面底部的字幕），输出带时间戳的台词。

为什么不用 ASR：硬字幕本来就是文字，OCR 不会有同音错字。ASR 只在没有字幕时才用。

做法
----
1. 按 2 fps 采样，裁底部字幕带，灰度 → 放大 2 倍 → 二值化。
2. 与上一张二值图比较，变化够大才 OCR（同一句字幕不会重复识别）。
3. 相同文本合并成一条，记录起止时间。

用法：
    python ocr_subtitles.py 视频.mp4 --out subs.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import winocr
from PIL import Image

# 字幕带：避开右侧水印
Y0, Y1 = 0.84, 0.99
X0, X1 = 0.06, 0.86
WATERMARKS = ("冬冰", "bilibili", "哔哩哔哩")


def crop_band(frame: np.ndarray):
    h, w = frame.shape[:2]
    band = frame[int(h * Y0):int(h * Y1), int(w * X0):int(w * X1)]
    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
    return bw


def clean(text: str) -> str:
    t = re.sub(r"\s+", "", text or "")
    for w in WATERMARKS:
        t = t.replace(w, "")
    t = re.sub(r"[`'\"’‘“”·．\^_]", "", t)
    return t.strip("，,。 ｜|")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--diff", type=float, default=8.0, help="二值图平均差超过它才算换字幕")
    args = ap.parse_args()

    src = Path(args.video)
    out = Path(args.out) if args.out else src.with_name("subs.json")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"打不开: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / args.fps)))

    rows: list[dict] = []
    prev_bw = None
    idx = 0
    ocr_calls = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        t = idx / fps
        bw = crop_band(frame)

        if prev_bw is not None and bw.shape == prev_bw.shape:
            if float(np.mean(cv2.absdiff(bw, prev_bw))) < args.diff:
                # 同一句还在屏幕上：把这句的结束时间往后推
                if rows:
                    rows[-1]["end"] = round(t, 2)
                idx += 1
                continue

        tmp = Path(tempfile.gettempdir()) / "_video_notes_band.png"
        cv2.imwrite(str(tmp), bw)
        res = winocr.recognize_pil_sync(Image.open(tmp), "zh-Hans")
        ocr_calls += 1
        text = clean(res.get("text", ""))
        prev_bw = bw

        if not text:
            idx += 1
            continue
        if rows and rows[-1]["text"] == text:
            rows[-1]["end"] = round(t, 2)
        else:
            rows.append({"start": round(t, 2), "end": round(t, 2), "text": text})
        if ocr_calls % 20 == 0:
            print(f"  [{t:6.1f}s] ocr={ocr_calls} lines={len(rows)}", flush=True)
        idx += 1
    cap.release()

    out.write_text(json.dumps({"video": str(src), "fps": args.fps,
                               "ocr_calls": ocr_calls, "lines": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ocr calls: {ocr_calls}, subtitle lines: {len(rows)}")
    print(f"out: {out}")
    for r in rows[:8]:
        print(f"  [{r['start']:6.1f}-{r['end']:6.1f}] {r['text']}")


if __name__ == "__main__":
    main()
