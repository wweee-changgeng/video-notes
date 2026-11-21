# -*- coding: utf-8 -*-
"""第 2 层：用语音转写做语义切分。

输入 transcribe_segments.py 产出的 segments.json，把带时间戳的全文交给模型，
让它标出话题转换点，输出章节表（起始时间 + 标题）。

用法：
    python split_semantic.py segments.json --out chapters.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

CFG = Path(os.environ.get("DEEPSEEK_CONFIG") or (Path(__file__).parent / "config.json"))

SYSTEM = (
    "你是一个视频内容编辑。给你一段带时间戳的语音转写，你要把它按话题切成章节。\n"
    "规则：\n"
    "1. 一章讲一件事。同一件事被来回提到，仍算同一章，不要按句子切。\n"
    "2. 章节数按内容决定，通常 3～8 章；内容单一时可以只有 1 章。\n"
    "3. 每章的 start 必须等于原文里某一行的起始时间（数字，秒）。\n"
    "4. 标题写这一章在讲什么，不超过 18 个字，不要用「第一部分」这种编号式。\n"
    "5. 只输出 JSON 数组，不要解释、不要代码块标记。\n"
    '格式：[{"start": 12.3, "title": "……"}, …]'
)


def call(text: str) -> str:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    body = {
        "model": cfg.get("model", "deepseek-chat"),
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": text}],
        "temperature": 0.3,
        "max_tokens": 900,
    }
    req = urllib.request.Request(
        cfg.get("base_url", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = json.loads(r.read())
    return (payload["choices"][0]["message"].get("content") or "").strip()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("segments")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = Path(args.segments)
    out = Path(args.out) if args.out else src.with_name("chapters.json")
    data = json.loads(src.read_text(encoding="utf-8"))
    segs = data.get("segments") or data.get("lines") or []
    if "duration" not in data:
        data["duration"] = segs[-1]["end"] if segs else 0

    lines = [f"[{s['start']:.1f}] {s['text']}" for s in segs]
    transcript = "\n".join(lines)
    print(f"segments={len(segs)}, chars={len(transcript)}", flush=True)

    raw = call(f"视频时长 {data['duration']:.0f} 秒。以下是转写：\n\n{transcript}")
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end < 0:
        print("模型没给 JSON：\n" + raw[:400])
        return
    chapters = json.loads(raw[start:end + 1])

    # 用最近的段落起点校正时间
    starts = [s["start"] for s in segs]
    fixed = []
    for c in chapters:
        t = float(c.get("start", 0))
        nearest = min(starts, key=lambda x: abs(x - t))
        fixed.append({"start": nearest, "title": str(c.get("title", "")).strip()})

    # 补每章的结束时间和字数
    for i, c in enumerate(fixed):
        c["end"] = fixed[i + 1]["start"] if i + 1 < len(fixed) else data["duration"]
        body = [s["text"] for s in segs if c["start"] <= s["start"] < c["end"]]
        c["chars"] = sum(len(x) for x in body)
        c["excerpt"] = "".join(body)[:60]

    out.write_text(json.dumps({"audio": data.get("audio") or data.get("video"),
                               "duration": data["duration"],
                               "chapters": fixed}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"chapters: {len(fixed)}")
    for c in fixed:
        print(f"  [{c['start']:6.1f}s-{c['end']:6.1f}s] {c['title']}  ({c['chars']}字)")
    print(f"out: {out}")


if __name__ == "__main__":
    main()
