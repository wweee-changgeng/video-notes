# -*- coding: utf-8 -*-
"""给每一章配「看图理解」：把该章的几张代表帧 + 台词一起送进模型，让它说画面里在发生什么。

这是整条链路里唯一真正调用视觉模型的一步。前面切镜头、OCR 字幕都只是准备材料。

用法：
    python describe_chapters.py chapters.json subs.json shots_dir --out desc.json --frames 4
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

CFG = Path(os.environ.get("DEEPSEEK_CONFIG") or (Path(__file__).parent / "config.json"))
USAGE = {"prompt": 0, "completion": 0, "calls": 0}

SYSTEM = (
    "你在看图说话。给你一段视频里的连续几张画面和这段时间的台词。\n"
    "用三句话回答：\n"
    "1. 画面里是什么地方、有谁（看得出角色形象就说，看不出就说看不清）。\n"
    "2. 这几张图之间发生了什么变化（动作、镜头、表情）。\n"
    "3. 画面和台词对上了没有：台词在说的事，画面是不是在演同一件事。\n"
    "只写画面里看得见的东西，不要推测剧情，不要复述台词。总字数不超过 120 字。"
)


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def describe(frames: list[Path], lines: list[str]) -> str:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    content = [{"type": "text",
                "text": "台词（按时间顺序）：\n" + " / ".join(lines[:20]) + "\n\n画面："}]
    for f in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + b64(f)}})
    body = {"model": cfg.get("model", "deepseek-chat"),
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": content}],
            "temperature": 0.4, "max_tokens": 300}
    req = urllib.request.Request(
        cfg.get("base_url", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    usage = data.get("usage") or {}
    USAGE["prompt"] += int(usage.get("prompt_tokens") or 0)
    USAGE["completion"] += int(usage.get("completion_tokens") or 0)
    USAGE["calls"] += 1
    return (data["choices"][0]["message"].get("content") or "").strip()


def pick_even(items, n):
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("chapters")
    ap.add_argument("subs")
    ap.add_argument("shots_dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--only", type=int, default=0, help="只跑第几章（1 起，0 表示全部）")
    args = ap.parse_args()

    ch = json.loads(Path(args.chapters).read_text(encoding="utf-8"))
    sb = json.loads(Path(args.subs).read_text(encoding="utf-8"))
    shots_dir = Path(args.shots_dir)
    shots = sorted(shots_dir.glob("shot_*.jpg"),
                   key=lambda p: float(p.stem.split("_")[-1].rstrip("s")))

    out = Path(args.out) if args.out else Path(args.chapters).with_name("desc.json")
    results = []
    for i, c in enumerate(ch["chapters"], 1):
        if args.only and i != args.only:
            continue
        s, e = c["start"], c["end"]
        body = [x["text"] for x in (sb.get("lines") or sb.get("segments") or [])
                if s <= x["start"] < e]
        def t_of(p: Path) -> float:
            return float(p.stem.split("_")[-1].rstrip("s"))
        # 章节可能比镜头短（一个长镜头横跨好几章）：取离章节中点最近的几张
        mid = (s + e) / 2
        cands = sorted(shots, key=lambda p: abs(t_of(p) - mid))
        frames = sorted(cands[:args.frames], key=t_of)
        if not frames:
            continue
        print(f"[{i}] {c['title']}  frames={len(frames)}  lines={len(body)}", flush=True)
        desc = describe(frames, body)
        print("    " + desc.replace("\n", " ")[:160], flush=True)
        results.append({"n": i, "title": c["title"], "start": s, "end": e,
                        "frames": [p.name for p in frames], "desc": desc})

    out.write_text(json.dumps({"chapters": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    cost = USAGE["prompt"]/1e6*1 + USAGE["completion"]/1e6*4
    print(f"calls={USAGE['calls']} prompt={USAGE['prompt']} completion={USAGE['completion']} -> 约 {cost:.4f} 元（空闲时段价）")
    print(f"out: {out}")


if __name__ == "__main__":
    main()
