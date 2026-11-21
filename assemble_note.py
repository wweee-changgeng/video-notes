# -*- coding: utf-8 -*-
"""把「语义章节 + 镜头帧 + 字幕」拼成一份带图的视频笔记（Markdown）。

用法：
    python assemble_note.py chapters.json shots.json subs.json --out 笔记.md --per-chapter 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def pick_even(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("chapters")
    ap.add_argument("shots")
    ap.add_argument("subs")
    ap.add_argument("--out", default="")
    ap.add_argument("--per-chapter", type=int, default=4)
    ap.add_argument("--desc", default="", help="describe_chapters.py 产出的 desc.json")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    ch = json.loads(Path(args.chapters).read_text(encoding="utf-8"))
    sh = json.loads(Path(args.shots).read_text(encoding="utf-8"))
    sb = json.loads(Path(args.subs).read_text(encoding="utf-8"))

    shots = [s for s in sh.get("seconds", []) if s.get("kind") == "shot" and s.get("frame")]
    subs = sb.get("lines") or sb.get("segments") or []
    descs = {}
    if args.desc:
        dd = json.loads(Path(args.desc).read_text(encoding="utf-8"))
        descs = {int(d["n"]): d for d in dd.get("chapters", [])}
    out = Path(args.out) if args.out else Path(args.chapters).with_name("视频笔记.md")
    base = Path(args.shots).parent.name  # 图片相对 md 的目录名

    lines = [f"# {args.title or Path(ch.get('audio') or '').stem}", ""]
    lines.append(f"- 时长 {ch['duration']:.0f} 秒")
    lines.append(f"- 镜头 {len(shots)} 个，字幕 {len(subs)} 行")
    lines.append("")

    for i, c in enumerate(ch["chapters"], 1):
        s, e = c["start"], c["end"]
        body = [x["text"] for x in subs if s <= x["start"] < e]
        # 镜头段可能比章节长，用「有交集」而不是「中点落在章内」
        in_ch = [x for x in shots if x["start"] < e and x["end"] > s]
        frames = pick_even(in_ch, args.per_chapter)
        lines.append(f"## {i}. {c['title']}")
        lines.append("")
        lines.append(f"`{s:.1f}s – {e:.1f}s`，{len(in_ch)} 个镜头，{len(body)} 行字幕")
        lines.append("")
        d = descs.get(i)
        if d and d.get("desc"):
            lines.append("**画面**：" + d["desc"].replace("\n", " "))
            lines.append("")
        if body:
            lines.append("> " + " / ".join(body))
            lines.append("")
        if frames:
            lines.append(" ".join(f"![{f['mid']:.1f}s]({base}/{f['frame']})" for f in frames))
            lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out}: {len(ch['chapters'])} chapters, {len(shots)} shots")


if __name__ == "__main__":
    main()
