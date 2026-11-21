# -*- coding: utf-8 -*-
"""清洗 OCR 字幕：规则去重 + 模型校对错字。

OCR 从画面里抓字幕有两个毛病：同一句会被识别成几条近似文本；动画描边会让个别字认错。
规则先合并近似行，剩下的交给模型按上下文改。

用法：
    python polish_subs.py subs.json --out subs_clean.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import urllib.request
from pathlib import Path

CFG = Path(os.environ.get("DEEPSEEK_CONFIG") or (Path(__file__).parent / "config.json"))

SYSTEM = (
    "你在校对一段视频的 OCR 字幕。输入是若干行，每行是 [起始秒-结束秒] 文本。\n"
    "要做三件事：\n"
    "1. 合并重复：同一句话被连续识别成多条（哪怕有个别字不同），只保留最完整的一条。\n"
    "2. 改错字：根据上下文把明显认错的字改对（同音、形近、多出的符号）。改不准的保持原样，不要瞎猜。\n"
    "3. 保留时间戳：合并后的行用第一条的起始时间和最后一条的结束时间。\n"
    "只输出 JSON 数组，每项 {\"start\": 数字, \"end\": 数字, \"text\": \"……\"}，不要解释。"
)


def dedup_rules(lines: list, ratio: float = 0.8) -> list:
    out = []
    for row in lines:
        if out:
            prev = out[-1]
            sim = difflib.SequenceMatcher(None, prev["text"], row["text"]).ratio()
            near = row["start"] - prev["end"] <= 2.0
            if sim >= ratio and near:
                if len(row["text"]) > len(prev["text"]):
                    prev["text"] = row["text"]
                prev["end"] = row["end"]
                continue
        out.append(dict(row))
    return out


def polish(lines: list) -> list:
    body = "\n".join(f"[{r['start']:.1f}-{r['end']:.1f}] {r['text']}" for r in lines)
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    payload = {
        "model": cfg.get("model", "deepseek-chat"),
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": body}],
        "temperature": 0.2,
        "max_tokens": 3000,
    }
    req = urllib.request.Request(
        cfg.get("base_url", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    raw = (data["choices"][0]["message"].get("content") or "").strip()
    a, b = raw.find("["), raw.rfind("]")
    if a < 0 or b < 0:
        print("模型没给 JSON，保留规则去重结果")
        return lines
    fixed = json.loads(raw[a:b + 1])
    out = []
    for item in fixed:
        out.append({"start": float(item.get("start", 0)),
                    "end": float(item.get("end", 0)),
                    "text": str(item.get("text", "")).strip()})
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("subs")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-model", action="store_true")
    args = ap.parse_args()

    src = Path(args.subs)
    out = Path(args.out) if args.out else src.with_name("subs_clean.json")
    data = json.loads(src.read_text(encoding="utf-8"))
    raw_lines = data.get("lines", [])

    step1 = dedup_rules(raw_lines)
    print(f"rule dedup: {len(raw_lines)} -> {len(step1)}")

    step2 = step1 if args.no_model else polish(step1)
    print(f"model polish: {len(step1)} -> {len(step2)}")

    data["lines"] = step2
    data["raw_lines"] = len(raw_lines)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"out: {out}")
    for r in step2[:10]:
        print(f"  [{r['start']:6.1f}-{r['end']:6.1f}] {r['text']}")


if __name__ == "__main__":
    main()
