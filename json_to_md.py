#!/usr/bin/env python3
"""把頁面文件 JSON 渲染成 Markdown（**僅供人工檢查/debug**）。

這個模組刻意不碰圖片、不碰 OCR、不碰模型：輸入只有 doc.json，輸出只有 md 字串。
所以人工修完 JSON 只要重跑本模組就能重新預覽；要進資料庫則完全不需要它。

用法：
    python json_to_md.py output/bbw_1/bbw_1.json -o output/bbw_1/bbw_1.md
    python json_to_md.py output/bbw_1/bbw_1.json            # 印到 stdout
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from page_document import load_document, normalize_document


# ---------------------------------------------------------------------------
# front matter
# ---------------------------------------------------------------------------

class _Dumper(yaml.SafeDumper):
    """數字陣列（bbox）壓成一列，可讀性與人工稿一致。"""


def _represent_sequence(dumper: yaml.SafeDumper, sequence: list[Any]):
    flow = bool(sequence) and all(isinstance(v, (int, float)) for v in sequence)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", sequence, flow_style=flow)


_Dumper.add_representer(list, _represent_sequence)


def render_front_matter(doc: dict[str, Any]) -> str:
    front: dict[str, Any] = {
        "id": doc.get("id"),
        "title": doc.get("title"),
        "type": doc.get("type") or _page_type_zh(doc.get("page_type")),
        "page": doc.get("page"),
        "language": doc.get("language"),
        "page_type": doc.get("page_type"),
        "source_image": doc.get("source_image"),
        "page_image": doc.get("page_image"),
        "image_size": doc.get("image_size"),
        "capture": doc.get("capture"),
        "ocr_provider": doc.get("ocr_provider"),
        "proofread": doc.get("proofread"),
        "proofread_date": doc.get("proofread_date"),
        "summary": doc.get("summary"),
    }
    if doc.get("subjects"):
        front["subjects"] = [_clean(subject) for subject in doc["subjects"]]
    front["other_artwork"] = doc.get("other_artwork") or []
    front["images"] = [_image_front(i) for i in doc.get("images") or []]
    front["image_source"] = _clean(doc.get("image_source") or {})

    quality = doc.get("quality") or {}
    front["quality"] = _clean({
        "needs_review": quality.get("needs_review"),
        "uncertain_points": quality.get("uncertain_points") or [],
        "warnings": quality.get("warnings") or [],
    })
    front["pipeline"] = _clean({
        "generator": (doc.get("provenance") or {}).get("generator"),
        "model": (doc.get("provenance") or {}).get("model"),
        "created_at": (doc.get("provenance") or {}).get("created_at"),
    })

    body = yaml.dump(
        {k: v for k, v in front.items() if v not in (None, "", [], {})},
        Dumper=_Dumper, allow_unicode=True, sort_keys=False, default_flow_style=False, width=4096,
    )
    return f"---\n{body}---\n"


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v not in (None, "", [], {})}


def _image_front(image: dict[str, Any]) -> dict[str, Any]:
    return _clean({
        "id": image.get("id"),
        "file": image.get("file"),
        "section": image.get("section"),
        "subject": image.get("subject"),
        "caption": image.get("caption"),
        "alt": image.get("alt"),
        "identifier_color": image.get("identifier_color"),
        "non_text_cues": image.get("non_text_cues"),
        "retention": image.get("retention"),
        "verified": image.get("verified") or None,
        "bbox": image.get("bbox"),
    })


_PAGE_TYPE_ZH = {
    "cover": "封面／首頁", "components": "配件總覽", "setup": "遊戲設置",
    "turn_rules": "回合規則", "scoring": "計分與結束", "example": "範例",
    "reference": "速查表", "credits": "版權資訊", "other": "其他", "unclear": "未判定",
}


def _page_type_zh(page_type: Any) -> str:
    return _PAGE_TYPE_ZH.get(str(page_type or ""), "")


# ---------------------------------------------------------------------------
# 正文
# ---------------------------------------------------------------------------

def render_body(doc: dict[str, Any]) -> str:
    images_by_id = {str(i.get("id")): i for i in doc.get("images") or []}
    lines: list[str] = []
    title = str(doc.get("title") or "").strip()
    if title:
        lines += [f"# {title}", ""]
    summary = str(doc.get("summary") or "").strip()
    if summary:
        lines += [summary, ""]

    used: set[str] = set()
    for block in doc.get("blocks") or []:
        block_type = str(block.get("type") or "paragraph")
        text = str(block.get("text") or "").strip()
        image = images_by_id.get(str(block.get("image_id") or ""))

        if block_type == "heading" and text:
            level = min(4, max(1, int(block.get("level") or 2)))
            lines += [f"{'#' * level} {text}", ""]
        elif block_type == "list_item" and text:
            bullet = "" if text.startswith(("-", "*", "·")) else "- "
            lines.append(f"{bullet}{text}")
        elif block_type == "note" and text:
            lines.append(f"> {text}")
        elif block_type == "table_row" and text:
            lines.append(f"`|` {text}")
        elif text:
            lines.append(f"**{text}**" if block.get("style") == "bold" else text)

        # 圖片列與圖說（圖說讓 md 與 DB 讀到同一份非文字資訊）
        if image is not None and image.get("file"):
            used.add(str(image.get("id")))
            lines.append("")
            alt = image.get("alt") or image.get("caption") or image.get("id")
            lines.append(f"![{alt}]({image['file']})")
            caption = str(block.get("caption") or image.get("caption") or "").strip()
            if caption:
                lines.append(f"*{caption}*")
        if block_type not in ("list_item",) and (text or image is not None):
            lines.append("")

    # 有檔卻沒綁進正文的圖片：還是渲染並在尾端標記，避免 debug 時看不到
    orphans = [i for i in doc.get("images") or [] if str(i.get("id")) not in used and i.get("file")]
    if orphans:
        lines += ["<!-- 以下圖片尚未綁定到任何小節（請在 JSON 的 blocks 加 image_id） -->", ""]
        for image in orphans:
            alt = image.get("alt") or image.get("caption") or image.get("id")
            lines += [f"![{alt}]({image['file']})", ""]

    return _tidy("\n".join(lines))


def _tidy(text: str) -> str:
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


def render_footer(doc: dict[str, Any]) -> str:
    """版面／判讀備註放 HTML 註解：不影響渲染，但留在檔內（§5）。"""
    notes: list[str] = []
    if str(doc.get("layout_notes") or "").strip():
        notes += ["版面備註（供資料庫還原版面使用）："]
        for line in str(doc["layout_notes"]).strip().splitlines():
            notes.append("- " + re.sub(r"^\s*[-*]\s*", "", line))
    quality = doc.get("quality") or {}
    if quality.get("uncertain_points"):
        if notes:
            notes.append("")
        notes.append("待人工確認（自動判讀不確定處）：")
        notes += [f"- {x}" for x in quality["uncertain_points"]]
    if quality.get("warnings"):
        if notes:
            notes.append("")
        notes.append("自動驗證警告：")
        notes += [f"- {x}" for x in quality["warnings"]]
    if not notes:
        return ""
    return "\n<!--\n" + "\n".join(notes) + "\n-->\n"


def render_markdown(doc: dict[str, Any]) -> str:
    """文件 JSON -> Markdown 字串（LF、UTF-8 可直寫）。"""
    normalize_document(doc)
    parts = [render_front_matter(doc), "", render_body(doc)]
    footer = render_footer(doc)
    if footer:
        parts += ["", footer]
    return "\n".join(p for p in parts if p is not None)


def write_markdown(doc: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(doc), encoding="utf-8", newline="\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="頁面文件 JSON -> Markdown（debug 預覽用）")
    parser.add_argument("document", type=Path, help="parse 階段產出的 <doc>.json")
    parser.add_argument("-o", "--out", type=Path, default=None, help="輸出 md；預設與 json 同槽名的 .md")
    parser.add_argument("--check", action="store_true", help="順就跑一輪文件驗證並列問題")
    args = parser.parse_args()

    doc = load_document(args.document)
    out = args.out or args.document.with_suffix(".md")
    write_markdown(doc, out)
    print(f"Markdown: {out}")

    if args.check:
        from page_document import validate_document

        issues = validate_document(doc, args.document.parent)
        for issue in issues:
            print(issue)
        if not issues:
            print("驗證通過（仍需人工複核 §7）")
        sys.exit(1 if any(i.startswith("E") for i in issues) else 0)


if __name__ == "__main__":
    main()
