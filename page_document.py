#!/usr/bin/env python3
"""頁面文件（Page Document）的資料契約、驗證與向量資料庫匯出。

本模組是整條 workflow 的**唯一資料真相來源（source of truth）**：

    圖片 --(mannual_parser parse)--> <doc>.json
                                      |
                                      +--> json_to_md.render_markdown()  -> md（僅 debug 預覽）
                                      +--> to_vector_records()           -> 向量資料庫記錄

設計原則（對應 mannual_to_md.md）：
- md 是可丟棄的渲染結果；任何「md 有、JSON 沒有」的資訊都是 bug。
- 非文字資訊（顏色、箭頭、序號、數量）必須以文字欄位存在，不能只活在圖片裡（§7.2）。
- 圖片一律記錄 bbox（矯正後座標系）與 caption/alt，DB 端可直接抽成圖說（§4.5）。
- 自動化產物必須明確標示未經人工校對，避免被當成已驗證規則（§2）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "2.0"

PAGE_TYPES = (
    "cover", "components", "setup", "turn_rules", "scoring",
    "example", "reference", "credits", "other", "unclear",
)

#: type=image 的 block 代表「正文中獨立的一個圖片位」，必須帶 image_id。
BLOCK_TYPES = ("heading", "paragraph", "list_item", "image", "table_row", "note")

IMAGE_RETENTION = ("keep", "review", "not_needed")

_REQUIRED_TOP_LEVEL = ("id", "title", "blocks", "images", "image_source")


# ---------------------------------------------------------------------------
# 建立／正規化
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_document(
    doc_id: str,
    title: str = "",
    *,
    page: int | None = None,
    source_image: str = "",
    language: str = "zh-TW",
) -> dict[str, Any]:
    """建立一份帶預設值、可直接序列化的文件骨架。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": doc_id,
        "title": title,
        "type": "",
        "page": page,
        "language": language,
        "page_type": "unclear",
        "summary": "",
        "source_image": source_image,
        "page_image": "",
        "image_size": "",
        "capture": "",
        "ocr_provider": "",
        "proofread": "尚未經人工校對（自動化產出）",
        "proofread_date": None,
        "subjects": [],
        "images": [],
        "other_artwork": [],
        "blocks": [],
        "layout_notes": "",
        "image_source": {"file": source_image, "coordinate_space": "", "processing": ""},
        "quality": {"uncertain_points": [], "warnings": [], "needs_review": True},
        "provenance": {"created_at": utc_now(), "steps": []},
    }


def normalize_document(doc: dict[str, Any]) -> dict[str, Any]:
    """補齊缺漏欄位、統一型別，讓下游（md／DB）不必再防呆。就地修改並回傳。"""
    defaults = new_document(str(doc.get("id") or "page"), str(doc.get("title") or ""))
    for key, value in defaults.items():
        doc.setdefault(key, value)

    doc["schema_version"] = str(doc.get("schema_version") or SCHEMA_VERSION)
    doc["blocks"] = [normalize_block(b, i) for i, b in enumerate(doc.get("blocks") or []) if isinstance(b, dict)]
    doc["images"] = [normalize_image(im, i) for i, im in enumerate(doc.get("images") or []) if isinstance(im, dict)]
    doc["subjects"] = [s for s in doc.get("subjects") or [] if isinstance(s, dict)]
    doc["other_artwork"] = [str(a) for a in doc.get("other_artwork") or [] if str(a).strip()]

    quality = doc.setdefault("quality", {})
    quality.setdefault("uncertain_points", [])
    quality.setdefault("warnings", [])
    # 自動化產物一律標記 needs_review，避免未經校對的規則直接入庫
    quality["needs_review"] = bool(quality.get("needs_review", True))

    image_source = doc.setdefault("image_source", {})
    image_source.setdefault("file", doc.get("source_image", ""))
    return doc


def normalize_block(block: dict[str, Any], index: int) -> dict[str, Any]:
    block_type = str(block.get("type") or "paragraph")
    if block_type not in BLOCK_TYPES:
        block_type = "paragraph"
    block["type"] = block_type
    block["id"] = str(block.get("id") or f"B{index + 1:03d}")
    block["text"] = str(block.get("text") or "").strip()
    if block_type == "heading":
        try:
            block["level"] = min(4, max(1, int(block.get("level") or 2)))
        except (TypeError, ValueError):
            block["level"] = 2
    else:
        block.pop("level", None)
    block["image_id"] = str(block.get("image_id")) if block.get("image_id") else None
    block["caption"] = str(block.get("caption")).strip() if block.get("caption") else None
    block["bbox"] = clean_bbox(block.get("bbox"))
    block["source_paragraph_ids"] = [str(x) for x in block.get("source_paragraph_ids") or []]
    block["notes"] = str(block.get("notes")).strip() if block.get("notes") else None
    return block


def normalize_image(image: dict[str, Any], index: int) -> dict[str, Any]:
    image["id"] = str(image.get("id") or f"img{index + 1}")
    image["file"] = str(image.get("file") or "")
    image["caption"] = str(image.get("caption") or "").strip()
    image["alt"] = str(image.get("alt") or image.get("caption") or "").strip()
    image["section"] = str(image.get("section") or "").strip()
    image["subject"] = str(image.get("subject") or "").strip()
    image["bbox"] = clean_bbox(image.get("bbox"))
    retention = str(image.get("retention") or "review")
    image["retention"] = retention if retention in IMAGE_RETENTION else "review"
    image["importance"] = str(image.get("importance") or "").strip()
    image["identifier_color"] = str(image.get("identifier_color") or "").strip()
    image["non_text_cues"] = [str(c) for c in image.get("non_text_cues") or [] if str(c).strip()]
    image["verified"] = bool(image.get("verified"))
    return image


def clean_bbox(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        values = [int(round(float(v))) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def parse_image_size(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", str(value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def save_document(doc: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_document(path: Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} 不是頁面文件物件")
    return doc


def add_step(doc: dict[str, Any], step: str, **info: Any) -> None:
    """記錄流程軌跡，方便日後追查「這份 JSON 怎麼來的」。"""
    entry: dict[str, Any] = {"step": step, "at": utc_now()}
    entry.update({k: v for k, v in info.items() if v is not None})
    doc.setdefault("provenance", {}).setdefault("steps", []).append(entry)


# ---------------------------------------------------------------------------
# 驗證（對應 mannual_to_md.md §6：寫成腳本跑，別用看的）
# ---------------------------------------------------------------------------

def validate_document(doc: dict[str, Any], base_dir: Path | None = None) -> list[str]:
    """回傳問題清單，空代表通過。"E "=錯誤、"W "=警告。"""
    issues: list[str] = []

    for key in _REQUIRED_TOP_LEVEL:
        if key not in doc:
            issues.append(f"E 缺少必要欄位: {key}")
    if str(doc.get("schema_version", "")) != SCHEMA_VERSION:
        issues.append(f"W schema_version={doc.get('schema_version')}，預期 {SCHEMA_VERSION}")

    blocks = doc.get("blocks") or []
    if not blocks:
        issues.append("E blocks 為空：頁面沒有任何文字內容")
    for block in blocks:
        if block.get("type") == "image" and not block.get("image_id"):
            issues.append(f"E 圖片區塊 {block.get('id')} 缺少 image_id")

    images = doc.get("images") or []
    seen_ids: set[str] = set()
    for image in images:
        image_id = str(image.get("id") or "")
        if not image_id:
            issues.append("E images[].id 是空的")
        elif image_id in seen_ids:
            issues.append(f"E 圖片 id 重複: {image_id}")
        seen_ids.add(image_id)

        file_name = str(image.get("file") or "")
        if not file_name:
            issues.append(f"E 圖片 {image_id} 沒有 file")
            continue
        if re.search(r"[\u4e00-\u9fff]", file_name):
            issues.append(f"E 圖片路徑含中文（DB/URL 不安全）: {file_name}")
        if base_dir is not None and not (base_dir / file_name).is_file():
            issues.append(f"E 圖片檔不存在: {file_name}")
        if not str(image.get("caption") or "").strip():
            issues.append(f"W 圖片 {image_id} 缺少 caption（DB 無法抽圖說）")

    # front matter images[] 與正文引用必須互為同一集合（§6）
    referenced = {str(b.get("image_id")) for b in blocks if b.get("image_id")}
    defined = {str(i.get("id")) for i in images}
    for missing in sorted(referenced - defined):
        issues.append(f"E 正文引用了未定義的圖片 id: {missing}")
    for orphan in sorted(defined - referenced):
        issues.append(f"E 圖片 {orphan} 有定義但正文未引用（渲染後會看不到）")

    size = parse_image_size(doc.get("image_size", ""))
    boxes: list[tuple[str, list[int]]] = []
    for image in images:
        bbox = clean_bbox(image.get("bbox"))
        if not bbox:
            issues.append(f"W 圖片 {image.get('id')} 沒有 bbox，無法回溯原圖位置")
            continue
        boxes.append((str(image.get("id")), bbox))
        if size:
            width, height = size
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
                issues.append(
                    f"E 圖片 {image.get('id')} bbox 超出 image_size: {bbox} / {width}x{height}"
                )

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _iou(boxes[i][1], boxes[j][1]) > 0.2:
                issues.append(f"W 圖片 bbox 重疊: {boxes[i][0]} 與 {boxes[j][0]}")

    # 段落數對照 OCR，確認沒有整行消失（§6）
    ocr_reference = str(doc.get("_ocr_text") or (doc.get("ocr_draft") or {}).get("text") or "")
    ocr_lines = [x for x in ocr_reference.splitlines() if x.strip()]
    if ocr_lines:
        block_lines = sum(1 for b in blocks if str(b.get("text") or "").strip())
        if block_lines < len(ocr_lines) * 0.5:
            issues.append(
                f"W 正文僅 {block_lines} 段，OCR 有 {len(ocr_lines)} 行，可能漏了整段"
            )

    if not str(doc.get("title") or "").strip():
        issues.append("W 沒有標題")
    if not str(doc.get("image_source", {}).get("coordinate_space") or "").strip():
        issues.append("W image_source.coordinate_space 未填寫，bbox 無法回溯")
    return issues


def errors_of(issues: Iterable[str]) -> list[str]:
    return [issue for issue in issues if issue.startswith("E")]


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if not inter:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


# ---------------------------------------------------------------------------
# 向量資料庫匯出（JSON -> DB；md 只是另一種渲染，DB 端不需要它）
# ---------------------------------------------------------------------------

def flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """多數向量庫的 metadata 只收純量／純量陣列，這裡先壓平並丢空值。"""
    flat: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, (list, tuple)):
            items = [v for v in value if isinstance(v, (str, int, float, bool))]
            if items:
                flat[key] = items
        elif isinstance(value, dict):
            parts = [f"{k}={v}" for k, v in value.items() if v not in (None, "", [])]
            if parts:
                flat[key] = "; ".join(parts)
    return flat


def image_mention(image: dict[str, Any]) -> str:
    """把圖片的非文字資訊寫成 embedding 讀得懂的句子（§7.2）。"""
    caption = image.get("caption") or image.get("alt") or image.get("id") or "圖片"
    details: list[str] = []
    for label, key in (("所屬小節", "section"), ("代表", "subject"), ("識別色", "identifier_color")):
        value = str(image.get(key) or "").strip()
        if value:
            details.append(f"{label}：{value}")
    if image.get("non_text_cues"):
        details.append("視覺線索：" + "；".join(str(c) for c in image["non_text_cues"]))
    text = f"[圖片] {caption}"
    if details:
        text += "（" + "；".join(details) + "）"
    return text


def _record(rid: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    meta = flatten_metadata(metadata)
    breadcrumb = "／".join(x for x in (meta.get("title"), meta.get("section")) if x)
    embedding_text = f"【{breadcrumb}】\n{text}" if breadcrumb else text
    return {"id": rid, "text": text, "text_for_embedding": embedding_text, "metadata": meta}


def to_vector_records(doc: dict[str, Any], *, include_image_records: bool = True) -> list[dict[str, Any]]:
    """把一份頁面文件拆成適合 RAG 的記錄。

    切塊邏輯：以「小節（heading）＋下屬內容」為一個 chunk，圖片以文字描述融入所屬
    chunk（這樣問規則時會連圖說一起檢索）；另外為每張關鍵圖片另外產一筆
    chunk_type=image 的記錄，支援「以圖查規則」。最後附一頁 overview 與 layout。
    """
    doc_id = str(doc.get("id") or "page")
    doc_title = str(doc.get("title") or "")
    base_meta: dict[str, Any] = {
        "doc_id": doc_id,
        "title": doc_title,
        "page": doc.get("page"),
        "page_type": doc.get("page_type"),
        "language": doc.get("language"),
        "source_image": doc.get("source_image"),
        "ocr_provider": doc.get("ocr_provider"),
        "proofread": doc.get("proofread"),
        "needs_review": bool((doc.get("quality") or {}).get("needs_review", True)),
        "images": [],
    }
    records: list[dict[str, Any]] = []
    images_by_id = {str(i.get("id")): i for i in doc.get("images") or []}

    # 1) overview：標題＋摘要＋結構化主體（勝負條件之類最常被查詢）
    overview_lines: list[str] = [doc_title] if doc_title else []
    if doc.get("summary"):
        overview_lines.append(str(doc["summary"]))
    for subject in doc.get("subjects") or []:
        details: list[str] = []
        if subject.get("key"):
            details.append(f"代號 {subject['key']}")
        if subject.get("win_condition"):
            details.append(f"獲勝條件：{subject['win_condition']}")
        identifier = subject.get("identifier") or subject.get("identifier_color")
        if identifier:
            details.append(f"識別：{identifier}")
        line = f"- {subject.get('name', '')}".rstrip()
        if details:
            line += "：" + "；".join(details)
        if line.strip("- "):
            overview_lines.append(line)
    if overview_lines:
        meta = {**base_meta, "chunk_type": "overview", "section": "頁面總覽"}
        records.append(_record(f"{doc_id}#overview", "\n".join(overview_lines), meta))

    # 2) 依 heading 切塊；圖片與圖說融入所屬小節
    section_index = 0
    section_title = ""
    buffer: list[str] = []
    buffer_images: list[str] = []
    buffer_blocks: list[str] = []

    def flush() -> None:
        if not buffer and not buffer_images:
            return
        text = "\n".join(x for x in buffer if x).strip()
        if not text:
            return
        meta = {
            **base_meta,
            "chunk_type": "section",
            "section": section_title,
            "images": list(buffer_images),
            "block_ids": list(buffer_blocks),
        }
        records.append(_record(f"{doc_id}#S{section_index:02d}", text, meta))

    for block in doc.get("blocks") or []:
        if block.get("type") == "heading":
            flush()
            section_index += 1
            section_title = str(block.get("text") or "")
            level = int(block.get("level") or 2)
            buffer = [f"{'#' * min(level, 6)} {section_title}"]
            buffer_images, buffer_blocks = [], []
            continue

        text = str(block.get("text") or "").strip()
        if block.get("type") == "list_item" and text and not text.startswith(("-", "*", "·")):
            text = f"- {text}"
        if text:
            buffer.append(text)
            buffer_blocks.append(str(block.get("id")))
        image = images_by_id.get(str(block.get("image_id") or ""))
        if image is not None:
            buffer.append(image_mention(image))
            if block.get("caption"):
                buffer.append(str(block["caption"]))
            if image.get("file"):
                buffer_images.append(str(image["file"]))
    flush()

    # 3) 版面備註也入庫（日後要還原版面時用得到）
    if str(doc.get("layout_notes") or "").strip():
        meta = {**base_meta, "chunk_type": "layout", "section": "版面備註"}
        records.append(_record(f"{doc_id}#layout", str(doc["layout_notes"]).strip(), meta))

    # 4) 關鍵圖片自己的記錄
    if include_image_records:
        for image in doc.get("images") or []:
            if str(image.get("retention")) == "not_needed":
                continue
            lines = [image_mention(image)]
            if image.get("alt"):
                lines.append(f"圖說：{image['alt']}")
            meta = {
                **base_meta,
                "chunk_type": "image",
                "section": str(image.get("section") or doc_title),
                "image_id": str(image.get("id")),
                "image_file": str(image.get("file")),
                "image_bbox": list(image.get("bbox") or []),
                "images": [str(image.get("file"))] if image.get("file") else [],
            }
            records.append(_record(f"{doc_id}#img-{image.get('id')}", "\n".join(lines), meta))

    return records


def write_records_jsonl(records: Iterable[dict[str, Any]], path: Path) -> Path:
    """向量庫匯入檔：一行一顆 record，方便增補與重跑（upsert 依 id）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_records_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同檔重複 run 時依 id 保留最後一份（向量庫 upsert 語意）。"""
    seen: dict[str, dict[str, Any]] = {}
    for record in records:
        seen[str(record.get("id"))] = record
    return list(seen.values())
