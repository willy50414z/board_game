#!/usr/bin/env python3
"""Locally extract OCR/layout/visual-region candidates with PP-StructureV3.

Output contract: page_context.json, compatible with 03_ollama_qwen3vl_analyze.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
from paddleocr import PPStructureV3


VISUAL_LABEL_HINTS = ("image", "figure", "chart", "picture", "illustration", "seal")


def to_jsonable(value: Any) -> Any:
    """Convert common NumPy/Paddle values into JSON-compatible Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_bbox(raw_bbox: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if raw_bbox is None:
        return None
    values = to_jsonable(raw_bbox)
    # PP-StructureV3 block_bbox is normally [x1, y1, x2, y2].
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in values[:4]]
    except (TypeError, ValueError):
        return None
    left = max(0, min(width, int(round(min(x1, x2)))))
    top = max(0, min(height, int(round(min(y1, y2)))))
    right = max(0, min(width, int(round(max(x1, x2)))))
    bottom = max(0, min(height, int(round(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def parse_pic_to_json(image_path: Path, output_dir: Path, device: str = "cpu") -> Path:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "visual_regions"
    crops_dir.mkdir(exist_ok=True)

    # Use CPU for the first AMD/ROCm MVP to avoid assuming Paddle's GPU backend
    # is compatible with the Ollama ROCm setup.
    pipeline = PPStructureV3(
        lang="chinese_cht",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        device=device,
        enable_mkldnn=False,
    )

    outputs = pipeline.predict(input=str(image_path))
    if not outputs:
        raise RuntimeError("PP-StructureV3 沒有回傳分析結果")

    source_image = Image.open(image_path).convert("RGB")
    width, height = source_image.size

    paragraphs: list[dict[str, Any]] = []
    visual_regions: list[dict[str, Any]] = []
    all_layout_blocks: list[dict[str, Any]] = []
    markdown_parts: list[str] = []

    for page_result_index, result in enumerate(outputs, start=1):
        raw = getattr(result, "json", {}) or {}
        parsing_blocks = raw.get("parsing_res_list", []) if isinstance(raw, dict) else []

        md_info = getattr(result, "markdown", {}) or {}
        md_text = md_info.get("markdown_texts", "") if isinstance(md_info, dict) else ""
        if isinstance(md_text, list):
            md_text = "\n".join(str(x) for x in md_text)
        if md_text:
            markdown_parts.append(str(md_text))

        for block_index, block in enumerate(parsing_blocks, start=1):
            label = str(block.get("block_label") or "unknown")
            text = str(block.get("block_content") or "").strip()
            block_id = block.get("block_id", block_index)
            paragraph_id = f"P{page_result_index:02d}_{block_id}"
            bbox = to_jsonable(block.get("block_bbox"))

            all_layout_blocks.append(
                {
                    "id": paragraph_id,
                    "label": label,
                    "text": text,
                    "bbox": bbox,
                    "block_order": block.get("block_order"),
                }
            )

            if text:
                paragraphs.append(
                    {
                        "id": paragraph_id,
                        "text": text,
                        "label": label,
                        "bbox": bbox,
                    }
                )

            normalized_label = label.lower()
            if any(hint in normalized_label for hint in VISUAL_LABEL_HINTS):
                crop_box = normalize_bbox(block.get("block_bbox"), width, height)
                if crop_box:
                    crop = source_image.crop(crop_box)
                    crop_name = f"region_{len(visual_regions) + 1:03d}.jpg"
                    crop_path = crops_dir / crop_name
                    crop.save(crop_path, format="JPEG", quality=92)
                    visual_regions.append(
                        {
                            "id": f"V{len(visual_regions) + 1:04d}",
                            "label": label,
                            "caption": text,
                            "crop_path": str(crop_path.resolve()),
                            "related_paragraph_ids": [],
                            "bbox": list(crop_box),
                        }
                    )

        # Save full PP-StructureV3 response for inspection.
        raw_json_path = output_dir / f"paddle_structure_{page_result_index:02d}.json"
        raw_json_path.write_text(
            json.dumps(to_jsonable(raw), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            result.save_to_img(save_path=str(output_dir / "visualized"))
        except Exception:
            # Visualization is optional; raw and normalized results are primary.
            pass

    ocr_text = "\n\n".join(markdown_parts)
    if not ocr_text:
        ocr_text = "\n".join(p["text"] for p in paragraphs)

    # Fallback: if layout parser did not expose paragraph blocks, split Markdown.
    if not paragraphs and ocr_text.strip():
        for index, line in enumerate(ocr_text.splitlines(), start=1):
            line = line.strip()
            if line:
                paragraphs.append({"id": f"P{index:04d}", "text": line})

    context = {
        "schema_version": "1.0",
        "source_image": str(image_path.resolve()),
        "ocr_provider": "paddleocr-pp-structure-v3",
        "page_count": len(outputs),
        "ocr_text": ocr_text,
        "paragraphs": paragraphs,
        "tables": [],
        "layout_blocks": all_layout_blocks,
        "visual_regions": visual_regions,
    }

    context_path = output_dir / "page_context.json"
    context_path.write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "page.md").write_text(ocr_text, encoding="utf-8")

    print(f"OCR paragraphs/blocks: {len(paragraphs)}")
    print(f"Visual regions cropped: {len(visual_regions)}")
    print(f"Context: {context_path}")
    return context_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用本地 PaddleOCR PP-StructureV3 解析一張桌遊規則書頁面"
    )
    parser.add_argument("image", type=Path, help="單頁照片，例如 page_001.jpg")
    parser.add_argument("--out", type=Path, default=Path("output_paddle"))
    parser.add_argument("--device", default="cpu", help="先用 cpu；確認後再測 gpu")
    args = parser.parse_args()
    parse_pic_to_json(args.image, args.out, args.device)


if __name__ == "__main__":
    main()
