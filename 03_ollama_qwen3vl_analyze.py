#!/usr/bin/env python3
"""Analyze one photographed rulebook page using OCR context + Ollama Qwen3-VL.

Consumes page_context.json created by either 01_azure_document_intelligence.py
or paddleocr_pp_structure.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ollama import Client


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "page_type": {
            "type": "string",
            "enum": [
                "cover", "components", "setup", "turn_rules", "scoring",
                "example", "reference", "credits", "other", "unclear"
            ],
        },
        "page_summary": {"type": "string"},
        "game_metadata": {
            "type": "object",
            "properties": {
                "game_name": {"type": "string"},
                "publisher": {"type": "string"},
                "player_count": {"type": "string"},
                "play_time": {"type": "string"},
                "recommended_age": {"type": "string"},
            },
            "required": [
                "game_name", "publisher", "player_count", "play_time",
                "recommended_age"
            ],
            "additionalProperties": False,
        },
        "rules_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule": {"type": "string"},
                    "source_paragraph_ids": {
                        "type": "array", "items": {"type": "string"}
                    },
                },
                "required": ["rule", "source_paragraph_ids"],
                "additionalProperties": False,
            },
        },
        "visual_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "visual_region_id": {"type": "string"},
                    "retention": {
                        "type": "string",
                        "enum": ["keep", "review", "not_needed"],
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "setup_diagram", "turn_sequence", "component_reference",
                            "game_board_example", "card_example", "scoring_example",
                            "icon_legend", "decorative_art", "logo", "other", "unclear"
                        ],
                    },
                    "visual_description": {"type": "string"},
                    "rule_explanation": {"type": "string"},
                    "linked_paragraph_ids": {
                        "type": "array", "items": {"type": "string"}
                    },
                    "confidence": {"type": "number"},
                    "needs_review": {"type": "boolean"},
                },
                "required": [
                    "visual_region_id", "retention", "category",
                    "visual_description", "rule_explanation",
                    "linked_paragraph_ids", "confidence", "needs_review"
                ],
                "additionalProperties": False,
            },
        },
        "uncertain_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "page_type", "page_summary", "game_metadata", "rules_points",
        "visual_assessments", "uncertain_points"
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """
你是桌遊說明書的文件理解助手。請只根據輸入的頁面圖片和 OCR 文字分析。

規則：
1. 不要猜測看不清楚的名稱、數量、玩家人數或遊戲時間；未知欄位填空字串。
2. 不要自行發明規則。rules_points 必須有對應的 OCR 段落 ID。
3. 圖片是否保留：有助於理解設置、回合、配件、卡牌、計分、圖示的標記 keep。
4. 純裝飾、重複 Logo 或沒有知識價值的圖形可標記 not_needed。
5. 無法確認用途時標記 review，並設定 needs_review=true。
6. linked_paragraph_ids 只能使用提供的 paragraph IDs；沒有證據就回傳空陣列。
7. visual_region_id 只能使用提供的 visual region IDs。
8. confidence 是模型自評，不是經過校準的準確率。
9. 請使用繁體中文，並符合 JSON Schema。
""".strip()


def resolve_path(raw_path: str, context_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (context_dir / path).resolve()


def validate_ids(result: dict[str, Any], paragraphs: list[dict[str, Any]], regions: list[dict[str, Any]]) -> None:
    valid_paragraph_ids = {str(p["id"]) for p in paragraphs if p.get("id") is not None}
    valid_region_ids = {str(r["id"]) for r in regions if r.get("id") is not None}

    for rule in result.get("rules_points", []):
        invalid = [pid for pid in rule.get("source_paragraph_ids", []) if pid not in valid_paragraph_ids]
        if invalid:
            rule["source_paragraph_ids"] = [pid for pid in rule["source_paragraph_ids"] if pid in valid_paragraph_ids]
            result["uncertain_points"].append(f"規則引用不存在的段落 ID：{invalid}")

    for assessment in result.get("visual_assessments", []):
        if assessment.get("visual_region_id") not in valid_region_ids:
            assessment["needs_review"] = True
            result["uncertain_points"].append(
                f"模型回傳不存在的 visual_region_id：{assessment.get('visual_region_id')}"
            )
        invalid = [pid for pid in assessment.get("linked_paragraph_ids", []) if pid not in valid_paragraph_ids]
        if invalid:
            assessment["linked_paragraph_ids"] = [
                pid for pid in assessment["linked_paragraph_ids"] if pid in valid_paragraph_ids
            ]
            assessment["needs_review"] = True
            result["uncertain_points"].append(f"圖片引用不存在的段落 ID：{invalid}")


def run(context_path: Path, output_path: Path, max_regions: int = 6) -> Path:
    if not context_path.is_file():
        raise FileNotFoundError(context_path)

    context = json.loads(context_path.read_text(encoding="utf-8"))
    context_dir = context_path.parent
    page_image_raw = context.get("source_image")
    if not page_image_raw:
        raise ValueError("page_context.json 缺少 source_image")
    page_image = resolve_path(page_image_raw, context_dir)
    if not page_image.is_file():
        raise FileNotFoundError(f"找不到原始頁面圖片：{page_image}")

    paragraphs = context.get("paragraphs", [])
    visual_regions = context.get("visual_regions", [])
    selected_regions = visual_regions[:max_regions]

    # Image 1 is always the full page; following images are extracted visual crops.
    image_paths = [str(page_image)]
    image_manifest = ["image 1 = 完整說明書頁面"]
    model_regions: list[dict[str, Any]] = []

    if selected_regions:
        for index, region in enumerate(selected_regions, start=2):
            crop_raw = region.get("crop_path")
            if not crop_raw:
                continue
            crop_path = resolve_path(crop_raw, context_dir)
            if crop_path.is_file():
                image_paths.append(str(crop_path))
                image_manifest.append(f"image {index} = visual_region_id {region['id']} ({region.get('label', '')})")
                model_regions.append(region)
    else:
        # When the layout parser did not identify separate figure crops, ask the
        # model to assess the full page as one page-level visual candidate.
        model_regions = [{"id": "PAGE", "label": "full_page", "crop_path": str(page_image)}]
        image_manifest.append("image 1 同時作為 visual_region_id PAGE 的整頁候選")

    # Bound the OCR text sent to the VLM; preserve paragraph IDs.
    paragraph_payload = [
        {"id": str(p.get("id", "")), "text": str(p.get("text", ""))[:2000]}
        for p in paragraphs
    ]
    ocr_text = str(context.get("ocr_text", ""))[:14000]

    prompt = f"""
請分析這張桌遊說明書頁面，並依照系統指示回傳結構化資料。

OCR 來源：{context.get('ocr_provider', 'unknown')}

影像順序：
{json.dumps(image_manifest, ensure_ascii=False, indent=2)}

候選段落（paragraph IDs 不可自行改寫）：
{json.dumps(paragraph_payload, ensure_ascii=False, indent=2)}

完整 OCR 文字：
{ocr_text}

可評估的 visual_region IDs：
{json.dumps([r['id'] for r in model_regions], ensure_ascii=False)}

只分析本頁可見內容。圖片與文字對應必須有證據；如果只是同頁但無法確認語意關聯，不要建立連結。
""".strip()

    model = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b-8k")
    host = os.getenv("OLLAMA_HOST", "http://100.123.189.49:7401")
    client = Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
                "images": image_paths,
            },
        ],
        format=OUTPUT_SCHEMA,
        options={"temperature": 0.1,"num_ctx": 8192,},
        stream=False,
    )

    result = json.loads(response.message.content)
    result["source"] = {
        "page_context": str(context_path.resolve()),
        "page_image": str(page_image),
        "ocr_provider": context.get("ocr_provider"),
        "model": model,
    }
    validate_ids(result, paragraphs, model_regions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 Ollama + Qwen3-VL 分析 OCR 已解析的桌遊說明書頁面"
    )
    parser.add_argument("page_context", type=Path, help="01 或 02 產生的 page_context.json")
    parser.add_argument("--out", type=Path, default=Path("analysis.json"))
    parser.add_argument("--max-regions", type=int, default=6)
    args = parser.parse_args()
    run(args.page_context, args.out, args.max_regions)


if __name__ == "__main__":
    main()
