#!/usr/bin/env python3
"""Analyze one photographed rulebook page using OCR context + Ollama Qwen3-VL.

This updated script shifts the burden of visual element detection to Qwen3-VL.
It passes the full page image to the VLM, asking it to locate all visual elements,
generate Markdown with placeholders, and then the script crops the images
and composes the final Markdown file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image
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
        "markdown": {
            "type": "string",
            "description": "將頁面文字轉寫為 Markdown，並在對應的圖片位置插入佔位符，如 `![圖片描述](__CROP_0__)`。"
        },
        "visual_elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "placeholder_id": {
                        "type": "string",
                        "description": "對應 Markdown 中的佔位符，例如 `__CROP_0__`"
                    },
                    "description": {"type": "string", "description": "對該視覺元素的簡短描述"},
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "精確邊界框 [x_min, y_min, x_max, y_max]"
                    },
                    "retention": {
                        "type": "string",
                        "enum": ["keep", "review", "not_needed"],
                        "description": "評估該圖片是否對理解規則有幫助"
                    }
                },
                "required": ["placeholder_id", "description", "bbox_2d", "retention"],
                "additionalProperties": False,
            },
        },
        "uncertain_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "page_type", "page_summary", "game_metadata", "rules_points",
        "markdown", "visual_elements", "uncertain_points"
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """
你是桌遊說明書的專業文件解析引擎。請根據輸入的頁面圖片和 OCR 參考文字進行分析。

【任務零：版面順序核對】
輸入的 paragraphs 列表來自 OCR 自動切分，排列順序不一定符合實際閱讀順序（尤其多欄版面）。請：
1. 先觀察圖片判斷版面結構（單欄 / 雙欄 / 多欄 / 網格）。
2. 中文閱讀順序：由上而下；多欄時先讀完左欄再讀右欄。
3. paragraphs 只用來取得「文字內容」與「ID」，**不要依賴其陣列順序**。
   若段落有提供 bbox，可作為輔助依據；沒有 bbox 就以圖片為準。
4. 你的 markdown 順序即為你判定的閱讀順序。
5. 若同一語意塊被 OCR 切成多段，請合併成一段，並在 rules_points 引用所有對應 ID。
6. 輸出前自我檢查：markdown 的段落順序是否與圖片實際版面一致？

【任務一：提取結構化規則】
1. 提取遊戲基本資訊與規則點。不要猜測看不清楚的名稱、數量、玩家人數或遊戲時間；未知欄位填空字串。
2. rules_points 必須有對應的 OCR 段落 ID，不要自行發明規則。

【任務二：生成帶圖片的 Markdown】
1. 將頁面文字按閱讀順序轉寫為 Markdown。
2. 識別頁面上所有獨立的「非文本視覺元素」（圖標、插畫、徽章、裝飾圖形、照片等）。
3. 在 Markdown 中對應的語義位置插入佔位符 `![描述](__CROP_N__)`，N 為從 0 開始的遞增數字。
4. 評估每個視覺元素：有助於理解規則的標記 `keep`，純裝飾或無知識價值的標記 `not_needed`，不確定則標記 `review`。

【任務三：輸出座標】
在 `visual_elements` 陣列中，為每個佔位符輸出精確的邊界框 `bbox_2d`。
請輸出 **0 到 1000 的歸一化座標**（相對於圖片寬高的比例乘以 1000），格式為 [x_min, y_min, x_max, y_max]。
請嚴格依照 JSON Schema 輸出，不要包含任何額外解釋。
""".strip()


def resolve_path(raw_path: str, context_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (context_dir / path).resolve()


def crop_and_replace(markdown_text: str, visual_elements: list[dict], page_image_path: Path, output_dir: Path) -> str:
    """根據模型返回的 BBox 裁切圖片，並替換 Markdown 中的佔位符。"""
    if not visual_elements:
        return markdown_text

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    source_image = Image.open(page_image_path).convert("RGB")
    width, height = source_image.size

    final_markdown = markdown_text

    for element in visual_elements:
        placeholder = element.get("placeholder_id")
        bbox = element.get("bbox_2d")

        if not placeholder or not bbox or len(bbox) != 4:
            continue

        # 如果判定不需要這張圖，就從 Markdown 中移除佔位符
        if element.get("retention") == "not_needed":
            final_markdown = final_markdown.replace(placeholder, "")
            continue

        try:
            # 1. 座標轉換：判斷是 0-1000 的歸一化座標，還是絕對像素
            x1, y1, x2, y2 = [float(v) for v in bbox]

            # 如果座標最大值小於等於 1000，視為 0-1000 的歸一化座標，需要按比例放大
            if max(x1, y1, x2, y2) <= 1000:
                x1 = (x1 / 1000.0) * width
                y1 = (y1 / 1000.0) * height
                x2 = (x2 / 1000.0) * width
                y2 = (y2 / 1000.0) * height

            # 確保是整數且不超出圖片邊界
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # 如果裁切區域太小，可能是模型幻覺，直接跳過
            if (x2 - x1) < 15 or (y2 - y1) < 15:
                print(f"Warning: BBox too small for {placeholder}, skipped.")
                final_markdown = final_markdown.replace(placeholder, "")
                continue

            # 2. 裁切並儲存圖片
            crop_img = source_image.crop((x1, y1, x2, y2))
            crop_filename = f"{placeholder.strip('_')}.jpg"
            crop_path = crops_dir / crop_filename
            crop_img.save(crop_path, format="JPEG", quality=95)

            # 3. 替換 Markdown 佔位符為實際圖片
            # 注意：這裡直接替換 placeholder (例如 __CROP_0__)，不需要管前面的描述文字是否完全一致
            image_md = f"![{element.get('description', '')}](crops/{crop_filename})"
            final_markdown = final_markdown.replace(placeholder, image_md)

            print(f"Successfully cropped {placeholder} to {crop_path}")

        except Exception as e:
            print(f"Error cropping image for {placeholder}: {e}")
            final_markdown = final_markdown.replace(placeholder, "")

    # 清理未被替換的殘留佔位符（如果有）
    final_markdown = re.sub(r'!\[.*?\]\(__CROP_\d+__\)', '', final_markdown)
    final_markdown = re.sub(r'__CROP_\d+__', '', final_markdown)

    return final_markdown


def parse_paadle_to_md(context_path: Path, output_md_path: Path, output_json_path: Path) -> None:
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

    paragraph_payload = [
        {
            "id": str(p.get("id", "")),
            "text": str(p.get("text", ""))[:1000],
            "bbox": p.get("bbox"),  # None 也可以，讓 LLM 知道沒有
        }
        for p in paragraphs
    ]
    ocr_text = str(context.get("ocr_text", ""))[:10000]

    prompt = f"""
請分析這張桌遊說明書頁面，並依照系統指示回傳結構化資料。

OCR 來源：{context.get('ocr_provider', 'unknown')}

參考段落（請從中提取規則，並在 rules_points 引用這些 ID）：
{json.dumps(paragraph_payload, ensure_ascii=False, indent=2)}

完整 OCR 文字（輔助閱讀）：
{ocr_text}

請注意：你需要從整張圖片中找出所有視覺元素，並提供精確的 0-1000 歸一化 BBox 座標。圖片與文字對應必須有證據；如果只是同頁但無法確認語意關聯，不要建立連結。
""".strip()

    model = os.getenv("OLLAMA_MODEL", "qwen3.8:27b")
    host = os.getenv("OLLAMA_HOST", "http://100.123.189.49:7401")
    client = Client(host=host)

    print(f"Sending image to {model} for analysis...")
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt,
                "images": [str(page_image)],
            },
        ],
        format=OUTPUT_SCHEMA,
        options={"temperature": 0.1, "num_ctx": 8192},
        stream=False,
    )

    try:
        result = json.loads(response.message.content)
    except json.JSONDecodeError as e:
        print("模型回傳的不是合法的 JSON:", e)
        print("原始回傳內容:", response.message.content)
        return

    print("Cropping visual elements and composing Markdown...")
    final_markdown = crop_and_replace(
        result.get("markdown", ""),
        result.get("visual_elements", []),
        page_image,
        output_md_path.parent
    )

    result["final_markdown"] = final_markdown
    result["source"] = {
        "page_context": str(context_path.resolve()),
        "page_image": str(page_image),
        "ocr_provider": context.get("ocr_provider"),
        "model": model,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(final_markdown, encoding="utf-8")

    print(f"Analysis saved to: {output_json_path}")
    print(f"Markdown saved to: {output_md_path}")
    print(f"Cropped images saved in: {output_md_path.parent / 'crops'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 Ollama + Qwen3-VL 分析 OCR 已解析的桌遊說明書頁面，並輸出圖文並茂的 Markdown"
    )
    parser.add_argument("page_context", type=Path, help="01 或 02 產生的 page_context.json")
    parser.add_argument("--out-md", type=Path, default=Path("output/page.md"), help="輸出的 Markdown 檔案路徑")
    parser.add_argument("--out-json", type=Path, default=Path("output/analysis.json"), help="輸出的詳細分析 JSON 路徑")
    args = parser.parse_args()

    parse_paadle_to_md(args.page_context, args.out_md, args.out_json)


if __name__ == "__main__":
    main()