
import json
import re
from pathlib import Path

import ollama
from PIL import Image, ImageOps


# ==========================================
# Configuration
# ==========================================

INPUT_IMAGE = Path("input/board_game_rule.jpg")
OUTPUT_DIR = Path("output")

MODEL = "qwen3-vl:8b-8k"

OLLAMA_HOST = "http://100.123.189.49:7401/v1"

NUM_CTX = 8192
NUM_PREDICT = 4096

# Vision 模型分析用圖片最大長邊
MAX_IMAGE_SIZE = 1600

# 手動旋轉圖片：0 / 90 / 180 / 270
ROTATE_DEGREES = 0

# Bounding Box 座標範圍：0 ~ 1000


# ==========================================
# Stage 1: Image preprocessing
# ==========================================

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    if ROTATE_DEGREES:
        image = image.rotate(
            ROTATE_DEGREES,
            expand=True
        )

    original_size = image.size

    # 縮放 Vision 分析用圖片
    image.thumbnail(
        (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    analysis_path = OUTPUT_DIR / "analysis_input.jpg"
    image.save(analysis_path, quality=90)

    return image, analysis_path, original_size


# ==========================================
# Stage 2: Vision LLM analysis
# ==========================================

def analyze_image(image_path):

    prompt = """
你是一個專業的桌遊說明書 OCR 與版面分析系統。

請分析整張圖片，將內容轉換成結構化 JSON。

必須遵守：

1. 完整辨識所有可讀的繁體中文文字。
2. 保留原始閱讀順序。
3. 區分標題、段落、清單、表格與圖片。
4. 桌遊卡牌、遊戲插圖、遊戲示意圖必須標記為 image。
5. 不要把裝飾線條、背景、文字區域當成圖片。
6. 圖片必須回傳 Bounding Box。
7. Bounding Box 使用 0~1000 的正規化座標。
8. 座標格式為 [x1, y1, x2, y2]。
9. 座標必須對應圖片中的實際區域。
10. 不要自行補充圖片中不存在的內容。
11. 不確定的文字使用 [無法辨識]。

JSON 格式：

{
  "title": "文件標題",
  "blocks": [
    {
      "type": "heading",
      "level": 1,
      "text": "遊戲準備"
    },
    {
      "type": "text",
      "text": "每位玩家取得一張角色卡。"
    },
    {
      "type": "image",
      "bbox": [100, 200, 500, 600],
      "description": "角色卡",
      "alt": "角色卡圖片"
    }
  ]
}

請只輸出 JSON，不要使用 Markdown Code Fence。
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path.resolve())]
            }
        ],
        format="json",
        options={
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": 0
        }
    )

    content = response["message"]["content"]

    return json.loads(content)


# ==========================================
# Stage 3: Crop image regions
# ==========================================

def crop_image_regions(original_image, blocks):

    image_dir = OUTPUT_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    width, height = original_image.size

    image_count = 0

    for block in blocks:

        if block.get("type") != "image":
            continue

        bbox = block.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # 正規化座標轉成像素座標
        left = int(x1 / 1000 * width)
        top = int(y1 / 1000 * height)
        right = int(x2 / 1000 * width)
        bottom = int(y2 / 1000 * height)

        # 邊界檢查
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))

        if right - left < 20 or bottom - top < 20:
            continue

        cropped = original_image.crop(
            (left, top, right, bottom)
        )

        image_count += 1

        filename = f"image_{image_count:03d}.png"

        cropped.save(image_dir / filename)

        # 將實際圖片路徑寫回 block
        block["image_path"] = f"images/{filename}"

    return blocks


# ==========================================
# Stage 4: Markdown generation
# ==========================================

def build_markdown(data, blocks):

    lines = []

    title = data.get("title", "桌遊說明書")

    lines.append(f"# {title}\n")

    for block in blocks:

        block_type = block.get("type")

        if block_type == "heading":

            level = max(
                1,
                min(int(block.get("level", 2)), 6)
            )

            text = block.get("text", "")

            lines.append(
                f"\n{'#' * level} {text}\n"
            )

        elif block_type == "text":

            text = block.get("text", "")

            if text:
                lines.append(f"\n{text}\n")

        elif block_type == "image":

            image_path = block.get("image_path")

            if image_path:

                description = (
                    block.get("alt")
                    or block.get("description")
                    or "圖片"
                )

                lines.append(
                    f"\n![{description}]({image_path})\n"
                )

                if block.get("description"):
                    lines.append(
                        f"\n*圖片說明：{block['description']}*\n"
                    )

    return "\n".join(lines)


# ==========================================
# Main
# ==========================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("[1/4] Preprocessing image...")

    original_image, analysis_path, _ = preprocess_image(
        INPUT_IMAGE
    )

    print("[2/4] Analyzing image with Ollama...")

    data = analyze_image(analysis_path)

    # 儲存原始分析 JSON，方便除錯
    json_path = OUTPUT_DIR / "analysis.json"

    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    blocks = data.get("blocks", [])

    print("[3/4] Cropping image regions...")

    blocks = crop_image_regions(
        original_image,
        blocks
    )

    print("[4/4] Generating Markdown...")

    markdown = build_markdown(data, blocks)

    output_md = OUTPUT_DIR / f"{INPUT_IMAGE.stem}.md"

    output_md.write_text(
        markdown,
        encoding="utf-8"
    )

    print("\nCompleted!")
    print(f"Markdown: {output_md}")
    print(f"Images: {OUTPUT_DIR / 'images'}")
    print(f"Image blocks: {sum(1 for b in blocks if b.get('image_path'))}")


if __name__ == "__main__":
    main()
