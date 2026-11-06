from pathlib import Path

from markitdown import MarkItDown
from openai import OpenAI


def image_to_markdown(image_path: str, output_path: str = None):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://100.123.189.49:7401/v1",

    )

    md = MarkItDown(
        llm_client=client,
        llm_model="qwen3-vl:8b-8k",
        llm_prompt="請完整提取圖片中的繁體中文文字，輸出 Markdown。"
    )

    # Convert image to Markdown
    result = md.convert(image_path)

    markdown_content = result.text_content

    # Output to console
    print(markdown_content)

    # Save as Markdown file
    if output_path:
        Path(output_path).write_text(
            markdown_content,
            encoding="utf-8"
        )

    return markdown_content


if __name__ == "__main__":
    image_to_markdown(
        image_path=r"E:\code\雲端平台AI應用實務-蔡文龍\assets\board_game\assets\20261031_112142.jpg",
        output_path="output/sample.md"
    )