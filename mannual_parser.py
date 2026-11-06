from pathlib import Path

from ollama_qwen3vl_analyze import parse_paadle_to_md
from paddleocr_pp_structure import parse_pic_to_json

if __name__ == '__main__':
    img_path = Path(r"E:\code\雲端平台AI應用實務-蔡文龍\assets\board_game\assets\bbw\bbw_1.jpg")
    output_dir = Path(f"output/{img_path.stem}")
    parse_pic_to_json(img_path, output_dir=output_dir)
    parse_paadle_to_md(output_dir / "page_context.json", Path("output/page.md"), Path("output/analysis.json"))

