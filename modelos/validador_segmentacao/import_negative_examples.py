import argparse
from pathlib import Path
import uuid

from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TARGET_DIR = Path("dataset_rotulado/rejeitar")


def iter_images(source_dir: Path):
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def make_negative_pair(image: Image.Image) -> Image.Image:
    original = image.convert("RGB")
    width, height = original.size

    flagged = original.copy()
    overlay = Image.new("RGBA", flagged.size, (180, 32, 32, 90))
    flagged = Image.alpha_composite(flagged.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(flagged)
    draw.line((0, 0, width, height), fill=(255, 255, 255), width=max(4, width // 80))
    draw.line((width, 0, 0, height), fill=(255, 255, 255), width=max(4, width // 80))
    draw.rectangle((8, 8, width - 8, height - 8), outline=(255, 255, 255), width=max(3, width // 120))

    canvas = Image.new("RGB", (width * 2, height))
    canvas.paste(original, (0, 0))
    canvas.paste(flagged, (width, 0))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importa fotos comuns como exemplos negativos óbvios para dataset_rotulado/rejeitar."
    )
    parser.add_argument("source_dir", help="Pasta com imagens comuns para importar como rejeitar")
    parser.add_argument("--limit", type=int, default=0, help="Máximo de imagens para importar (0 = sem limite)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Pasta de origem não encontrada: {source_dir}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    imported = 0
    for image_path in iter_images(source_dir):
        if args.limit and imported >= args.limit:
            break

        with Image.open(image_path) as image:
            pair = make_negative_pair(image)
            output_name = f"neg_{image_path.stem}_{uuid.uuid4().hex[:8]}.png"
            output_path = TARGET_DIR / output_name
            pair.save(output_path)
            imported += 1
            print(f"Importada: {output_path}")

    print(f"Total importado para rejeitar: {imported}")


if __name__ == "__main__":
    main()
