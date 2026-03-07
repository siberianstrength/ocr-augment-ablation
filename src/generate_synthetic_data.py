import argparse
import os
import random
from typing import List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from utils import ensure_dir, set_seed


DEFAULT_TEXTS: List[str] = [
    "TOTAL: 123.45 USD",
    "Invoice #2025-03",
    "Due date: 23 Feb 2026",
    "Thank you for your purchase!",
    "Item A x2  19.99",
    "Subtotal  39.98",
    "Tax (10%)   3.99",
    "Grand Total 43.97",
    "Paid by card",
    "Customer: John Doe",
    "Order ID: 000123",
    "Delivery: Express",
    "Discount: 5.00",
    "Ref: ABC-4567-XYZ",
    "Tel: +1 (555) 123-4567",
]


def load_fonts(fonts_dir: Optional[str] = None) -> List[ImageFont.FreeTypeFont]:
    fonts: List[ImageFont.FreeTypeFont] = []
    if fonts_dir and os.path.isdir(fonts_dir):
        for name in os.listdir(fonts_dir):
            if name.lower().endswith(".ttf"):
                try:
                    fonts.append(ImageFont.truetype(os.path.join(fonts_dir, name), size=28))
                except Exception:
                    continue
    if not fonts:
        fonts.append(ImageFont.load_default())
    return fonts


def random_text(rng: random.Random) -> str:
    if rng.random() < 0.7:
        return rng.choice(DEFAULT_TEXTS)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    length = rng.randint(8, 20)
    return "".join(rng.choice(alphabet) for _ in range(length))


def create_image(text: str, font: ImageFont.FreeTypeFont, width: int = 600, height: int = 140) -> Image.Image:
    bg_color = (255, 255, 255)
    fg_color = (0, 0, 0)
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)

    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), text, fill=fg_color, font=font)
    return img


def create_synthetic_dataset(
    out_dir: str,
    num_samples: int = 30,
    seed: int = 42,
    fonts_dir: Optional[str] = None,
) -> None:
    """Create a small synthetic text-image dataset."""

    set_seed(seed)
    rng = random.Random(seed)
    ensure_dir(out_dir)

    fonts = load_fonts(fonts_dir)

    for i in range(num_samples):
        text = random_text(rng)
        font = rng.choice(fonts)
        img = create_image(text, font)

        base_name = f"synthetic_{i+1:04d}"
        img_path = os.path.join(out_dir, base_name + ".png")
        txt_path = os.path.join(out_dir, base_name + ".txt")

        img.save(img_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small synthetic OCR test set.")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for synthetic images and text files (e.g., data/test).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of synthetic examples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--fonts_dir",
        type=str,
        default=None,
        help="Optional directory with .ttf fonts to use.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    create_synthetic_dataset(
        out_dir=args.out,
        num_samples=args.num_samples,
        seed=args.seed,
        fonts_dir=args.fonts_dir,
    )


if __name__ == "__main__":
    main()

