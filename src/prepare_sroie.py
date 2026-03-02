"""
Convert ICDAR SROIE dataset to our format (image + single-line .txt per receipt).

SROIE annotations: each line is x1,y1,x2,y2,x3,y3,x4,y4,transcript
We join all transcripts into one line per image.

Usage:
  python prepare_sroie.py --sroie_dir "data/0325updated.task1train(626p)" --images_dir "data/task1train" --out data/test
  (If images are in the same folder as annotations, omit --images_dir)
"""
import argparse
import os
import shutil
from typing import Optional

from utils import ensure_dir


def parse_sroie_txt(path: str) -> str:
    """Parse SROIE annotation file: extract transcript from each line and join with space."""
    transcripts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 9:
                # First 8 are coords, rest is transcript (may contain commas)
                transcript = ",".join(parts[8:]).strip()
                if transcript:
                    transcripts.append(transcript)
    return " ".join(transcripts)


def find_image_for_txt(txt_path: str, images_dir: str) -> Optional[str]:
    """Find matching image (same basename, .jpg or .png) in images_dir."""
    base = os.path.splitext(os.path.basename(txt_path))[0]
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        img_path = os.path.join(images_dir, base + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SROIE dataset to our format.")
    parser.add_argument("--sroie_dir", required=True, help="Folder with SROIE .txt annotations")
    parser.add_argument("--images_dir", default=None, help="Folder with images (default: same as sroie_dir)")
    parser.add_argument("--out", default="data/test", help="Output folder for image+txt pairs")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples to convert (0=all)")
    args = parser.parse_args()

    images_dir = args.images_dir or args.sroie_dir
    ensure_dir(args.out)

    txt_files = [
        f for f in os.listdir(args.sroie_dir)
        if f.lower().endswith(".txt") and not f.startswith(".")
    ]
    txt_files = sorted(txt_files)[: args.max_samples] if args.max_samples else sorted(txt_files)

    created = 0
    skipped_no_image = 0
    for name in txt_files:
        txt_path = os.path.join(args.sroie_dir, name)
        img_path = find_image_for_txt(txt_path, images_dir)
        if img_path is None:
            skipped_no_image += 1
            continue

        gt = parse_sroie_txt(txt_path)
        base = os.path.splitext(name)[0]
        out_img = os.path.join(args.out, base + os.path.splitext(img_path)[1])
        out_txt = os.path.join(args.out, base + ".txt")

        shutil.copy2(img_path, out_img)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(gt)
        created += 1

    print(f"Created {created} pairs in {args.out}")
    if skipped_no_image:
        print(f"Skipped {skipped_no_image} annotations (no matching image in {images_dir})")
    if created == 0:
        print("\nNo images found! SROIE images are usually in a separate folder/zip.")
        print("Download from: https://drive.google.com/open?id=1ShItNWXyiYtFDM5W02bceHuJjyeeJl2")
        print("Extract and pass --images_dir to the folder containing the .jpg files.")


if __name__ == "__main__":
    main()
