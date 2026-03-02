import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import cer, ensure_dir, load_image_as_rgb, set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OCRDataset(Dataset):
    def __init__(self, root: str, charset: str, img_width: int = 160, img_height: int = 32) -> None:
        self.samples: List[Tuple[str, str]] = []
        for name in os.listdir(root):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                img_path = os.path.join(root, name)
                base, _ = os.path.splitext(img_path)
                txt_path = base + ".txt"
                if os.path.exists(txt_path):
                    self.samples.append((img_path, txt_path))

        self.charset = charset
        self.char_to_idx = {c: i + 1 for i, c in enumerate(charset)}  # 0 is CTC blank
        self.img_width = img_width
        self.img_height = img_height
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, txt_path = self.samples[idx]
        text = open(txt_path, "r", encoding="utf-8").read().strip()

        img = load_image_as_rgb(img_path)
        # Convert to grayscale and resize to fixed size.
        img_gray = np.mean(img, axis=2).astype(np.uint8)
        from PIL import Image

        pil_img = Image.fromarray(img_gray)
        pil_img = pil_img.resize((self.img_width, self.img_height), Image.BILINEAR)
        img_tensor = self.transform(pil_img)  # [1, H, W]

        label_indices = [self.char_to_idx.get(c, 0) for c in text]
        label = torch.tensor(label_indices, dtype=torch.long)

        return {"image": img_tensor, "label": label, "text": text}


class CRNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 x 16 x 80
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 x 8 x 40
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),  # 128 x 4 x 20
        )
        self.rnn = nn.LSTM(
            input_size=128 * 4,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(128 * 2, num_classes)  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        features = self.cnn(x)  # [B, C, H', W']
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2)  # [B, W', C, H']
        features = features.contiguous().view(b, w, c * h)  # [B, W', C*H']
        seq, _ = self.rnn(features)  # [B, W', 2*hidden]
        logits = self.fc(seq)  # [B, W', num_classes]
        return logits.permute(1, 0, 2)  # [T, B, C] for CTC


def ctc_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = [b["image"] for b in batch]
    labels = [b["label"] for b in batch]

    images_tensor = torch.stack(images, dim=0)

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_cat = torch.cat(labels, dim=0)

    batch_size, _, _, width = images_tensor.size()
    # After CNN, width is downsampled by 2*2*2 = 8.
    t = width // 8
    input_lengths = torch.full(size=(batch_size,), fill_value=t, dtype=torch.long)

    return {
        "images": images_tensor,
        "labels": labels_cat,
        "label_lengths": label_lengths,
        "input_lengths": input_lengths,
        "texts": [b["text"] for b in batch],
    }


def greedy_decode(logits: torch.Tensor, charset: str) -> List[str]:
    # logits: [T, B, C]
    probs = logits.softmax(dim=-1)
    indices = probs.argmax(dim=-1)  # [T, B]
    blank_idx = 0
    idx_to_char = {i + 1: c for i, c in enumerate(charset)}

    results: List[str] = []
    t_max, batch_size = indices.shape
    for b in range(batch_size):
        prev = None
        chars: List[str] = []
        for t in range(t_max):
            idx = int(indices[t, b])
            if idx == blank_idx or idx == prev:
                prev = idx
                continue
            prev = idx
            chars.append(idx_to_char.get(idx, ""))
        results.append("".join(chars))
    return results


@dataclass
class TrainConfig:
    train_dir: str
    val_dir: str
    out_dir: str
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3
    seed: int = 42


def build_charset(root: str) -> str:
    charset_set = set()
    for name in os.listdir(root):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            base, _ = os.path.splitext(os.path.join(root, name))
            txt_path = base + ".txt"
            if os.path.exists(txt_path):
                text = open(txt_path, "r", encoding="utf-8").read().strip()
                charset_set.update(text)
    charset = "".join(sorted(charset_set))
    if not charset:
        # Fallback basic ASCII subset.
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.:,-/ "
    return charset


def train_one_condition(cfg: TrainConfig) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("Finetune experiment requires a CUDA-capable GPU.")

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    charset = build_charset(cfg.train_dir)
    num_classes = len(charset) + 1  # +blank

    train_ds = OCRDataset(cfg.train_dir, charset)
    val_ds = OCRDataset(cfg.val_dir, charset)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=ctc_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ctc_collate,
    )

    model = CRNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_cer = float("inf")
    best_path = os.path.join(cfg.out_dir, "best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["images"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            input_lengths = batch["input_lengths"].to(DEVICE)
            label_lengths = batch["label_lengths"].to(DEVICE)

            logits = model(images)
            log_probs = logits.log_softmax(dim=-1)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()

        # Validation CER
        model.eval()
        cer_vals: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(DEVICE)
                texts = batch["texts"]
                logits = model(images)
                preds = greedy_decode(logits, charset)
                for ref, hyp in zip(texts, preds):
                    cer_vals.append(cer(ref, hyp))

        mean_cer = float(np.mean(cer_vals)) if cer_vals else 1.0
        if mean_cer < best_val_cer:
            best_val_cer = mean_cer
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "charset": charset,
                },
                best_path,
            )

    return best_val_cer


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional CRNN finetuning helper (GPU-only).")
    parser.add_argument("--train_dir", type=str, required=True, help="Training data directory.")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation/test data directory.")
    parser.add_argument("--out_dir", type=str, default="results_crnn", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = TrainConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    if not torch.cuda.is_available():
        print("CUDA not available; skipping CRNN finetuning.")
        return

    best_cer = train_one_condition(cfg)
    print(f"Best validation CER: {best_cer:.3f}")


if __name__ == "__main__":
    main()

