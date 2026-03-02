from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np


@dataclass
class Augmentation:
    """Single augmentation configuration."""

    name: str
    description: str

    def apply(self, image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError


class NoChangeAug(Augmentation):
    def apply(self, image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        return image.copy(), {}


class RotationAug(Augmentation):
    def apply(self, image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        angle = float(rng.uniform(-15.0, 15.0))
        transform = A.Rotate(
            limit=(angle, angle),
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            interpolation=cv2.INTER_LINEAR,
            p=1.0,
        )
        augmented = transform(image=image)["image"]
        return augmented, {"angle_deg": angle}


class ElasticAug(Augmentation):
    def apply(self, image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        transform = A.ElasticTransform(
            alpha=40,
            sigma=4,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            interpolation=cv2.INTER_LINEAR,
            p=1.0,
        )
        augmented = transform(image=image)["image"]
        return augmented, {"alpha": 40, "sigma": 4}


class GaussianBlurAug(Augmentation):
    def apply(self, image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        ksize = int(rng.choice([3, 5, 7]))
        sigma = float(rng.uniform(0.5, 2.0))
        transform = A.GaussianBlur(blur_limit=(ksize, ksize), sigma_limit=(sigma, sigma), p=1.0)
        augmented = transform(image=image)["image"]
        return augmented, {"kernel_size": ksize, "sigma": sigma}


class JPEGCompressionAug(Augmentation):
    def apply(self, image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        quality = int(rng.choice([30, 50, 80]))
        transform = A.ImageCompression(quality_range=(quality, quality), p=1.0)
        augmented = transform(image=image)["image"]
        return augmented, {"quality": quality}


class BrightnessContrastAug(Augmentation):
    def apply(self, image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        # contrast in [0.7, 1.3], brightness shift in [-20, 20]
        contrast = float(rng.uniform(0.7, 1.3))
        brightness = float(rng.uniform(-20, 20))
        transform = A.RandomBrightnessContrast(
            brightness_limit=(brightness / 255.0, brightness / 255.0),
            contrast_limit=(contrast - 1.0, contrast - 1.0),
            p=1.0,
        )
        augmented = transform(image=image)["image"]
        return augmented, {"contrast": contrast, "brightness_shift": brightness}


class SaltAndPepperAug(Augmentation):
    def apply(self, image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        amount = float(rng.uniform(0.01, 0.03))
        out = image.copy()
        h, w = out.shape[:2]
        num_pixels = int(amount * h * w)
        # salt (white)
        coords = (
            rng.randint(0, h, num_pixels),
            rng.randint(0, w, num_pixels),
        )
        out[coords[0], coords[1]] = 255
        # pepper (black)
        coords = (
            rng.randint(0, h, num_pixels),
            rng.randint(0, w, num_pixels),
        )
        out[coords[0], coords[1]] = 0
        return out, {"amount": amount}


class AffineAug(Augmentation):
    def apply(self, image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        scale = float(rng.uniform(0.9, 1.1))
        shear = float(rng.uniform(-10.0, 10.0))
        transform = A.Affine(
            scale=(scale, scale),
            shear=(shear, shear),
            fit_output=True,
            p=1.0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
        )
        augmented = transform(image=image)["image"]
        return augmented, {"scale": scale, "shear_deg": shear}


class CombinedMixAug(Augmentation):
    """Apply a random combination of 2–3 of the other augmentations."""

    def __init__(self, name: str, description: str, base_augs: List[Augmentation]) -> None:
        super().__init__(name=name, description=description)
        # exclude self and no_change
        self._base = [a for a in base_augs if a.name not in {"no_change", "combined_mix"}]

    def apply(self, image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        num = int(rng.randint(2, min(3, len(self._base)) + 1))
        chosen = list(rng.choice(self._base, size=num, replace=False))  # type: ignore[arg-type]
        out = image.copy()
        params: Dict[str, Any] = {"num_transforms": num, "transforms": []}
        for aug in chosen:
            out, p = aug.apply(out, rng)
            params["transforms"].append({"name": aug.name, "params": p})
        return out, params


def get_augmentations() -> List[Augmentation]:
    """Return the ordered list of augmentations, including baseline and combined mix."""

    base: List[Augmentation] = [
        NoChangeAug(
            name="no_change",
            description="Baseline: original image without any augmentation.",
        ),
        RotationAug(
            name="rotation",
            description="Random rotation within [-15, 15] degrees.",
        ),
        ElasticAug(
            name="elastic",
            description="Elastic transform with alpha_affine=30, sigma=4.",
        ),
        GaussianBlurAug(
            name="gaussian_blur",
            description="Gaussian blur with kernel size 3–7 and sigma in [0.5, 2.0].",
        ),
        JPEGCompressionAug(
            name="jpeg_compression",
            description="JPEG compression artifacts with quality in {30, 50, 80}.",
        ),
        BrightnessContrastAug(
            name="brightness_contrast",
            description="Contrast scaling in [0.7, 1.3] and brightness shift ±20.",
        ),
        SaltAndPepperAug(
            name="salt_and_pepper",
            description="Salt-and-pepper noise with amount in [0.01, 0.03].",
        ),
        AffineAug(
            name="affine",
            description="Affine scale [0.9, 1.1] and shear ±10 degrees.",
        ),
    ]
    combined = CombinedMixAug(
        name="combined_mix",
        description="Random combination of 2–3 augmentations from the above (stress test).",
        base_augs=base,
    )
    return base + [combined]

