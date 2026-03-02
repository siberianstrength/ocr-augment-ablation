# src/augmentations.py
import cv2
import numpy as np
from typing import Callable, Dict, Tuple

def no_change(image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, dict]:
    return image.copy(), {}

def rotation(image: np.ndarray, rng: np.random.RandomState, max_angle: float = 15.0) -> Tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    angle = float(rng.uniform(-max_angle, max_angle))
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    out = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, {"angle_deg": angle}

def gaussian_blur(image: np.ndarray, rng: np.random.RandomState, ksize_range: Tuple[int,int]=(3,7)) -> Tuple[np.ndarray, dict]:
    k = int(rng.randint(ksize_range[0], ksize_range[1]+1))
    if k % 2 == 0:
        k += 1
    sigma = float(rng.uniform(0.5, 2.0))
    out = cv2.GaussianBlur(image, (k,k), sigmaX=sigma)
    return out, {"kernel_size": k, "sigma": sigma}

def jpeg_compression(image: np.ndarray, rng: np.random.RandomState, qualities=(30,50,80)) -> Tuple[np.ndarray, dict]:
    q = int(rng.choice(qualities))
    success, enc = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not success:
        return image.copy(), {"quality": None}
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec, {"quality": q}

def brightness_contrast(image: np.ndarray, rng: np.random.RandomState, contrast_range=(0.7,1.3), brightness_shift=20) -> Tuple[np.ndarray, dict]:
    alpha = float(rng.uniform(contrast_range[0], contrast_range[1]))
    beta = int(rng.randint(-brightness_shift, brightness_shift+1))
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return out, {"contrast": alpha, "brightness_shift": beta}

def salt_and_pepper(image: np.ndarray, rng: np.random.RandomState, amount_range=(0.01,0.03)) -> Tuple[np.ndarray, dict]:
    out = image.copy()
    h, w = out.shape[:2]
    amount = float(rng.uniform(amount_range[0], amount_range[1]))
    num = int(np.ceil(amount * h * w))
    ys = rng.integers(0, h, num)
    xs = rng.integers(0, w, num)
    out[ys, xs] = 255
    ys = rng.integers(0, h, num)
    xs = rng.integers(0, w, num)
    out[ys, xs] = 0
    return out, {"amount": amount}

def affine_transform(image: np.ndarray, rng: np.random.RandomState, scale_range=(0.9,1.1), shear_deg=10.0) -> Tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    sx = float(rng.uniform(scale_range[0], scale_range[1]))
    sy = sx
    shear = np.deg2rad(float(rng.uniform(-shear_deg, shear_deg)))
    M = np.array([[sx, np.tan(shear), 0.0],
                  [np.tan(shear), sy, 0.0]], dtype=np.float32)
    cx, cy = w/2, h/2
    T = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], dtype=np.float32)
    A = np.eye(3, dtype=np.float32)
    A[:2,:2] = M
    A = np.linalg.inv(T) @ A @ T
    M2 = A[:2,:3]
    out = cv2.warpAffine(image, M2, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, {"scale": sx, "shear_deg": np.rad2deg(shear)}

def elastic_transform(image: np.ndarray, rng: np.random.RandomState, alpha=30, sigma=4) -> Tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    dx = (rng.standard_normal((h, w)).astype(np.float32)) * alpha
    dy = (rng.standard_normal((h, w)).astype(np.float32)) * alpha
    dx = cv2.GaussianBlur(dx, (0,0), sigmaX=sigma, sigmaY=sigma)
    dy = cv2.GaussianBlur(dy, (0,0), sigmaX=sigma, sigmaY=sigma)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    remapped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return remapped, {"alpha": alpha, "sigma": sigma}

def combined_mix(image: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, dict]:
    funcs = [rotation, elastic_transform, gaussian_blur, jpeg_compression, brightness_contrast, salt_and_pepper, affine_transform]
    k = int(rng.integers(2, 4))
    out = image.copy()
    params = {"transforms": []}
    for i in range(k):
        f = rng.choice(funcs)
        out, p = f(out, rng)
        params["transforms"].append({"name": f.__name__, "params": p})
    return out, params

def get_augmentations(seed: int = 42):
    base_rng = np.random.RandomState(seed)
    def wrapper(fn):
        def w(img, rng=base_rng):
            sub = np.random.RandomState(rng.randint(0, 2**31 - 1))
            return fn(img, sub)
        return w
    # return list of simple augmentation objects expected by run_robustness.py
    class AugObj:
        def __init__(self, name, fn): self.name = name; self._fn = fn
        def apply(self, img, rng): return self._fn(img, rng)
    aus = [
        AugObj("no_change", lambda img, rng: (img.copy(), {})),
        AugObj("rotation", rotation),
        AugObj("elastic", elastic_transform),
        AugObj("gaussian_blur", gaussian_blur),
        AugObj("jpeg_compression", jpeg_compression),
        AugObj("brightness_contrast", brightness_contrast),
        AugObj("salt_and_pepper", salt_and_pepper),
        AugObj("affine", affine_transform),
        AugObj("combined_mix", combined_mix),
    ]
    return aus