import cv2
import numpy as np
from typing import Tuple, List

def _ensure_int(k):
    try:
        return int(k)
    except Exception:
        return 0

def no_change_fn(img, rng):
    return img.copy(), {}

def rotation_fn(img, rng, max_angle: float = 15.0):
    h, w = img.shape[:2]
    angle = float(rng.uniform(-max_angle, max_angle))
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, {"angle_deg": angle}

def gaussian_blur_fn(img, rng, ksize_range=(3,7)):
    k = int(rng.randint(ksize_range[0], ksize_range[1] + 1))
    if k % 2 == 0:
        k += 1
    sigma = float(rng.uniform(0.5, 2.0))
    out = cv2.GaussianBlur(img, (k, k), sigmaX=sigma)
    return out, {"kernel_size": k, "sigma": sigma}

def jpeg_compression_fn(img, rng, qualities=(30,50,80)):
    q = int(rng.choice(qualities))
    success, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not success:
        return img.copy(), {"quality": None}
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec, {"quality": q}

def brightness_contrast_fn(img, rng, contrast_range=(0.7,1.3), brightness_shift=20):
    alpha = float(rng.uniform(contrast_range[0], contrast_range[1]))
    beta = int(rng.randint(-brightness_shift, brightness_shift + 1))
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out, {"contrast": alpha, "brightness_shift": beta}

def salt_and_pepper_fn(img, rng, amount_range=(0.01,0.03)):
    out = img.copy()
    h, w = out.shape[:2]
    amount = float(rng.uniform(amount_range[0], amount_range[1]))
    num = int(np.ceil(amount * h * w))
    # salt
    ys = rng.randint(0, h, num)
    xs = rng.randint(0, w, num)
    out[ys, xs] = 255
    # pepper
    ys = rng.randint(0, h, num)
    xs = rng.randint(0, w, num)
    out[ys, xs] = 0
    return out, {"amount": amount}

def affine_fn(img, rng, scale_range=(0.9,1.1), shear_deg=10.0):
    h, w = img.shape[:2]
    sx = float(rng.uniform(scale_range[0], scale_range[1]))
    sy = sx
    shear = np.deg2rad(float(rng.uniform(-shear_deg, shear_deg)))
    M = np.array([[sx, np.tan(shear)],
              [np.tan(shear), sy]], dtype=np.float32)
    cx, cy = w/2, h/2
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    A = np.eye(3, dtype=np.float32)
    A[:2, :2] = M
    A = np.linalg.inv(T) @ A @ T
    M2 = A[:2, :3]
    out = cv2.warpAffine(img, M2, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, {"scale": sx, "shear_deg": float(np.rad2deg(shear))}

def elastic_fn(img, rng, alpha=30, sigma=4):
    h, w = img.shape[:2]
    dx = (rng.standard_normal((h, w)).astype(np.float32)) * alpha
    dy = (rng.standard_normal((h, w)).astype(np.float32)) * alpha
    dx = cv2.GaussianBlur(dx, (0,0), sigmaX=sigma, sigmaY=sigma)
    dy = cv2.GaussianBlur(dy, (0,0), sigmaX=sigma, sigmaY=sigma)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    remapped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return remapped, {"alpha": alpha, "sigma": sigma}

def combined_mix_fn(img, rng):
    funcs = [rotation_fn, elastic_fn, gaussian_blur_fn, jpeg_compression_fn, brightness_contrast_fn, salt_and_pepper_fn, affine_fn]
    k = int(rng.randint(2, 4))
    out = img.copy()
    params = {"transforms": []}
    for i in range(k):
        f = rng.choice(funcs)
        out, p = f(out, rng)
        params["transforms"].append({"name": f.__name__, "params": p})
    return out, params

class AugObj:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn
    def apply(self, img, rng):
        return self._fn(img, rng)

def get_augmentations(seed: int = 42) -> List[AugObj]:
    return [
        AugObj("no_change", no_change_fn),
        AugObj("rotation", rotation_fn),
        AugObj("elastic", elastic_fn),
        AugObj("gaussian_blur", gaussian_blur_fn),
        AugObj("jpeg_compression", jpeg_compression_fn),
        AugObj("brightness_contrast", brightness_contrast_fn),
        AugObj("salt_and_pepper", salt_and_pepper_fn),
        AugObj("affine", affine_fn),
        AugObj("combined_mix", combined_mix_fn),
    ]