# iqa/iqa_model.py
import cv2
import numpy as np

def _clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))

def process_bgr_image(img_bgr: np.ndarray) -> dict:
    """
    Returns a dict with:
      blur_%, lowres_%, under_%, over_%, noise_%, haze_%, edge_density, fft_energy,
      quality_% (higher=better), quality_score_percent (alias)
    """
    if img_bgr is None or img_bgr.size == 0:
        return {"quality_%": 0.0, "quality_score_percent": 0.0}

    h, w = img_bgr.shape[:2]

    # --- blur metric (variance of Laplacian) ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())

    # Map blur_var to "blur_%": higher blur_% = worse
    # (tune these two numbers for your camera)
    blur_bad = 50.0
    blur_good = 500.0
    blur_score = (blur_var - blur_bad) / (blur_good - blur_bad + 1e-9)
    blur_pct_worse = _clamp(100.0 * (1.0 - blur_score), 0, 100)

    # --- low resolution proxy (small image or low texture) ---
    # (simple proxy: if width < 640 or height < 480, penalize)
    lowres_pct_worse = 0.0
    if w < 640 or h < 480:
        lowres_pct_worse = 60.0
    elif w < 960 or h < 720:
        lowres_pct_worse = 20.0

    # --- exposure (under/over) ---
    mean_int = float(gray.mean())
    under_pct = _clamp(max(0.0, (80.0 - mean_int) * 1.2), 0, 100)
    over_pct  = _clamp(max(0.0, (mean_int - 180.0) * 1.2), 0, 100)

    # --- noise proxy (high-frequency energy after blur) ---
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float32) - blur.astype(np.float32)
    noise_std = float(noise.std())
    noise_pct = _clamp(noise_std * 3.0, 0, 100)

    # --- haze proxy (low contrast) ---
    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float(p95 - p5)
    haze_pct = _clamp(max(0.0, (60.0 - contrast) * 1.5), 0, 100)

    # --- edge density & fft energy (informative) ---
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean() / 255.0)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1e-9)
    fft_energy = float(mag.mean())

    # --- combine into quality_% (higher better) ---
    # weights (tune if needed)
    w_blur   = 0.35
    w_lowres = 0.20
    w_under  = 0.10
    w_over   = 0.10
    w_noise  = 0.15
    w_haze   = 0.10

    penalty = (
        w_blur * blur_pct_worse +
        w_lowres * lowres_pct_worse +
        w_under * under_pct +
        w_over * over_pct +
        w_noise * noise_pct +
        w_haze * haze_pct
    )
    quality = _clamp(100.0 - penalty, 0, 100)

    return {
        "blur_%": blur_pct_worse,
        "lowres_%": lowres_pct_worse,
        "under_%": under_pct,
        "over_%": over_pct,
        "noise_%": noise_pct,
        "haze_%": haze_pct,
        "edge_density": edge_density,
        "fft_energy": fft_energy,
        "quality_%": quality,
        "quality_score_percent": quality,   # IMPORTANT alias
    }
