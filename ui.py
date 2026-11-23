import base64
import json
import re
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import binary_fill_holes

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_IMPORT_ERROR = None
except Exception as _e:
    mp = mp_python = mp_vision = None
    MP_IMPORT_ERROR = repr(_e)

st.set_page_config(page_title="ThermHand Screening", layout="wide")

TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
TASK_MIN_BYTES = 100_000
FINGER_CHAINS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
PALM_IDXS = [0, 5, 9, 13, 17]
TIP_IDXS = (4, 8, 12, 16, 20)
LR_BASE = [
    "f_mask_mean_c", "f_focal_peak_c", "f_local_joint_contrast_c",
    "f_joint_p90_contrast_c", "f_palm_minus_tip_c",
]
DISCLAIMER = "Research screening aid, not a diagnosis. Scores are relative to the 17-subject cohort."

# Must be identical in the notebook.
GATE = dict(
    area_lo=0.03, area_hi=0.60,
    wrist_margin=0.05,
    scale_lo=0.80, scale_hi=1.25,
    finger_lo=0.60, finger_hi=1.30,
    jump_max=0.25,
    iou_min=0.80,
)
ACCEPT_MIN_FRAC = 0.50

ELL5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
YY, XX = np.mgrid[0:480, 0:640].astype(np.float32)


# ============================================================================ setup
def find_task_file():
    cands = [
        Path("hand_landmarker.task"),
        Path(tempfile.gettempdir()) / "hand_landmarker.task",
        Path("/tmp/hand_landmarker.task"),
    ]
    if Path("/kaggle/input").exists():
        cands += list(Path("/kaggle/input").glob("**/hand_landmarker.task"))
    for p in cands:
        if p.is_file() and p.stat().st_size > TASK_MIN_BYTES:
            return p, f"found at `{p.resolve()}`"
    target = Path("hand_landmarker.task")
    try:
        urllib.request.urlretrieve(TASK_URL, str(target))
        if target.is_file() and target.stat().st_size > TASK_MIN_BYTES:
            return target, f"downloaded to `{target}`"
        return None, "download returned a file that is too small"
    except Exception as e:
        return None, f"not found locally and automatic download failed ({e!r})."


def make_detector(task_path: Path):
    if mp_vision is None:
        return None, f"MediaPipe Tasks not importable: {MP_IMPORT_ERROR}"
    try:
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return mp_vision.HandLandmarker.create_from_options(opts), None
    except Exception as e:
        return None, f"detector creation failed: {e!r}"


def parse_meta(p: Path, idx: int):
    subj = None
    for tok in reversed(p.parts):
        m = re.fullmatch(r"[sS](\d+)", tok.strip())
        if m:
            subj = f"S{int(m.group(1))}"
            break
    if subj is None:
        # Strict parsing: do not assume or fallback to an arbitrary subject ID
        subj = p.parent.parent.name if len(p.parts) >= 3 else f"Unknown_S{idx:02d}"
    low = str(p).lower()
    hand = "left" if "left" in low else "right" if "right" in low else ("left" if idx % 2 == 0 else "right")
    stem = p.stem.lower()
    task = "open_close" if ("open" in stem or "close" in stem or "open" in low or "close" in low) else "static"
    return subj, hand, task


def find_rgb_for_npy(npy_path, hand, task, mp4s):
    rgb_like = [p for p in mp4s if "thermal" not in p.name.lower() and "radiometric" not in p.name.lower()]
    same = [p for p in rgb_like if p.stem.lower() == npy_path.stem.lower()]
    if same:
        return same[0]
    toks = ["open", "close"] if task == "open_close" else ["static"]
    scored = []
    for p in rgb_like:
        s = str(p).lower()
        sc = (2 if hand in s else 0) + (2 if any(t in s for t in toks) else 0) + (1 if "rgb" in s else 0)
        scored.append((sc, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored and scored[0][0] >= 2 else None


def read_rgb_frames(path: Path, wanted: set, rgb_k: int) -> dict:
    """Sequential read; returns {frame_idx: 640x480 RGB in canonical orientation}."""
    out = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return out
    last, i = max(wanted), 0
    try:
        while i <= last:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            if i in wanted:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = np.rot90(rgb, rgb_k) if rgb_k else rgb
                if rgb.shape[:2] != (480, 640):
                    rgb = cv2.resize(np.ascontiguousarray(rgb), (640, 480), interpolation=cv2.INTER_AREA)
                out[i] = np.ascontiguousarray(rgb, dtype=np.uint8)
            i += 1
    finally:
        cap.release()
    return out


# ============================================================================ thermal
def to_celsius(raw):
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    med = float(np.nanmedian(arr))
    if med > 10000.0:
        return arr / 100.0 - 273.15
    if med > 180.0:
        return arr - 273.15
    return arr


def orient_thermal(arr, mode):
    a = np.asarray(arr, dtype=np.float32)
    # Ensure array is at least 2D (height, width)
    if a.ndim < 2:
        return np.full((480, 640), 30.0, dtype=np.float32)
    if a.ndim == 3:
        # If multi-channel or extra dimension was passed, take first slice
        a = a[0] if a.shape[0] < a.shape[-1] else a[..., 0]

    if a.shape == (480, 640):
        out = a
    elif mode == "T":
        out = a.T
    elif mode == "rot_ccw":
        out = np.rot90(a, 1)
    elif mode == "rot_cw":
        out = np.rot90(a, -1)
    elif mode == "T_rot180":
        out = a.T[::-1, ::-1]
    else:
        out = a
    if out.shape != (480, 640):
        out = cv2.resize(np.ascontiguousarray(out), (640, 480), interpolation=cv2.INTER_LINEAR)
    out = np.ascontiguousarray(out, dtype=np.float32)
    bad = ~np.isfinite(out)
    if bad.any():
        fill = float(np.nanmedian(out)) if (~bad).any() else 30.0
        out = np.where(bad, fill, out).astype(np.float32)
    return out


def norm_u8(th_c):
    lo, hi = float(np.percentile(th_c, 5)), float(np.percentile(th_c, 98))
    return (np.clip((th_c - lo) / max(1e-4, hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def largest_component(m):
    m = binary_fill_holes(m > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ELL5)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lbl == best).astype(np.uint8)


def otsu_mask(th_c):
    finite = np.isfinite(th_c)
    vals = th_c[finite] if finite.any() else np.array([30.0], np.float32)
    bg = float(np.percentile(vals, 10))
    span = max(1.0, float(np.percentile(vals, 98)) - bg)
    t = np.nan_to_num(th_c, nan=bg)
    low = (t > bg + max(0.6, 0.12 * span)).astype(np.uint8)
    high = t > bg + max(1.5, 0.30 * span)
    n, lbl = cv2.connectedComponents(low, connectivity=8)
    if n <= 1:
        return np.zeros(th_c.shape[:2], np.uint8)
    keep = np.unique(lbl[high])
    keep = keep[keep > 0]
    if keep.size == 0:
        return np.zeros(th_c.shape[:2], np.uint8)
    return largest_component(np.isin(lbl, keep).astype(np.uint8))


def detect_thermal_hand(det, th_c):
    rgb = cv2.cvtColor(cv2.applyColorMap(norm_u8(th_c), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.hand_landmarks:
        return None
    return np.array([[p.x * 640.0, p.y * 480.0] for p in res.hand_landmarks[0]], dtype=np.float32)


def detect_rgb_display(det, rgb_canon):
    """Display only. Never used for scoring."""
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_canon)))
    if not res.hand_landmarks:
        return None
    return np.array([[p.x * 640.0, p.y * 480.0] for p in res.hand_landmarks[0]], dtype=np.float32)


def cut_forearm(th_m, pts):
    w, m9 = pts[0], pts[9]
    ax = m9 - w
    L = float(np.hypot(*ax))
    if L < 10:
        return th_m
    u = ax / L
    proj = (XX - w[0]) * u[0] + (YY - w[1]) * u[1]
    return largest_component(th_m * (proj >= -GATE["wrist_margin"] * L).astype(np.uint8))


def hand_length(pts):
    return float(np.hypot(*(pts[9] - pts[0])))


def finger_lengths(pts):
    return {
        fn: float(sum(np.hypot(*(pts[ch[i + 1]] - pts[ch[i]])) for i in range(3)))
        for fn, ch in FINGER_CHAINS.items()
    }


def mask_iou(a, b):
    inter = float(np.logical_and(a > 0, b > 0).sum())
    union = float(np.logical_or(a > 0, b > 0).sum())
    return inter / union if union > 0 else 0.0


# ============================================================================ features (unchanged math)
def focal_peak_c(th_c, th_m, scale):
    t = cv2.resize(th_c, (320, 240), interpolation=cv2.INTER_AREA)
    m = cv2.resize(th_m.astype(np.float32), (320, 240), interpolation=cv2.INTER_NEAREST)
    s1, s2 = max(1.5, 0.02 * scale / 2.0), max(6.0, 0.10 * scale / 2.0)
    tm = t * m
    nb1 = cv2.GaussianBlur(tm, (0, 0), s1) / (cv2.GaussianBlur(m, (0, 0), s1) + 1e-6)
    nb2 = cv2.GaussianBlur(tm, (0, 0), s2) / (cv2.GaussianBlur(m, (0, 0), s2) + 1e-6)
    inner = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    v = (nb1 - nb2)[inner]
    return float(np.percentile(v, 99)) if v.size > 20 else np.nan


def local_contrast(th_c, th_m, x, y, inner, outer, hand_med):
    h, w = th_c.shape
    r0, r1, c0, c1 = max(0, y - inner), min(h, y + inner + 1), max(0, x - inner), min(w, x + inner + 1)
    sel = th_m[r0:r1, c0:c1] > 0
    if sel.sum() < 5:
        return np.nan, np.nan, False
    loc = float(np.median(th_c[r0:r1, c0:c1][sel]))
    R0, R1, C0, C1 = max(0, y - outer), min(h, y + outer + 1), max(0, x - outer), min(w, x + outer + 1)
    ring = th_m[R0:R1, C0:C1] > 0
    ring[max(0, r0 - R0):max(0, r1 - R0), max(0, c0 - C0):max(0, c1 - C0)] = False
    sur = float(np.median(th_c[R0:R1, C0:C1][ring])) if ring.sum() >= 10 else hand_med
    return loc, loc - sur, True


def frame_features(th_c, th_m, pts, scale):
    valid_j = np.zeros(21, dtype=bool)
    contr_j = np.full(21, np.nan, dtype=np.float64)
    inner = int(max(3, 0.025 * scale))
    outer = int(max(9, 0.08 * scale))
    px = th_c[th_m > 0]
    if px.size < 200:
        return None, valid_j, contr_j
    med = float(np.median(px))
    gy, gx = np.gradient(th_c)
    st_dict = {
        "mean": float(px.mean()),
        "p90": float(np.percentile(px, 90)),
        "sd": float(px.std()),
        "grad_norm": float(np.hypot(gx, gy)[th_m > 0].mean()) * max(1.0, scale),
        "p99_minus_med": float(np.percentile(px, 99) - med),
        "focal_peak": focal_peak_c(th_c, th_m, scale),
        "j_valid": 0.0, "j_contrast": np.nan, "j_p90_contrast": np.nan,
        "j_peak": np.nan, "j_palm_tip": np.nan,
    }
    for fn in FINGER_CHAINS:
        st_dict[f"chain_{fn}"] = np.nan
    if pts is not None:
        th_dil = cv2.dilate(th_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        temps_j = np.full(21, np.nan)
        for j in range(21):
            xj, yj = int(round(pts[j, 0])), int(round(pts[j, 1]))
            if 0 <= xj < 640 and 0 <= yj < 480 and th_dil[yj, xj] > 0:
                temps_j[j], contr_j[j], valid_j[j] = local_contrast(th_c, th_m, xj, yj, inner, outer, med)
        st_dict["j_valid"] = float(valid_j.mean())
        if valid_j.any():
            c_fin = contr_j[valid_j & np.isfinite(contr_j)]
            if c_fin.size:
                st_dict["j_contrast"] = float(np.mean(c_fin))
                st_dict["j_p90_contrast"] = float(np.percentile(c_fin, 90))
                st_dict["j_peak"] = float(np.max(c_fin))
            palm = [temps_j[i] for i in PALM_IDXS if np.isfinite(temps_j[i])]
            tips = [temps_j[i] for i in TIP_IDXS if np.isfinite(temps_j[i])]
            if palm and tips:
                st_dict["j_palm_tip"] = float(np.mean(palm) - np.mean(tips))
            for fn, ch in FINGER_CHAINS.items():
                if np.isfinite(temps_j[ch[0]]) and np.isfinite(temps_j[ch[-1]]):
                    st_dict[f"chain_{fn}"] = float(temps_j[ch[0]] - temps_j[ch[-1]])
    return st_dict, valid_j, contr_j


def safe_nanmean(values):
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else np.nan


# ============================================================================ gate
def assess_recording(recs):
    det = [r for r in recs if r["pts"] is not None]
    L_med = float(np.median([r["L"] for r in det])) if det else np.nan
    fl_med = {fn: float(np.median([r["fl"][fn] for r in det])) for fn in FINGER_CHAINS} if det else {}
    for r in recs:
        reasons = []
        if r["detector_error"]:
            reasons.append("detector_error")
        if r["pts"] is None:
            reasons.append("no_detection")
        if not (GATE["area_lo"] <= r["area"] <= GATE["area_hi"]):
            reasons.append("mask_area")
        if r["pts"] is not None and np.isfinite(L_med) and L_med > 0:
            ratio = r["L"] / L_med
            if not (GATE["scale_lo"] <= ratio <= GATE["scale_hi"]):
                reasons.append("scale")
            if any(not (GATE["finger_lo"] <= r["fl"][fn] / max(1e-6, fl_med[fn]) <= GATE["finger_hi"])
                   for fn in FINGER_CHAINS):
                reasons.append("finger_len")
            if r["jump_px"] is not None and r["jump_px"] / L_med > GATE["jump_max"]:
                reasons.append("jump")
        if r["iou_prev"] is not None and r["iou_prev"] < GATE["iou_min"]:
            reasons.append("mask_unstable")
        r["reasons"] = reasons
        r["accepted"] = not reasons


# ============================================================================ drawing
def draw_thermal(th_c, th_m, pts, valid_mask, contrasts, inner, outer, accepted):
    vis = cv2.applyColorMap(norm_u8(th_c), cv2.COLORMAP_INFERNO)
    cnts, _ = cv2.findContours(th_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (255, 255, 0), 2)
    if pts is not None:
        col = (200, 255, 200) if accepted else (80, 80, 255)
        for a, b in HAND_EDGES:
            xa, ya = int(round(pts[a, 0])), int(round(pts[a, 1]))
            xb, yb = int(round(pts[b, 0])), int(round(pts[b, 1]))
            if 0 <= xa < 640 and 0 <= ya < 480 and 0 <= xb < 640 and 0 <= yb < 480:
                cv2.line(vis, (xa, ya), (xb, yb), col, 1)
        highlight = None
        if contrasts is not None and np.any(valid_mask & np.isfinite(contrasts)):
            highlight = int(np.argmax(np.where(valid_mask & np.isfinite(contrasts), contrasts, -1e9)))
        for j in range(21):
            xj, yj = int(round(pts[j, 0])), int(round(pts[j, 1]))
            if not (0 <= xj < 640 and 0 <= yj < 480):
                continue
            cv2.circle(vis, (xj, yj), 4, (0, 255, 0) if valid_mask[j] else (0, 180, 255), -1)
            if highlight == j:
                cv2.rectangle(vis, (xj - inner, yj - inner), (xj + inner, yj + inner), (255, 255, 255), 2)
                cv2.rectangle(vis, (xj - outer, yj - outer), (xj + outer, yj + outer), (0, 255, 255), 1)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def draw_rgb(rgb_canon, pts, accepted):
    """Display-only overlay on the RGB frame."""
    vis = cv2.cvtColor(np.ascontiguousarray(rgb_canon), cv2.COLOR_RGB2BGR)
    if pts is not None:
        col = (200, 255, 200) if accepted else (80, 80, 255)
        for a, b in HAND_EDGES:
            xa, ya = int(round(pts[a, 0])), int(round(pts[a, 1]))
            xb, yb = int(round(pts[b, 0])), int(round(pts[b, 1]))
            cv2.line(vis, (xa, ya), (xb, yb), col, 2)
        for j in range(21):
            xj, yj = int(round(pts[j, 0])), int(round(pts[j, 1]))
            cv2.circle(vis, (xj, yj), 5, (0, 255, 0), -1)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def to_b64(img_rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


# ============================================================================ scoring (unchanged)
def score_record_with_bundle(bundle, agg):
    cols = bundle["feature_columns"]
    x = np.array([agg.get(c, np.nan) for c in cols], dtype=np.float64)
    imp = np.array(bundle["impute"], dtype=np.float64)
    ctr = np.array(bundle["center"], dtype=np.float64)
    scl = np.array(bundle["scale"], dtype=np.float64)
    clip_z = float(bundle.get("clip_z", 6.0))
    pca_mean = np.array(bundle["pca_mean"], dtype=np.float64)
    pca_comp = np.array(bundle["pca_components"], dtype=np.float64)
    mcd_loc = np.array(bundle["mcd_location"], dtype=np.float64)
    mcd_prec = np.array(bundle["mcd_precision"], dtype=np.float64)

    xi = np.where(np.isfinite(x), x, imp)
    z = np.clip((xi - ctr) / scl, -clip_z, clip_z)
    p = (z - pca_mean) @ pca_comp.T
    z_hat = pca_mean + p @ pca_comp
    d_sub_raw = float(np.sqrt(max(0.0, (p - mcd_loc) @ mcd_prec @ (p - mcd_loc))))
    d_res_raw = float(np.sqrt(np.mean((z - z_hat) ** 2)))
    z_sub = max(0.0, (d_sub_raw - float(bundle["sub_med"])) / float(bundle["sub_iqr"]))
    z_res = max(0.0, (d_res_raw - float(bundle["res_med"])) / float(bundle["res_iqr"]))
    w = float(bundle.get("spe_weight", 0.5))
    d_comb = float(np.sqrt(z_sub ** 2 + w * (z_res ** 2)))
    top_idx = np.argsort(np.abs(z))[::-1][:5]
    top_drivers = [
        {"feature": cols[i], "raw_value": round(float(xi[i]), 4), "robust_z": round(float(z[i]), 3)}
        for i in top_idx
    ]
    return {
        "d_cohort_deviation": round(d_comb, 4),
        "z_subspace": round(z_sub, 4),
        "z_residual_spe": round(z_res, 4),
        "top_drivers": top_drivers,
    }


FEATURE_NAME_MAP = {
    "f_mask_mean_c": "Hand Mean Temperature",
    "f_mask_p90_c": "Hand Peak Temperature (90th percentile)",
    "f_mask_sd_c": "Hand Thermal Spread (Std Dev)",
    "f_mask_grad_norm_c": "Skin Thermal Gradient Norm",
    "f_mask_p99_minus_med_c": "Focal Thermal Peak (99th vs median)",
    "f_focal_peak_c": "Focal Hotspot Excess",
    "f_drift_slope_c": "Temporal Temperature Drift Rate",
    "f_step_noise_c": "Sensor Frame Noise Level",
    "f_local_joint_contrast_c": "Local Joint vs Periphery Contrast",
    "f_joint_p90_contrast_c": "Peak Joint vs Periphery Contrast",
    "f_peak_joint_excess_c": "Maximum Joint Heat Elevation",
    "f_palm_minus_tip_c": "Palm-to-Fingertip Heat Gradient",
    "f_index_mcp_minus_tip_c": "Index Finger Heat Gradient (Knuckle to Tip)",
    "f_middle_mcp_minus_tip_c": "Middle Finger Heat Gradient (Knuckle to Tip)",
    "f_ring_mcp_minus_tip_c": "Ring Finger Heat Gradient (Knuckle to Tip)",
    "f_pinky_mcp_minus_tip_c": "Little Finger Heat Gradient (Knuckle to Tip)",
    "f_mask_mean_c_lr_abs": "Left–Right Hand Mean Difference",
    "f_focal_peak_c_lr_abs": "Left–Right Focal Hotspot Asymmetry",
    "f_local_joint_contrast_c_lr_abs": "Left–Right Joint Contrast Asymmetry",
    "f_joint_p90_contrast_c_lr_abs": "Left–Right Peak Joint Asymmetry",
    "f_palm_minus_tip_c_lr_abs": "Left–Right Palm-Tip Gradient Asymmetry",
}

REASON_NAME_MAP = {
    "jump": "Rapid Motion Jump",
    "mask_unstable": "Tracking / Mask Shift",
    "no_detection": "No Landmarks Detected",
    "mask_area": "Hand Mask Area Out of Bounds",
    "area_small": "Hand Area Too Small",
    "area_large": "Hand Area Too Large",
    "scale": "Hand Depth / Scale Shift",
    "finger_len": "Finger Occlusion / Length Shift",
    "inside_too_few": "Too Few Joints Inside Mask",
    "low_temp": "Hand Temperature Too Low (<22°C)",
    "high_temp": "Hand Temperature Too High (>40°C)",
    "detector_error": "Detector Pipeline Error",
}


def friendly_reason(reason: str) -> str:
    return REASON_NAME_MAP.get(reason, reason.replace("_", " ").title())


def friendly(name: str) -> str:
    if name in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[name]
    s = name
    lr = s.endswith("_lr_abs")
    if lr:
        s = s[: -len("_lr_abs")]
    s = s.replace("f_", "", 1)
    if s.endswith("_c"):
        s = s[:-2]
    s = s.replace("_", " ").title()
    return f"{s} (Left–Right Difference)" if lr else s


# ============================================================================ HTML cards
def gate_badge(status_key):
    palette = {
        "WITHIN_COHORT_RANGE": ("#1f3a2c", "#7fd9a6", "Within cohort reference range"),
        "OUTLIER_VS_COHORT": ("#4a3412", "#ffb84d", "Atypical vs cohort reference range (review)"),
        "RECAPTURE_RECOMMENDED": ("#3a2222", "#ff8c8c", "Quality criteria not met · recapture recommended"),
    }
    bg, fg, text = palette[status_key]
    return f'<span style="background:{bg};color:{fg};padding:6px 12px;border-radius:999px;font-size:13px">● {text}</span>'


def rgb_panel_html(rgb_b64, thermal_b64, gate_ok, has_rgb, position="Top-Right Inset"):
    gate_txt = "Gate passed" if gate_ok else "Gate failed"
    gate_col = "#7fd9a6" if gate_ok else "#ff8c8c"
    main_src = rgb_b64 if has_rgb else thermal_b64
    main_label = "RGB · display only" if has_rgb else "Thermal (no RGB video matched)"

    if not has_rgb or position == "Hidden":
        return (
            '<div style="position:relative;background:#1b1d22;border-radius:14px;padding:16px;'
            'border:1px solid #3a3d44;box-sizing:border-box;height:520px;display:flex;flex-direction:column;'
            'justify-content:space-between">'
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
            f'<span style="background:#15171c;color:#eee;padding:5px 12px;border-radius:8px;font-size:12px">{main_label}</span>'
            f'<span style="background:#15171c;color:{gate_col};padding:5px 12px;border-radius:8px;font-size:12px">● {gate_txt}</span>'
            '</div>'
            '<div style="position:relative;flex:1;display:flex;align-items:center;justify-content:center;background:#101216;'
            'border-radius:10px;overflow:hidden">'
            f'<img src="data:image/png;base64,{main_src}" style="max-width:100%;max-height:430px;object-fit:contain;display:block"/>'
            '</div>'
            '</div>'
        )

    if position == "Side-by-Side (Split)":
        return (
            '<div style="position:relative;background:#1b1d22;border-radius:14px;padding:16px;'
            'border:1px solid #3a3d44;box-sizing:border-box;height:520px;display:flex;flex-direction:column;'
            'justify-content:space-between">'
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
            f'<span style="background:#15171c;color:#eee;padding:5px 12px;border-radius:8px;font-size:12px">{main_label} & Thermal</span>'
            f'<span style="background:#15171c;color:{gate_col};padding:5px 12px;border-radius:8px;font-size:12px">● {gate_txt}</span>'
            '</div>'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;flex:1;align-items:center;'
            'background:#101216;border-radius:10px;padding:8px;overflow:hidden">'
            '<div style="display:flex;flex-direction:column;align-items:center;height:100%;justify-content:center">'
            '<div style="font-size:11px;color:#aaa;margin-bottom:3px">RGB Display</div>'
            f'<img src="data:image/png;base64,{rgb_b64}" style="max-width:100%;max-height:400px;object-fit:contain;border-radius:6px"/>'
            '</div>'
            '<div style="display:flex;flex-direction:column;align-items:center;height:100%;justify-content:center">'
            '<div style="font-size:11px;color:#ffb84d;margin-bottom:3px">Thermal (Scored)</div>'
            f'<img src="data:image/png;base64,{thermal_b64}" style="max-width:100%;max-height:400px;object-fit:contain;border-radius:6px"/>'
            '</div>'
            '</div>'
            '</div>'
        )

    pos_style = {
        "Top-Right Inset": "top:14px;right:14px",
        "Top-Left Inset": "top:14px;left:14px",
        "Bottom-Right Inset": "bottom:14px;right:14px",
        "Bottom-Left Inset": "bottom:14px;left:14px",
    }.get(position, "top:14px;right:14px")

    inset = (
        f'<div style="position:absolute;{pos_style};width:32%;min-width:130px;max-width:180px;'
        'border:1px solid #4a4e58;border-radius:10px;overflow:hidden;background:#15171c;'
        'box-shadow:0 6px 18px rgba(0,0,0,0.7);z-index:9">'
        '<div style="font-size:11px;font-weight:600;color:#ffb84d;padding:4px 8px;'
        'background:#22252c;border-bottom:1px solid #3a3d44">Thermal (scored)</div>'
        f'<img src="data:image/png;base64,{thermal_b64}" style="width:100%;height:auto;display:block"/>'
        '</div>'
    )

    return (
        '<div style="position:relative;background:#1b1d22;border-radius:14px;padding:16px;'
        'border:1px solid #3a3d44;box-sizing:border-box;height:520px;display:flex;flex-direction:column;'
        'justify-content:space-between">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
        f'<span style="background:#15171c;color:#eee;padding:5px 12px;border-radius:8px;font-size:12px">{main_label}</span>'
        f'<span style="background:#15171c;color:{gate_col};padding:5px 12px;border-radius:8px;font-size:12px">● {gate_txt}</span>'
        '</div>'
        '<div style="position:relative;flex:1;display:flex;align-items:center;justify-content:center;background:#101216;'
        'border-radius:10px;overflow:hidden">'
        f'<img src="data:image/png;base64,{main_src}" style="max-width:100%;max-height:430px;object-fit:contain;display:block"/>'
        f'{inset}'
        '</div>'
        '</div>'
    )


def frame_stats_card_html(frame_dict, bundle=None):
    """
    Renders real-time statistics strictly for the selected frame.
    Updates dynamically as the user slides across frames.
    """
    ti = frame_dict.get("ti", 0)
    accepted = frame_dict.get("accepted", False)
    status_txt = frame_dict.get("status", "UNKNOWN")
    reasons = frame_dict.get("reasons", [])
    area = frame_dict.get("area", 0.0)
    valid_inside = frame_dict.get("valid_inside", 0)
    detected = frame_dict.get("detected", False)
    jump_px = frame_dict.get("jump_px")
    iou_prev = frame_dict.get("iou_prev")
    stats = frame_dict.get("frame_stats") or {}

    badge_bg = "#1f3a2c" if accepted else "#3a2222"
    badge_fg = "#7fd9a6" if accepted else "#ff8c8c"
    
    if accepted:
        badge_label = "● Accepted"
    else:
        clean_reasons = [friendly_reason(r) for r in reasons]
        reason_str = ", ".join(clean_reasons) if clean_reasons else friendly_reason(status_txt)
        badge_label = f"● Rejected: {reason_str}"

    mean_t_txt = f"{stats['mean']:.2f} °C" if "mean" in stats and np.isfinite(stats["mean"]) else "N/A"
    p90_t_txt = f"{stats['p90']:.2f} °C" if "p90" in stats and np.isfinite(stats["p90"]) else "N/A"
    min_t_txt = f"{stats['min_t']:.2f} °C" if "min_t" in stats and np.isfinite(stats["min_t"]) else "N/A"
    max_t_txt = f"{stats['max_t']:.2f} °C" if "max_t" in stats and np.isfinite(stats["max_t"]) else "N/A"
    sd_t_txt = f"{stats['sd']:.2f} °C" if "sd" in stats and np.isfinite(stats["sd"]) else "N/A"
    palm_tip_txt = f"{stats['j_palm_tip']:.2f} °C" if "j_palm_tip" in stats and np.isfinite(stats["j_palm_tip"]) else "N/A"

    jump_txt = f"{jump_px:.1f} px" if jump_px is not None and np.isfinite(jump_px) else "-"
    iou_txt = f"{iou_prev:.3f}" if iou_prev is not None and np.isfinite(iou_prev) else "-"

    # Compute frame-level cohort z deviations if stats & bundle are available
    z_rows = ""
    if bundle and stats:
        cols = bundle.get("feature_columns", [])
        ctr = np.array(bundle.get("center", []))
        scl = np.array(bundle.get("scale", []))
        frame_feats = {
            "f_mask_mean_c": stats.get("mean"),
            "f_mask_p90_c": stats.get("p90"),
            "f_mask_sd_c": stats.get("sd"),
            "f_mask_grad_norm_c": stats.get("grad_norm"),
            "f_mask_p99_minus_med_c": stats.get("p99_minus_med"),
            "f_focal_peak_c": stats.get("focal_peak"),
            "f_local_joint_contrast_c": stats.get("j_contrast"),
            "f_joint_p90_contrast_c": stats.get("j_p90_contrast"),
            "f_peak_joint_excess_c": stats.get("j_peak"),
            "f_palm_minus_tip_c": stats.get("j_palm_tip"),
            "f_index_mcp_minus_tip_c": stats.get("chain_index"),
            "f_middle_mcp_minus_tip_c": stats.get("chain_middle"),
            "f_ring_mcp_minus_tip_c": stats.get("chain_ring"),
            "f_pinky_mcp_minus_tip_c": stats.get("chain_pinky"),
        }
        z_items = []
        for idx, col in enumerate(cols):
            if col in frame_feats and frame_feats[col] is not None and np.isfinite(frame_feats[col]) and idx < len(ctr):
                z_val = (frame_feats[col] - ctr[idx]) / scl[idx]
                z_items.append((friendly(col), z_val))
        z_items.sort(key=lambda item: abs(item[1]), reverse=True)
        max_z = max([abs(z) for _, z in z_items[:4]], default=1.0) or 1.0
        z_rows = "".join(
            '<div style="display:flex;align-items:center;gap:10px;margin:5px 0">'
            f'<div style="flex:1.5;color:#ddd;font-size:12px;font-weight:500">{name}</div>'
            '<div style="flex:2;background:#2a2d33;border-radius:4px;height:7px;overflow:hidden">'
            f'<div style="width:{min(100, 100 * abs(z) / max_z):.0f}%;height:100%;background:#ffb84d"></div>'
            '</div>'
            f'<div style="width:55px;text-align:right;color:#ddd;font-size:12px;font-weight:600">{z:+.2f}</div>'
            '</div>'
            for name, z in z_items[:4]
        )
    if not z_rows:
        if not accepted:
            clean_r_list = [friendly_reason(r) for r in reasons]
            r_summary = ", ".join(clean_r_list) if clean_r_list else "Quality checks not met"
            z_rows = (
                '<div style="background:#15171c;border-radius:8px;padding:10px 14px;border:1px dashed #3a3d44;'
                'color:#bbb;font-size:12px;line-height:1.5;margin:6px 0">'
                '<div style="color:#ff8c8c;font-weight:600;margin-bottom:2px">Frame Excluded From Cohort Scoring</div>'
                f'<div>Reason(s): <b>{r_summary}</b>.</div>'
                '<div style="color:#888;font-size:11px;margin-top:2px">Thermal features are only aggregated for uncorrupted frames.</div>'
                '</div>'
            )
        else:
            z_rows = '<div style="color:#888;font-size:12px;margin:6px 0">No valid joints identified inside mask.</div>'

    chips_html = (
        f'<span style="display:inline-block;background:#15171c;color:{"#7fd9a6" if detected else "#ff8c8c"};'
        f'padding:4px 10px;border-radius:6px;margin:2px 4px 2px 0;font-size:12px;font-weight:500;border:1px solid #2a2d33">'
        f'{"✔" if detected else "✖"} Landmarks</span>'
        f'<span style="display:inline-block;background:#15171c;color:{"#7fd9a6" if valid_inside >= 11 else "#ff8c8c"};'
        f'padding:4px 10px;border-radius:6px;margin:2px 4px 2px 0;font-size:12px;font-weight:500;border:1px solid #2a2d33">'
        f'{"✔" if valid_inside >= 11 else "✖"} Joints ({valid_inside}/21)</span>'
        f'<span style="display:inline-block;background:#15171c;color:{"#7fd9a6" if 0.05 <= area <= 0.60 else "#ff8c8c"};'
        f'padding:4px 10px;border-radius:6px;margin:2px 4px 2px 0;font-size:12px;font-weight:500;border:1px solid #2a2d33">'
        f'{"✔" if 0.05 <= area <= 0.60 else "✖"} Area {area*100:.1f}%</span>'
    )

    return (
        '<div style="background:#1b1d22;border-radius:14px;padding:16px 20px;border:1px solid #3a3d44;'
        'color:#e8e8e8;box-sizing:border-box;height:520px;display:flex;flex-direction:column;justify-content:space-between">'
        '<div>'
        '<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div style="font-size:18px;font-weight:600">Selected Frame Stats <span style="color:#8bb4f7;font-size:15px">(#{ti})</span></div>'
        f'<span style="background:{badge_bg};color:{badge_fg};padding:4px 12px;border-radius:999px;font-size:12px;font-weight:500">{badge_label}</span>'
        '</div>'
        '<div style="margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:10px">'
        '<div style="background:#15171c;border-radius:8px;padding:10px 12px;border:1px solid #2a2d33">'
        '<div style="font-size:11px;font-weight:600;color:#888;letter-spacing:0.5px">MEAN TEMPERATURE</div>'
        f'<div style="font-size:22px;font-weight:600;color:#f0f0f0;margin-top:2px">{mean_t_txt}</div>'
        f'<div style="font-size:11px;color:#aaa;margin-top:3px">Peak (p90): <b>{p90_t_txt}</b></div>'
        '</div>'
        '<div style="background:#15171c;border-radius:8px;padding:10px 12px;border:1px solid #2a2d33">'
        '<div style="font-size:11px;font-weight:600;color:#888;letter-spacing:0.5px">TEMP RANGE & SPREAD</div>'
        f'<div style="font-size:15px;font-weight:600;color:#f0f0f0;margin-top:4px">{min_t_txt} – {max_t_txt}</div>'
        f'<div style="font-size:11px;color:#aaa;margin-top:3px">SD: <b>{sd_t_txt}</b> · Palm–Tip: <b>{palm_tip_txt}</b></div>'
        '</div>'
        '</div>'
        '<div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:10px">'
        '<div style="background:#15171c;border-radius:8px;padding:10px 12px;border:1px solid #2a2d33">'
        '<div style="font-size:11px;font-weight:600;color:#888;letter-spacing:0.5px">GEOMETRY & TRACKING</div>'
        f'<div style="font-size:13px;color:#ddd;margin-top:3px">Mask Area: <b>{area*100:.1f}%</b></div>'
        f'<div style="font-size:12px;color:#aaa;margin-top:2px">IoU vs Prev: <b>{iou_txt}</b></div>'
        '</div>'
        '<div style="background:#15171c;border-radius:8px;padding:10px 12px;border:1px solid #2a2d33">'
        '<div style="font-size:11px;font-weight:600;color:#888;letter-spacing:0.5px">JOINT RIGIDITY & STABILITY</div>'
        f'<div style="font-size:13px;color:#ddd;margin-top:3px">Joints Inside: <b>{valid_inside}/21</b></div>'
        f'<div style="font-size:12px;color:#aaa;margin-top:2px">Hand Jump: <b>{jump_txt}</b></div>'
        '</div>'
        '</div>'
        '<div style="margin-top:12px;font-size:13px;font-weight:600;color:#bbb">Indicative Frame Feature Deviations <span style="font-size:11px;font-weight:normal;color:#888">(single-frame vs recording-level cohort norms)</span></div>'
        f'{z_rows}'
        '</div>'
        '<div style="margin-top:10px">'
        '<div style="font-size:13px;font-weight:600;color:#bbb;margin-bottom:6px">Frame Quality Checks</div>'
        f'<div>{chips_html}</div>'
        '</div>'
        '</div>'
    )


def result_card_html(d, cut_p95, status_key, contributors, gate_chips, n_used, n_total, rec_id=""):
    if np.isfinite(d):
        top = max(2.0 * cut_p95, d * 1.1)
        cut_pct = 100.0 * cut_p95 / top
        d_pct = 100.0 * min(d, top) / top
        d_txt = f"{d:.2f}"
    else:
        top, cut_pct, d_pct, d_txt = 1.0, 50.0, 0.0, "N/A"
    max_z = max([abs(c["z"]) for c in contributors], default=1.0) or 1.0
    contrib_rows = "".join(
        '<div style="display:flex;align-items:center;gap:12px;margin:8px 0">'
        f'<div style="flex:1.4;color:#e8e8e8">{c["name"]}</div>'
        '<div style="flex:2;background:#3a3d44;border-radius:6px;height:10px;overflow:hidden">'
        f'<div style="width:{min(100, 100 * abs(c["z"]) / max_z):.0f}%;height:100%;background:#ffb84d"></div>'
        '</div>'
        f'<div style="width:70px;text-align:right;color:#e8e8e8">{c["z"]:+.2f}</div>'
        '</div>'
        for c in contributors
    ) or '<div style="color:#999">No feature contributions (insufficient accepted frames).</div>'
    chips = "".join(
        f'<span style="display:inline-block;background:#1f2228;color:{"#7fd9a6" if ok else "#ff8c8c"};'
        f'padding:6px 10px;border-radius:8px;margin:4px 6px 4px 0;font-size:13px">{"✔" if ok else "✖"} {label}</span>'
        for label, ok in gate_chips
    )
    rec_badge = f'<div style="color:#8bb4f7;font-size:13px;font-weight:500;margin-top:2px">Recording: {rec_id}</div>' if rec_id else ""
    return (
        '<div style="background:#1b1d22;border-radius:14px;padding:18px 20px;border:1px solid #3a3d44;color:#e8e8e8">'
        '<div style="display:flex;justify-content:space-between;align-items:center">'
        '<div>'
        '<div style="font-size:18px;font-weight:600">Combined Recording Analysis</div>'
        f'{rec_badge}'
        '</div>'
        '<div style="color:#999;font-size:12px">Aggregated across all frames</div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px">'
        f'<div><span style="font-size:44px;font-weight:600">{d_txt}</span>'
        '<span style="color:#999;font-size:13px"> cohort distance</span></div>'
        f'<div>{gate_badge(status_key)}</div>'
        '</div>'
        '<div style="position:relative;margin-top:14px;height:14px;border-radius:7px;overflow:hidden;background:#222">'
        f'<div style="position:absolute;left:0;top:0;bottom:0;width:{cut_pct:.1f}%;background:#1f3a2c"></div>'
        f'<div style="position:absolute;left:{cut_pct:.1f}%;top:0;bottom:0;right:0;background:#4a3412"></div>'
        f'<div style="position:absolute;left:{d_pct:.1f}%;top:-3px;bottom:-3px;width:3px;background:#fff"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;color:#999;font-size:12px;margin-top:6px">'
        f'<span>Within range</span><span style="position:relative;left:{cut_pct - 50:.0f}%">95th percentile = {cut_p95:.2f}</span><span>Above</span>'
        '</div>'
        '<div style="margin-top:18px;font-size:15px;color:#bbb">Top contributors <span style="font-size:12px;color:#888">(robust z, cohort-relative)</span></div>'
        f'{contrib_rows}'
        f'<div style="margin-top:14px;font-size:15px;color:#bbb">Quality gate · {n_used} of {n_total} frames used</div>'
        f'<div style="margin-top:6px">{chips}</div>'
        f'<div style="margin-top:14px;color:#888;font-size:12px">{DISCLAIMER}</div>'
        '</div>'
    )


# ============================================================================ UI
st.title("ThermHand Screening")
st.caption("Scoring uses thermal data only. The RGB panel is a display of the same hand and is not used for any score.")

col_a, col_b = st.columns(2)
with col_a:
    bundle_path_str = st.text_input("Model Bundle JSON Path", value="model_bundle_excl_S17.json")
with col_b:
    default_sub_dir = "S17"
    for cand in [Path("S17"), Path("../S17"), Path("/kaggle/input/flirone-heldout/S17")]:
        if cand.exists() and cand.is_dir():
            default_sub_dir = str(cand)
            break
    subject_dir_str = st.text_input("Held-Out Subject Folder (e.g., .../S17)", value=default_sub_dir)

n_frames_to_sample = st.slider("Frames to Analyze per Recording", min_value=6, max_value=32, value=16,
                                help="Number of evenly spaced frames sampled across the entire recording sequence.")
run_btn = st.button("Run Analysis", type="primary")

if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False

if run_btn:
    st.session_state["analysis_complete"] = False
    b_path = Path(bundle_path_str)
    if not b_path.exists() and (Path.cwd() / b_path.name).exists():
        b_path = Path.cwd() / b_path.name
    if not b_path.exists():
        st.error(f"Bundle not found: {b_path}")
        st.stop()
    s_dir = Path(subject_dir_str)
    if not s_dir.is_dir():
        st.error(f"Subject folder not found: {s_dir}")
        st.stop()

    bundle = json.loads(b_path.read_text(encoding="utf-8"))
    # Filter out virtual environments, packages, and hidden directories
    npy_files = [
        p for p in sorted(s_dir.rglob("*.npy"))
        if not any(part.startswith(".") or part in ("thermhand_env", "site-packages", "venv", "env", "__pycache__") for part in p.parts)
    ]
    mp4_files = [
        p for p in sorted(s_dir.rglob("*.mp4"))
        if not any(part.startswith(".") or part in ("thermhand_env", "site-packages", "venv", "env", "__pycache__") for part in p.parts)
    ]
    if not npy_files:
        st.error(f"No valid thermal sequence .npy files found under `{s_dir}` (excluded virtualenv and system packages).")
        st.stop()

    st.subheader("Tracing setup")
    task_file, task_msg = find_task_file()
    det, det_err = (None, None)
    if task_file is not None:
        det, det_err = make_detector(task_file)
    orient_mode = bundle.get("thermal_orient", "rot_ccw")
    rgb_k = int(bundle.get("rgb_rotation_k", 1))
    if det is not None:
        st.success(f"Detector loaded ({task_msg}).")
    else:
        st.error(f"Joint tracing is disabled, so no joint features can be computed. Reason: {det_err or task_msg}")
    st.caption(f"Gate thresholds: {GATE} | min accepted fraction: {ACCEPT_MIN_FRAC}")

    progress_bar = st.progress(0.0)
    live_status = st.empty()
    rec_data_list, reject_counts, det_errors_all = [], {}, []
    saved_frames = {}
    n_total = len(npy_files)

    for r_idx, np_p in enumerate(npy_files):
        subj, hand, task = parse_meta(np_p, r_idx)
        rec_id = f"{subj}_{hand}_{task}_{r_idx:02d}"
        try:
            th_raw = np.load(np_p, mmap_mode="r")
        except Exception as e:
            st.warning(f"Skipping unreadable or incompatible file {np_p.name}: {e}")
            continue

        # Validate that the file is an actual 3D sequence: (num_frames, height, width)
        if th_raw.ndim == 2:
            # Single static frame: wrap as 1-frame sequence (1, H, W)
            th_raw = th_raw[np.newaxis, ...]
        elif th_raw.ndim != 3:
            # Metadata, 1D array, or corrupted file: skip gracefully
            st.warning(f"Skipping {np_p.name}: expected 3D image sequence (frames, H, W), got shape {th_raw.shape}")
            continue

        n_th = int(th_raw.shape[0])
        if n_th == 0:
            continue
        idxs = np.linspace(0, n_th - 1, min(n_th, n_frames_to_sample)).astype(int)

        # RGB for display only
        rgb_p = find_rgb_for_npy(np_p, hand, task, mp4_files) if det is not None else None
        rgb_index = {}
        rgb_frames = {}
        if rgb_p is not None:
            cap = cv2.VideoCapture(str(rgb_p))
            n_rgb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            if n_rgb > 0:
                rgb_index = {int(ti): int(round(ti / max(1, n_th - 1) * (n_rgb - 1))) for ti in idxs}
                rgb_frames = read_rgb_frames(rgb_p, set(rgb_index.values()), rgb_k)

        # pass 1: thermal detection, forearm cut, measurements
        # Temporal tracking check: examine consecutive adjacent frame pairs (ti vs ti - 1)
        recs = []
        for pos, ti in enumerate(idxs):
            ti = int(ti)
            th_c = orient_thermal(to_celsius(th_raw[ti]), orient_mode)
            raw_m = otsu_mask(th_c)
            pts, err = None, False
            if det is not None:
                try:
                    pts = detect_thermal_hand(det, th_c)
                except Exception as e:
                    err = True
                    det_errors_all.append(f"{rec_id} frame {ti}: {e!r}")
            th_m = cut_forearm(raw_m, pts) if pts is not None else raw_m

            # Measure true consecutive tracking stability against immediate prior frame (ti - 1)
            iou_prev = None
            jump_px = None
            if ti > 0:
                th_c_prev = orient_thermal(to_celsius(th_raw[ti - 1]), orient_mode)
                raw_m_prev = otsu_mask(th_c_prev)
                pts_prev = None
                if det is not None:
                    try:
                        pts_prev = detect_thermal_hand(det, th_c_prev)
                    except Exception:
                        pts_prev = None
                th_m_prev = cut_forearm(raw_m_prev, pts_prev) if pts_prev is not None else raw_m_prev
                iou_prev = mask_iou(th_m, th_m_prev)
                if pts is not None and pts_prev is not None:
                    jump_px = float(np.percentile(np.hypot(*(pts - pts_prev).T), 90))

            rgb_canon, rgb_pts = None, None
            if det is not None and ti in rgb_index and rgb_index[ti] in rgb_frames:
                rgb_canon = rgb_frames[rgb_index[ti]]
                try:
                    rgb_pts = detect_rgb_display(det, rgb_canon)
                except Exception:
                    rgb_pts = None
            recs.append({
                "ti": ti, "th_c": th_c, "th_m": th_m, "pts": pts,
                "area": float(th_m.mean()), "iou_prev": iou_prev, "jump_px": jump_px,
                "L": hand_length(pts) if pts is not None else np.nan,
                "fl": finger_lengths(pts) if pts is not None else None,
                "detector_error": err, "rgb_canon": rgb_canon, "rgb_pts": rgb_pts,
            })
            frac = (r_idx + (pos + 1) / max(1, len(idxs))) / n_total
            progress_bar.progress(min(1.0, frac * 0.5))
            live_status.info(f"Measuring **{rec_id}** ({r_idx + 1}/{n_total}) frame `{ti}`")

        assess_recording(recs)
        rej = {}
        for r in recs:
            for why in r["reasons"]:
                rej[why] = rej.get(why, 0) + 1
                reject_counts[why] = reject_counts.get(why, 0) + 1

        # pass 2: features from accepted frames, render every frame
        f_stats = []
        L_vals = [r["L"] for r in recs if r["pts"] is not None]
        for pos, r in enumerate(recs):
            th_c, th_m, pts = r["th_c"], r["th_m"], r["pts"]
            scale = float(np.sqrt(max(1, th_m.sum())))
            inner, outer = int(max(3, 0.025 * scale)), int(max(9, 0.08 * scale))
            if r["accepted"]:
                st_dict, valid_j, contr_j = frame_features(th_c, th_m, pts, scale)
                if st_dict is not None:
                    vals = th_c[th_m > 0]
                    st_dict["min_t"] = float(np.min(vals)) if len(vals) > 0 else np.nan
                    st_dict["max_t"] = float(np.max(vals)) if len(vals) > 0 else np.nan
                    f_stats.append(st_dict)
                contrasts = contr_j
            else:
                st_dict = None
                valid_j = np.zeros(21, dtype=bool)
                if pts is not None:
                    for j in range(21):
                        xj, yj = int(round(pts[j, 0])), int(round(pts[j, 1]))
                        valid_j[j] = 0 <= xj < 640 and 0 <= yj < 480 and th_m[yj, xj] > 0
                contrasts = None
            thermal_img = draw_thermal(th_c, th_m, pts, valid_j, contrasts, inner, outer, r["accepted"])
            rgb_img = draw_rgb(r["rgb_canon"], r["rgb_pts"], r["accepted"]) if r["rgb_canon"] is not None else None
            r["thermal_img"], r["rgb_img"] = thermal_img, rgb_img
            r["valid_inside"] = int(valid_j.sum())
            r["status_text"] = "ACCEPTED" if r["accepted"] else "REJECTED: " + ", ".join(r["reasons"])
            r["frame_stats"] = st_dict

        n_s = len(recs)
        n_det = sum(r["pts"] is not None for r in recs)
        acc = [r for r in recs if r["accepted"]]
        n_acc = len(acc)
        accept_frac = n_acc / max(1, n_s)

        if len(f_stats) >= 3:
            m_arr = np.array([x["mean"] for x in f_stats], dtype=np.float64)
            valid_m = np.isfinite(m_arr)
            drift_slope = float(np.polyfit(np.arange(len(m_arr))[valid_m], m_arr[valid_m], 1)[0]) \
                if valid_m.sum() >= 2 else 0.0
            step_noise = float(np.median(np.abs(np.diff(m_arr)))) if len(m_arr) >= 2 else 0.0
            agg = {
                "f_mask_mean_c": float(np.nanmean(m_arr)),
                "f_mask_p90_c": safe_nanmean([x["p90"] for x in f_stats]),
                "f_mask_sd_c": safe_nanmean([x["sd"] for x in f_stats]),
                "f_mask_grad_norm_c": safe_nanmean([x["grad_norm"] for x in f_stats]),
                "f_mask_p99_minus_med_c": safe_nanmean([x["p99_minus_med"] for x in f_stats]),
                "f_focal_peak_c": safe_nanmean([x["focal_peak"] for x in f_stats]),
                "f_drift_slope_c": drift_slope,
                "f_step_noise_c": step_noise,
                "f_local_joint_contrast_c": safe_nanmean([x["j_contrast"] for x in f_stats]),
                "f_joint_p90_contrast_c": safe_nanmean([x["j_p90_contrast"] for x in f_stats]),
                "f_peak_joint_excess_c": safe_nanmean([x["j_peak"] for x in f_stats]),
                "f_palm_minus_tip_c": safe_nanmean([x["j_palm_tip"] for x in f_stats]),
            }
            for fn in FINGER_CHAINS:
                agg[f"f_{fn}_mcp_minus_tip_c"] = safe_nanmean([x[f"chain_{fn}"] for x in f_stats])
        else:
            agg = {}

        gate_cfg = bundle.get("quality_gate", {})
        q_mask_ok = sum(GATE["area_lo"] <= r["area"] <= GATE["area_hi"] for r in recs) / max(1, n_s)
        areas_acc = [r["area"] for r in acc]
        q_area_cv = float(np.std(areas_acc) / max(1e-6, np.mean(areas_acc))) if len(areas_acc) >= 2 else np.nan
        q_valid_j = float(np.mean([x["j_valid"] for x in f_stats])) if f_stats else 0.0
        gate_pass = (
            n_acc >= int(gate_cfg.get("min_n_used", 3))
            and accept_frac >= ACCEPT_MIN_FRAC
            and q_mask_ok >= float(gate_cfg.get("min_mask_ok_frac", 0.85))
            and np.isfinite(q_area_cv)
            and q_area_cv <= float(gate_cfg.get("max_area_cv", 0.35))
            and q_valid_j >= float(gate_cfg.get("min_valid_joint_frac", 0.50))
        )
        saved_frames[rec_id] = [
            {"ti": r["ti"], "thermal": r["thermal_img"], "rgb": r["rgb_img"], "status": r["status_text"],
             "accepted": r["accepted"], "reasons": r["reasons"], "area": r["area"],
             "detected": r["pts"] is not None, "valid_inside": r["valid_inside"],
             "iou_prev": r["iou_prev"], "jump_px": r["jump_px"], "frame_stats": r.get("frame_stats")}
            for r in recs
        ]
        rec_data_list.append({
            "rec_id": rec_id, "subject": subj, "hand": hand, "task": task,
            "n_sampled": n_s, "n_detected": n_det, "n_accepted": n_acc, "accept_frac": accept_frac,
            "reject_summary": ", ".join(f"{k}:{v}" for k, v in sorted(rej.items())) or "-",
            "gate_pass": gate_pass, "agg": agg,
            "q_mask_ok": q_mask_ok, "q_area_cv": q_area_cv, "q_valid_j": q_valid_j,
            "f_stats": f_stats,
            "has_rgb": rgb_p is not None,
            "rgb_video": rgb_p.name if rgb_p is not None else "none matched",
        })
        progress_bar.progress(min(1.0, 0.5 + (r_idx + 1) / n_total * 0.5))

    live_status.empty()

    # cross-hand features (exclude gate-failed recordings)
    pair_map = {}
    for item in rec_data_list:
        pair_map.setdefault((item["subject"], item["task"]), {})[item["hand"]] = item
    for (subj_key, task_key), hd in pair_map.items():
        if "left" in hd and "right" in hd:
            left_item, right_item = hd["left"], hd["right"]
            # Both sides must pass the quality gate to compute valid bilateral symmetry features
            if left_item["gate_pass"] and right_item["gate_pass"]:
                l_agg, r_agg = left_item["agg"], right_item["agg"]
                if l_agg and r_agg:
                    for base_col in LR_BASE:
                        v_l, v_r = l_agg.get(base_col, np.nan), r_agg.get(base_col, np.nan)
                        diff = abs(v_l - v_r) if np.isfinite(v_l) and np.isfinite(v_r) else np.nan
                        l_agg[f"{base_col}_lr_abs"] = diff
                        r_agg[f"{base_col}_lr_abs"] = diff
            else:
                # If either side failed the quality gate, bilateral symmetry is invalid and set to NaN
                # (prevents corrupting a valid recording with an unvalidated recording)
                for item_side in [left_item, right_item]:
                    if item_side.get("agg"):
                        for base_col in LR_BASE:
                            item_side["agg"][f"{base_col}_lr_abs"] = np.nan

    cut_p95 = float(bundle.get("d_cut_p95", 5.0))
    rec_rows, explain_details = [], {}
    for item in rec_data_list:
        agg = item["agg"]
        sc = score_record_with_bundle(bundle, agg) if agg else {
            "d_cohort_deviation": np.nan, "z_subspace": np.nan, "z_residual_spe": np.nan, "top_drivers": [],
        }
        if not item["gate_pass"]:
            status = "RECAPTURE_RECOMMENDED"
        elif np.isfinite(sc["d_cohort_deviation"]) and sc["d_cohort_deviation"] > cut_p95:
            status = "OUTLIER_VS_COHORT"
        else:
            status = "WITHIN_COHORT_RANGE"
        rec_rows.append({
            "rec_id": item["rec_id"], "subject": item["subject"], "hand": item["hand"], "task": item["task"],
            "n_sampled": item["n_sampled"], "n_detected": item["n_detected"],
            "n_accepted": item["n_accepted"], "accept_frac": round(item["accept_frac"], 3),
            "rejects": item["reject_summary"], "rgb_video": item["rgb_video"],
            "gate_pass": item["gate_pass"], "status": status,
            "d_cohort_deviation": sc["d_cohort_deviation"], "d_cut_p95": round(cut_p95, 4),
            "z_subspace": sc["z_subspace"], "z_residual_spe": sc["z_residual_spe"],
            "q_mask_ok_frac": round(item["q_mask_ok"], 3),
            "q_area_cv": round(item["q_area_cv"], 3) if np.isfinite(item["q_area_cv"]) else None,
            "q_valid_joint_frac": round(item["q_valid_j"], 3),
            "agg": agg,
            "disclaimer": DISCLAIMER,
        })
        explain_details[item["rec_id"]] = sc["top_drivers"]

    st.session_state["rec_rows"] = rec_rows
    st.session_state["saved_frames"] = saved_frames
    st.session_state["reject_counts"] = reject_counts
    st.session_state["det_errors_all"] = det_errors_all
    st.session_state["explain_details"] = explain_details
    st.session_state["cut_p95"] = cut_p95
    st.session_state["bundle"] = bundle
    st.session_state["analysis_complete"] = True

if st.session_state.get("analysis_complete", False):
    rec_rows = st.session_state["rec_rows"]
    saved_frames = st.session_state["saved_frames"]
    reject_counts = st.session_state["reject_counts"]
    det_errors_all = st.session_state["det_errors_all"]
    explain_details = st.session_state["explain_details"]
    cut_p95 = float(st.session_state["cut_p95"])
    bundle_active = st.session_state.get("bundle", {})

    # =========================================================================
    # 0. RUN OVERVIEW & DATA SUMMARY (Top)
    # =========================================================================
    n_rec = len(rec_rows)
    passed_all = [r for r in rec_rows if r["gate_pass"]]
    n_pass_all = len(passed_all)
    subjs = sorted(list({r["subject"] for r in rec_rows}))
    subj_str = ", ".join(subjs) if subjs else "Unknown"

    b_heldout_raw = bundle_active.get("heldout_subject")
    b_heldout = b_heldout_raw if b_heldout_raw else "None (All-Subjects Baseline)"
    b_ntrain = bundle_active.get("n_train_recordings", len(bundle_active.get("train_subjects", [])))
    b_name = Path(bundle_path_str).name

    all_means = [r["agg"].get("f_mask_mean_c") for r in passed_all if r.get("agg") and np.isfinite(r["agg"].get("f_mask_mean_c", np.nan))]
    all_peaks = [r["agg"].get("f_mask_p90_c") for r in passed_all if r.get("agg") and np.isfinite(r["agg"].get("f_mask_p90_c", np.nan))]
    all_contrasts = [r["agg"].get("f_local_joint_contrast_c") for r in passed_all if r.get("agg") and np.isfinite(r["agg"].get("f_local_joint_contrast_c", np.nan))]
    all_palm_tips = [r["agg"].get("f_palm_minus_tip_c") for r in passed_all if r.get("agg") and np.isfinite(r["agg"].get("f_palm_minus_tip_c", np.nan))]

    mean_hand_t = f"{np.mean(all_means):.1f} °C" if all_means else "N/A"
    peak_hand_t = f"{np.mean(all_peaks):.1f} °C" if all_peaks else "N/A"
    avg_j_contrast = f"{np.mean(all_contrasts):+.2f} °C" if all_contrasts else "N/A"
    avg_palm_tip = f"{np.mean(all_palm_tips):+.2f} °C" if all_palm_tips else "N/A"

    if not passed_all:
        stat_summary = (
            f"Quality criteria not met across tested stream(s). Hand stability or detection coverage was insufficient. "
            f"Recapture is recommended before evaluating distance against the {b_heldout}-heldout model."
        )
        stat_color = "#ff8c8c"
        stat_title = "Data Quality Sub-Optimal · Recapture Recommended"
    else:
        outlier_streams = [r for r in passed_all if np.isfinite(r["d_cohort_deviation"]) and r["d_cohort_deviation"] > cut_p95]
        if outlier_streams:
            stat_summary = (
                f"{len(outlier_streams)} of {len(passed_all)} evaluated stream(s) exceed the cohort 95th percentile distance (d > {cut_p95:.2f}). "
                f"Relative to the {b_ntrain} training recordings in {b_name}, this pattern is statistical outlier."
            )
            stat_color = "#ffb84d"
            stat_title = "Outlier Distance Detected vs Cohort Baseline"
        else:
            stat_summary = (
                f"Evaluated stream(s) lie within the cohort reference distance bounds (all d ≤ {cut_p95:.2f}). "
                f"Distribution aligns with the {b_ntrain} reference recordings in {b_name}."
            )
            stat_color = "#7fd9a6"
            stat_title = "Within Cohort Reference Range"

    st.markdown(
        '<div style="background:#1b1d22;border:1px solid #3a3d44;border-radius:14px;padding:18px 22px;margin-bottom:22px">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid #2a2d33;padding-bottom:12px;margin-bottom:14px">'
        '<div>'
        f'<div style="font-size:18px;font-weight:600;color:{stat_color}">{stat_title}</div>'
        f'<div style="font-size:13px;color:#bbb;margin-top:4px;max-width:850px;line-height:1.5">{stat_summary}</div>'
        '</div>'
        '<div style="text-align:right">'
        f'<div style="background:#15171c;border:1px solid #3a3d44;color:#aaa;padding:6px 14px;border-radius:8px;font-size:12px">'
        f'Subject: <b style="color:#eee">{subj_str}</b> · {n_rec} Stream{"s" if n_rec > 1 else ""}'
        '</div>'
        f'<div style="font-size:11px;color:#888;margin-top:4px">Bundle: <b style="color:#aaa">{b_name}</b> (held-out: {b_heldout}, N={b_ntrain})</div>'
        '</div>'
        '</div>'
        '<div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:12px">'
        '<div style="background:#15171c;padding:12px 14px;border-radius:10px;border:1px solid #252830">'
        '<div style="font-size:11px;color:#888;font-weight:600;letter-spacing:0.5px">MEAN HAND TEMPERATURE</div>'
        f'<div style="font-size:22px;font-weight:600;color:#f0f0f0;margin-top:2px">{mean_hand_t}</div>'
        f'<div style="font-size:11px;color:#888;margin-top:4px">Peak (p90): <b style="color:#ccc">{peak_hand_t}</b></div>'
        '</div>'
        '<div style="background:#15171c;padding:12px 14px;border-radius:10px;border:1px solid #252830">'
        '<div style="font-size:11px;color:#888;font-weight:600;letter-spacing:0.5px">PALM–FINGERTIP GRADIENT</div>'
        f'<div style="font-size:22px;font-weight:600;color:#f0f0f0;margin-top:2px">{avg_palm_tip}</div>'
        '<div style="font-size:11px;color:#888;margin-top:4px">Palm mean minus fingertip mean</div>'
        '</div>'
        '<div style="background:#15171c;padding:12px 14px;border-radius:10px;border:1px solid #252830">'
        '<div style="font-size:11px;color:#888;font-weight:600;letter-spacing:0.5px">MAX JOINT CONTRAST</div>'
        f'<div style="font-size:22px;font-weight:600;color:#f0f0f0;margin-top:2px">{avg_j_contrast}</div>'
        '<div style="font-size:11px;color:#888;margin-top:4px">Peak joint minus surrounding ring</div>'
        '</div>'
        '<div style="background:#15171c;padding:12px 14px;border-radius:10px;border:1px solid #252830">'
        '<div style="font-size:11px;color:#888;font-weight:600;letter-spacing:0.5px">STREAM QUALITY VALIDATION</div>'
        f'<div style="font-size:22px;font-weight:600;color:{"#7fd9a6" if n_pass_all == n_rec else ("#ffb84d" if n_pass_all > 0 else "#ff8c8c")};margin-top:2px">'
        f'{n_pass_all} / {n_rec} <span style="font-size:13px;font-weight:normal;color:#888">Passed</span>'
        '</div>'
        f'<div style="font-size:11px;color:#888;margin-top:4px">Cut threshold: {cut_p95:.2f}</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # =========================================================================
    # 1. STREAM-SPECIFIC SCREENING ASSESSMENT
    # =========================================================================
    if n_rec > 1:
        st.subheader("Select Stream to Assess")
        rec_ids = [r["rec_id"] for r in rec_rows]
        sel_rec = st.selectbox("Active Stream", rec_ids, key="rec_sel")
    else:
        sel_rec = rec_rows[0]["rec_id"]

    row = next(r for r in rec_rows if r["rec_id"] == sel_rec)
    frames = saved_frames.get(sel_rec, [])
    d = float(row["d_cohort_deviation"]) if row["d_cohort_deviation"] is not None else np.nan

    if not row["gate_pass"]:
        status_key = "RECAPTURE_RECOMMENDED"
    elif np.isfinite(d) and d > cut_p95:
        status_key = "OUTLIER_VS_COHORT"
    else:
        status_key = "WITHIN_COHORT_RANGE"

    st.subheader(f"Screening Assessment: {sel_rec}")

    c1, c2, c3 = st.columns(3)
    with c1:
        if status_key == "RECAPTURE_RECOMMENDED":
            st.error("**Data Insufficient · Recapture Recommended**\n\nThis stream did not meet quality gate criteria (insufficient stable frames or tracking confidence).")
        elif status_key == "OUTLIER_VS_COHORT":
            st.warning(f"**Atypical Cohort Distance (Review)**\n\nThis stream's cohort distance ({d:.2f}) exceeds the reference 95th percentile ({cut_p95:.2f}).")
        else:
            st.success(f"**Within Cohort Reference Range**\n\nThis stream passed all quality criteria and its thermal features fall within normal cohort range (≤ {cut_p95:.2f}).")

    with c2:
        delta_val = f"{d - cut_p95:+.2f} vs cut_p95" if np.isfinite(d) else None
        st.metric("Stream Cohort Distance ($d$)", f"{d:.2f}" if np.isfinite(d) else "N/A", delta=delta_val, delta_color="inverse")

    with c3:
        n_acc_s = row["n_accepted"]
        n_samp_s = row["n_sampled"]
        gate_status_label = "Passed" if row["gate_pass"] else "Failed"
        st.metric("Quality Gate Decision", f"{gate_status_label} ({n_acc_s}/{n_samp_s} frames)",
                  help=f"Accept fraction: {row['accept_frac']*100:.0f}% (min 50% required)")

    rej = {}
    for fr in frames:
        for why in fr["reasons"]:
            rej[why] = rej.get(why, 0) + 1
    steady_ok = rej.get("jump", 0) == 0 and rej.get("mask_unstable", 0) == 0
    shape_ok = rej.get("finger_len", 0) == 0 and rej.get("scale", 0) == 0
    n_detected = sum(fr["detected"] for fr in frames)
    chips = [
        (f"Thermal joints detected {n_detected}/{len(frames)}", n_detected >= ACCEPT_MIN_FRAC * len(frames)),
        ("Hand in frame", row["q_mask_ok_frac"] >= 0.85),
        ("Hand steady", steady_ok),
        ("Hand shape consistent", shape_ok),
        ("Forearm removed", True),
    ]
    contributors = []
    for t in (explain_details.get(sel_rec) or [])[:4]:
        contributors.append({"name": friendly(t["feature"]), "z": float(t["robust_z"])})

    st.markdown(
        result_card_html(d, cut_p95, status_key, contributors, chips,
                         int(row["n_accepted"]), int(row["n_sampled"]),
                         rec_id=sel_rec),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # =========================================================================
    # 2. FRAME-BY-FRAME INSPECTOR (Slider Updates Live Frame Stats Only)
    # =========================================================================
    st.subheader(f"Interactive Frame Inspector: {sel_rec}")
    st.caption("Slide across frames to inspect frame-specific thermal measurements, geometry, joint rigidity, and deviations.")

    if frames:
        c_slider, c_view = st.columns([2.5, 1.2])
        with c_slider:
            if len(frames) > 1:
                fi = st.slider("Select Frame to Inspect", min_value=0, max_value=len(frames) - 1, value=0, key="frame_sel",
                               format="Frame %d")
            else:
                fi = 0
        with c_view:
            thermal_placement = st.selectbox(
                "Thermal View Position",
                ["Side-by-Side (Split)", "Top-Right Inset", "Top-Left Inset", "Bottom-Right Inset", "Bottom-Left Inset", "Hidden"],
                index=0,
                key="thermal_box_pos",
                help="Move or unstack the thermal scored image so it does not overlap the hand."
            )

        f = frames[fi]
        has_rgb = f["rgb"] is not None
        left, right = st.columns([1.1, 1], gap="medium")
        with left:
            st.markdown(
                rgb_panel_html(
                    to_b64(f["rgb"]) if has_rgb else "",
                    to_b64(f["thermal"]),
                    row["gate_pass"],
                    has_rgb,
                    position=thermal_placement,
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                f"Frame {f['ti']} · {f['status']} · mask {f['area'] * 100:.1f}% · "
                f"joints inside mask {f['valid_inside']}/21"
            )
        with right:
            # Here we show strictly that frame's stats!
            st.markdown(
                frame_stats_card_html(f, bundle=bundle_active),
                unsafe_allow_html=True,
            )
    else:
        st.info("No frames were produced for this recording.")

    # explainability below
    st.markdown("---")
    st.subheader("Feature Deviation & Quality Tables")
    st.caption("Per-recording feature deviations, quality gate decisions and detector health.")
    st.markdown(f"**Top Metric Drivers for {sel_rec}** (Ranked by Cohort Robust Deviation)")
    drivers = explain_details.get(sel_rec, [])
    if drivers:
        expl_df = pd.DataFrame([
            {
                "Thermal Feature Metric": friendly(t["feature"]),
                "Observed Value": f"{t['raw_value']:.2f}",
                "Cohort Deviation (Robust z-score)": f"{t['robust_z']:+.2f}",
            }
            for t in drivers
        ])
        st.dataframe(expl_df, width="stretch", hide_index=True)
    else:
        st.write("Insufficient accepted frames to compute feature deviations.")

    st.markdown(f"**Frame-by-Frame Quality Gate History for {sel_rec}**")
    if frames:
        fg_df = pd.DataFrame([
            {
                "Frame #": fr["ti"],
                "Quality Status": "Accepted" if fr["accepted"] else "Rejected",
                "Rejection Reason(s)": ", ".join([friendly_reason(r) for r in fr["reasons"]]) if fr["reasons"] else "None (Passed)",
                "Hand Landmarks Detected": "Yes" if fr["detected"] else "No",
                "Hand Mask Area": f"{round(fr['area'] * 100, 1)}%",
                "Tracking IoU (vs Prev Frame)": "N/A" if fr["iou_prev"] is None else f"{fr['iou_prev']:.3f}",
                "Motion Jump (px)": "N/A" if fr["jump_px"] is None else f"{fr['jump_px']:.1f}",
            }
            for fr in frames
        ])
        st.dataframe(fg_df, width="stretch", hide_index=True)

    st.markdown("**All Analyzed Recordings Summary**")
    clean_rec_rows = []
    for r in rec_rows:
        dev = r.get("d_cohort_deviation")
        m_frac = r.get("q_mask_ok_frac")
        a_cv = r.get("q_area_cv")
        m_frac_txt = f"{m_frac * 100:.0f}%" if m_frac is not None and np.isfinite(m_frac) else "N/A"
        a_cv_txt = f"{a_cv:.3f}" if a_cv is not None and np.isfinite(a_cv) else "N/A"
        clean_rec_rows.append({
            "Recording ID": r.get("rec_id"),
            "Screening Status": "Within Cohort Range" if (r.get("gate_pass") and dev is not None and np.isfinite(dev) and dev <= cut_p95)
                                else ("Atypical Distance (>95th)" if r.get("gate_pass") else "Recapture Needed"),
            "Cohort Distance (d)": f"{dev:.2f}" if dev is not None and np.isfinite(dev) else "N/A",
            "Gate Passed": "Yes" if r.get("gate_pass") else "No",
            "Accepted Frames": f"{r.get('n_accepted', 0)} / {r.get('n_sampled', 0)}",
            "Mask Valid Frac": m_frac_txt,
            "Area Stability (CV)": a_cv_txt,
        })
    st.dataframe(pd.DataFrame(clean_rec_rows), width="stretch", hide_index=True)

    st.markdown("**Frame Rejection Causes Across Entire Run**")
    REASON_MAP = {
        "area_small": "Hand Area Too Small (<5% of frame)",
        "area_large": "Hand Area Too Large (>60% of frame)",
        "mask_unstable": "Hand Segmentation Instability / Noise",
        "no_detection": "No Hand Landmarks Detected",
        "inside_too_few": "Too Few Joints Inside Hand Mask (<11)",
        "scale": "Hand Depth / Scale Fluctuation",
        "jump": "Rapid Motion Jump / Blur",
        "finger_len": "Finger Inconsistent / Occluded",
        "low_temp": "Hand Temperature Out of Range (<22°C)",
        "high_temp": "Hand Temperature Out of Range (>40°C)",
    }
    if reject_counts:
        rej_df = pd.DataFrame([
            {
                "Rejection Reason": REASON_MAP.get(k, k.replace("_", " ").title()),
                "Rejected Frame Count": v,
            }
            for k, v in sorted(reject_counts.items())
        ])
        st.dataframe(rej_df, width="stretch", hide_index=True)
    else:
        st.write("No frames rejected.")

    if det_errors_all:
        with st.expander(f"Detector errors ({len(det_errors_all)})"):
            st.code("\n".join(det_errors_all[:20]))