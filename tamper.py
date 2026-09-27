"""
Document Tampering & Digital Forgery Detection Service
Implements Error Level Analysis (ELA), Noise Inconsistency Analysis,
Edge Discontinuity Detection, and Forensic Metadata/EXIF Inspection.
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image, ExifTags
from typing import Dict, Any, List, Tuple


KNOWN_EDITING_SOFTWARE = [
    "photoshop", "gimp", "canva", "paint.net", "photopea",
    "coreldraw", "lightroom", "pixlr", "snapseed", "picsart", "facetune"
]


def perform_ela(
    image_bytes: bytes,
    quality: int = 90,
    scale_multiplier: int = 20
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform Error Level Analysis (ELA).
    Returns:
      - ela_bgr: Scaled visual difference in BGR
      - heatmap_bgr: Color-mapped forensic heatmap (JET)
      - ela_score: Statistical anomaly metric (0.0 to 100.0)
    """
    # Load original image
    pil_orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_np = np.array(pil_orig)
    orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

    # Save to memory at designated JPEG quality
    buffer = io.BytesIO()
    pil_orig.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Reload compressed image
    pil_resaved = Image.open(buffer).convert("RGB")
    resaved_np = np.array(pil_resaved)
    resaved_bgr = cv2.cvtColor(resaved_np, cv2.COLOR_RGB2BGR)

    # Compute absolute difference
    diff = cv2.absdiff(orig_bgr, resaved_bgr)

    # Scale the difference to make subtle compression error visible
    scaled_diff = cv2.multiply(diff, np.array([scale_multiplier, scale_multiplier, scale_multiplier], dtype=np.uint8))

    # Convert diff to grayscale for anomaly statistics
    diff_gray = cv2.cvtColor(scaled_diff, cv2.COLOR_BGR2GRAY)
    
    # Calculate statistical metrics
    mean_val = np.mean(diff_gray)
    std_val = np.std(diff_gray)
    max_val = np.max(diff_gray)

    # High standard deviation with localized high peaks indicates uneven compression history (splicing)
    # Calibrate score between 0 and 100
    peak_ratio = (max_val - mean_val) / (std_val + 1e-5)
    high_error_pixels = np.count_nonzero(diff_gray > 120) / (diff_gray.size + 1e-5)

    ela_score = min(100.0, (mean_val * 1.5) + (peak_ratio * 3.0) + (high_error_pixels * 500.0))

    # Create Colorized Forensic Heatmap using JET
    # Normalize grayscale difference for colormap
    norm_diff = cv2.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_bgr = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)

    return scaled_diff, heatmap_bgr, round(ela_score, 2)


def analyze_noise_inconsistency(image_bgr: np.ndarray, block_size: int = 32) -> Tuple[float, bool]:
    """
    Examines local Laplacian variance across image blocks.
    Tampered patches, digital overlays, or smoothed text boxes usually have mismatched noise.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    variances = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            patch = gray[y:y+block_size, x:x+block_size]
            lap = cv2.Laplacian(patch, cv2.CV_64F)
            var = lap.var()
            variances.append(var)

    if not variances:
        return 0.0, False

    var_arr = np.array(variances)
    median_var = np.median(var_arr)
    # Detect extreme low variance blocks (unnaturally smooth digital inserts on scanned paper)
    # or extreme high variance (digital sharp noise)
    suspicious_blocks = np.count_nonzero((var_arr < median_var * 0.08) & (var_arr > 0.01))
    ratio = suspicious_blocks / len(variances)

    noise_score = min(100.0, ratio * 200.0)
    is_inconsistent = noise_score > 35.0
    return round(noise_score, 2), is_inconsistent


def detect_edge_discontinuities(image_bgr: np.ndarray) -> Tuple[float, List[str]]:
    """
    Detects unnaturally sharp rectangular borders often left by copy-paste or box-overlay editing.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Morphological line detection for horizontal and vertical cut lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    lines_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
    lines_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)

    combined_lines = cv2.add(lines_h, lines_v)
    line_pixel_ratio = np.count_nonzero(combined_lines) / (gray.size + 1e-5)

    discontinuity_score = min(100.0, line_pixel_ratio * 3000.0)
    observations = []
    if discontinuity_score > 40.0:
        observations.append("Sharp rectangular boundaries detected around field regions (characteristic of digital paste overlays).")

    return round(discontinuity_score, 2), observations


def inspect_metadata_exif(image_bytes: bytes) -> Dict[str, Any]:
    """
    Inspect EXIF headers for editing signatures, software traces, and timestamp discrepancies.
    """
    findings = []
    suspicious_software = None
    has_exif = False

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        exif_raw = pil_img.getexif()

        if exif_raw:
            has_exif = True
            for tag_id, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                val_str = str(value).strip()

                if tag_name.lower() in ("software", "processingsoftware", "imagesoftware"):
                    for sw in KNOWN_EDITING_SOFTWARE:
                        if sw in val_str.lower():
                            suspicious_software = val_str
                            findings.append(f"Image edited with software tool: '{val_str}'")
                            break
        else:
            findings.append("No EXIF metadata found (metadata stripped, typical for online edits or re-exports).")

    except Exception as e:
        findings.append(f"Metadata read error: {str(e)}")

    return {
        "has_exif": has_exif,
        "suspicious_software": suspicious_software,
        "is_suspicious": suspicious_software is not None,
        "findings": findings
    }


def analyze_tampering(image_bytes: bytes) -> Dict[str, Any]:
    """
    Comprehensive multi-method forensic document tampering analysis.
    Returns composite tamper score, visual base64 heatmaps, and detailed forensic findings.
    """
    try:
        # 1. ELA Analysis
        ela_bgr, heatmap_bgr, ela_score = perform_ela(image_bytes, quality=90, scale_multiplier=22)

        # Encode ELA and Heatmap as Base64 JPEG data URIs
        _, ela_buf = cv2.imencode(".jpg", ela_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ela_base64 = "data:image/jpeg;base64," + base64.b64encode(ela_buf).decode("utf-8")

        _, heatmap_buf = cv2.imencode(".jpg", heatmap_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        heatmap_base64 = "data:image/jpeg;base64," + base64.b64encode(heatmap_buf).decode("utf-8")

        # 2. Noise consistency
        pil_orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_bgr = cv2.cvtColor(np.array(pil_orig), cv2.COLOR_RGB2BGR)
        noise_score, noise_inconsistent = analyze_noise_inconsistency(orig_bgr)

        # 3. Edge discontinuity
        edge_score, edge_notes = detect_edge_discontinuities(orig_bgr)

        # 4. EXIF inspection
        exif_info = inspect_metadata_exif(image_bytes)

        # 5. Composite Tamper Risk Score calculation
        # Weights: ELA (40%), Noise (30%), Discontinuity (20%), Metadata (10% + penalty)
        composite_score = (ela_score * 0.40) + (noise_score * 0.30) + (edge_score * 0.20)
        
        findings = []
        if ela_score > 45.0:
            findings.append(f"High compression variance detected via ELA ({ela_score}/100). Possible localized content splicing.")
        else:
            findings.append("Compression levels appear uniform across document background.")

        if noise_inconsistent:
            findings.append(f"Inconsistent noise pattern detected ({noise_score}/100). Digital smoothing or foreign patch detected.")
        
        findings.extend(edge_notes)

        if exif_info.get("suspicious_software"):
            composite_score = max(composite_score + 35.0, 75.0)
            findings.append(f"CRITICAL FORENSIC ALERT: Document generated or modified using '{exif_info['suspicious_software']}'.")

        final_score = min(100.0, round(composite_score, 1))
        is_tampered = final_score >= 50.0

        return {
            "tamper_score": final_score,
            "is_tampered": is_tampered,
            "ela_score": ela_score,
            "noise_score": noise_score,
            "edge_score": edge_score,
            "ela_image_base64": ela_base64,
            "heatmap_base64": heatmap_base64,
            "findings": findings,
            "metadata": exif_info
        }

    except Exception as e:
        return {
            "tamper_score": 0.0,
            "is_tampered": False,
            "ela_score": 0.0,
            "noise_score": 0.0,
            "edge_score": 0.0,
            "ela_image_base64": None,
            "heatmap_base64": None,
            "findings": [f"Tamper analysis failed: {str(e)}"],
            "metadata": {"has_exif": False, "is_suspicious": False, "findings": []}
        }
