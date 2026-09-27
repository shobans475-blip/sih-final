"""
Composite AI Risk Scoring & Fraud Assessment Engine
Aggregates OCR integrity, Verhoeff/PAN validation, Tamper & ELA forensics,
and Biometric Face Verification into an overall risk score and actionable decision tier.
"""

from typing import Dict, Any, List


def calculate_risk_profile(
    ocr_result: Dict[str, Any],
    validation_result: Dict[str, Any],
    tamper_result: Dict[str, Any],
    face_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates multi-modal evidence across all screening services.
    Generates a normalized composite risk score (0 to 100, where 0 = safe, 100 = critical fraud).
    """
    flags: List[Dict[str, str]] = []

    # 1. OCR Risk Component (0 to 100 risk, inverted from confidence)
    ocr_conf = float(ocr_result.get("ocr_confidence", 0.7))
    doc_number = ocr_result.get("document_number")
    doc_type = ocr_result.get("document_type", "UNKNOWN")

    ocr_risk = max(0.0, (1.0 - ocr_conf) * 100.0)
    if not doc_number:
        ocr_risk = min(100.0, ocr_risk + 40.0)
        flags.append({
            "severity": "CRITICAL",
            "category": "OCR",
            "message": "Critical ID number could not be extracted or is obscured."
        })
    elif doc_type == "UNKNOWN":
        ocr_risk = min(100.0, ocr_risk + 25.0)
        flags.append({
            "severity": "WARNING",
            "category": "OCR",
            "message": "Document format does not match recognized Indian identity standards."
        })

    # 2. Validation Risk Component
    val_score = float(validation_result.get("validation_score", 100.0))
    val_risk = 100.0 - val_score

    for v_check in validation_result.get("checks", []):
        if not v_check["passed"]:
            is_critical = "Checksum" in v_check["field"] or "Syntax" in v_check["field"]
            flags.append({
                "severity": "CRITICAL" if is_critical else "WARNING",
                "category": "Validation",
                "message": f"{v_check['field']}: {v_check['details']}"
            })

    # 3. Tamper Risk Component
    tamper_risk = float(tamper_result.get("tamper_score", 0.0))
    if tamper_result.get("is_tampered"):
        flags.append({
            "severity": "CRITICAL",
            "category": "Forensics",
            "message": f"Digital image tampering detected (Tamper Score: {tamper_risk}/100)."
        })
    if tamper_result.get("metadata", {}).get("suspicious_software"):
        sw = tamper_result["metadata"]["suspicious_software"]
        flags.append({
            "severity": "CRITICAL",
            "category": "Forensics",
            "message": f"Document manipulated using editing software: '{sw}'"
        })

    for finding in tamper_result.get("findings", []):
        if "CRITICAL" in finding or "inconsistent" in finding.lower() or "splicing" in finding.lower():
            flags.append({
                "severity": "WARNING",
                "category": "Forensics",
                "message": finding
            })

    # 4. Biometric Face Risk Component
    bio_risk = 0.0
    selfie_provided = face_result.get("selfie_provided", False)
    doc_face_found = face_result.get("doc_face_found", False)

    if not doc_face_found:
        bio_risk += 35.0
        flags.append({
            "severity": "WARNING",
            "category": "Biometrics",
            "message": "No recognizable face found on the identity document photo."
        })
    elif selfie_provided:
        selfie_face_found = face_result.get("selfie_face_found", False)
        if not selfie_face_found:
            bio_risk += 45.0
            flags.append({
                "severity": "WARNING",
                "category": "Biometrics",
                "message": "Selfie image provided but face could not be detected."
            })
        else:
            match_score = float(face_result.get("match_score", 0.0))
            is_match = face_result.get("is_match", False)
            if not is_match:
                bio_risk += (100.0 - match_score) * 0.8
                flags.append({
                    "severity": "CRITICAL",
                    "category": "Biometrics",
                    "message": f"Biometric mismatch: Live face does not match ID photo ({match_score}% similarity)."
                })
            else:
                bio_risk = max(0.0, (100.0 - match_score) * 0.2)

            if not face_result.get("is_live", True):
                bio_risk = min(100.0, bio_risk + 30.0)
                flags.append({
                    "severity": "CRITICAL",
                    "category": "Biometrics",
                    "message": "Live selfie failed anti-spoofing / passive liveness check (moiré or screen glare)."
                })
    else:
        # No selfie provided; minor baseline risk for unverified identity
        bio_risk = 15.0

    # Composite Risk Calculation
    # Dynamic weighting depending on whether selfie was submitted
    if selfie_provided:
        weights = {"ocr": 0.20, "validation": 0.30, "tamper": 0.30, "bio": 0.20}
    else:
        weights = {"ocr": 0.25, "validation": 0.40, "tamper": 0.35, "bio": 0.00}

    composite_score = (
        (ocr_risk * weights["ocr"]) +
        (val_risk * weights["validation"]) +
        (tamper_risk * weights["tamper"]) +
        (bio_risk * weights["bio"])
    )

    # Any critical Verhoeff or blatant editing software bumps score to high risk immediately
    has_critical_failure = any(f["severity"] == "CRITICAL" for f in flags)
    if has_critical_failure and composite_score < 65.0:
        composite_score = max(composite_score, 68.0)

    final_risk_score = round(max(0.0, min(100.0, composite_score)), 1)

    # Risk Tiering
    if final_risk_score < 30.0:
        tier = "LOW_RISK"
        action = "APPROVE"
        badge_color = "#00e699"
        decision_title = "Document Authenticity Verified"
        recommendation = "All mathematical checksums, forensic analysis, and biometric checks passed. Recommended for automated approval."
    elif final_risk_score < 65.0:
        tier = "MEDIUM_RISK"
        action = "MANUAL_REVIEW"
        badge_color = "#ffb703"
        decision_title = "Manual Inspection Recommended"
        recommendation = "Minor anomalies or partial confidence detected. Route to senior verification officer for manual inspection."
    else:
        tier = "HIGH_RISK"
        action = "REJECT"
        badge_color = "#ff3366"
        decision_title = "Potential Document Forgery / Fraud"
        recommendation = "Critical failure detected in document checksums, biometric matching, or digital image forensics. Recommend immediate rejection."

    return {
        "risk_score": final_risk_score,
        "risk_tier": tier,
        "action": action,
        "badge_color": badge_color,
        "decision_title": decision_title,
        "recommendation": recommendation,
        "breakdown": {
            "ocr_risk": round(ocr_risk, 1),
            "validation_risk": round(val_risk, 1),
            "tamper_risk": round(tamper_risk, 1),
            "biometric_risk": round(bio_risk, 1)
        },
        "flags": flags
    }
