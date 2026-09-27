"""
AI Screening Assistant & Verification Copilot Service
Provides intelligent conversational assistance for document analysis,
explaining tamper heatmaps, Verhoeff mathematical checksums, and KYC compliance.
"""

import os
import re
from typing import Dict, Any, List, Optional

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


SYSTEM_PROMPT = """
You are 'TrustLens Copilot', an expert AI Forensic Document Auditor and Compliance Assistant.
Your job is to assist KYC verification officers, government evaluators, and users in understanding
document screening results, forgery detection heatmaps, checksum algorithms, and identity verification decisions.

Always be precise, professional, objective, and provide actionable recommendations.
When discussing Aadhaar, reference the UIDAI Verhoeff D5 algorithm.
When discussing PAN cards, explain the 4th/5th character entity code logic.
When discussing Error Level Analysis (ELA), explain how JPEG compression artifacts expose spliced regions.
"""


def generate_offline_response(user_query: str, screening_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Intelligent rule-based copilot response when Gemini API is offline or unconfigured.
    """
    query_lower = user_query.lower()
    
    # Context data extraction
    ocr = screening_context.get("ocr", {}) if screening_context else {}
    risk = screening_context.get("risk", {}) if screening_context else {}
    validation = screening_context.get("validation", {}) if screening_context else {}
    tamper = screening_context.get("tamper", {}) if screening_context else {}
    face = screening_context.get("face", {}) if screening_context else {}

    doc_type = ocr.get("document_type", "Document")
    risk_score = risk.get("risk_score", 0.0)
    risk_tier = risk.get("risk_tier", "PENDING")

    # 1. Why was this document flagged?
    if any(k in query_lower for k in ["why", "flagged", "rejected", "risk", "reason", "score"]):
        flags = risk.get("flags", [])
        if not flags:
            return (
                f"### Analysis for {doc_type} (Risk Score: {risk_score}/100 - {risk_tier})\n\n"
                f"This document is classified as **{risk_tier}**. No critical security flags were triggered. "
                f"The mathematical checksums and forensic heatmaps indicate the document is authentic and unaltered."
            )
        
        response = [
            f"### Security Audit Breakdown for {doc_type}\n",
            f"**Current Status:** `{risk_tier}` (Composite Risk Score: **{risk_score}/100**)\n",
            "The following key factors influenced this assessment:\n"
        ]
        for f in flags:
            response.append(f"- **[{f['severity']}] {f['category']}**: {f['message']}")
        
        response.append(f"\n**Officer Recommendation:** {risk.get('recommendation', 'Inspect manually.')}")
        return "\n".join(response)

    # 2. Explaining ELA (Error Level Analysis)
    elif any(k in query_lower for k in ["ela", "heatmap", "error level", "tamper", "splic"]):
        tamper_score = tamper.get("tamper_score", 0.0)
        return (
            "### How Error Level Analysis (ELA) Works\n\n"
            "**Error Level Analysis (ELA)** works by recompressing the image at a known uniform JPEG quality (90%) "
            "and calculating the difference between the original and recompressed pixel values:\n\n"
            "1. **Uniform Background**: An authentic, untouched photo has a relatively uniform error rate across similar textures.\n"
            "2. **Digital Splicing / Alterations**: Areas modified using Photoshop, Canva, or MS Paint (such as modified DOBs or names) "
            "have undergone different compression cycles and highlight as **bright high-error peaks** on the heatmap (shown in bright red/yellow on the JET scale).\n\n"
            f"**This Document's Forensic Metrics:**\n"
            f"- ELA Score: **{tamper.get('ela_score', 'N/A')}/100**\n"
            f"- Tamper Score: **{tamper_score}/100**\n"
            f"- Status: {'⚠️ Forgery Suspicion High' if tamper.get('is_tampered') else '✅ Uniform compression confirmed.'}"
        )

    # 3. Explaining Verhoeff Checksum
    elif any(k in query_lower for k in ["verhoeff", "aadhaar", "checksum", "algorithm"]):
        verhoeff_passed = any(c.get("passed") for c in validation.get("checks", []) if "Verhoeff" in c.get("field", ""))
        return (
            "### Aadhaar Verhoeff Checksum Verification\n\n"
            "The **Verhoeff algorithm** is a mathematical error-detecting formula based on the non-commutative dihedral group $D_5$:\n\n"
            "- It was designed by Dutch mathematician Jacobus Verhoeff in 1969 and is adopted by **UIDAI** for 12-digit Aadhaar numbers.\n"
            "- It detects **all single-digit errors** and **all adjacent transposition errors**.\n"
            "- The 12th digit in an Aadhaar number is a mathematically calculated check digit based on permutation and multiplication matrices.\n\n"
            f"**Validation on this Document:**\n"
            f"- Checksum Status: **{'✅ PASSED (Mathematically Authentic)' if verhoeff_passed else '❌ FAILED (Invalid Check Digit)'}**\n"
            "- If an applicant changes even one digit of their Aadhaar number, the Verhoeff equation evaluates to non-zero, flagging the fraud instantly."
        )

    # 4. Explaining PAN Card logic
    elif any(k in query_lower for k in ["pan", "income tax", "entity"]):
        pan_num = ocr.get("document_number", "ABCDE1234F")
        return (
            "### Indian PAN Card Structural Logic\n\n"
            "A Permanent Account Number (PAN) is a 10-character alphanumeric code with strict structural rules:\n\n"
            "1. **Characters 1-3**: Random alphabetic sequence from `AAA` to `ZZZ`.\n"
            "2. **Character 4**: **Entity Status** (`P` = Individual/Person, `C` = Company, `H` = HUF, `F` = Firm, `T` = Trust, `G` = Government).\n"
            "3. **Character 5**: **Surname Initial** (For individual PAN 'P', this MUST match the first letter of the applicant's surname).\n"
            "4. **Characters 6-9**: Sequential digits from `0001` to `9999`.\n"
            "5. **Character 10**: Alphabetic check digit.\n\n"
            f"**Document PAN:** `{pan_num}`"
        )

    # 5. Explaining Biometric / Face Verification
    elif any(k in query_lower for k in ["face", "selfie", "biometric", "liveness"]):
        match_score = face.get("match_score", 0.0)
        is_match = face.get("is_match", False)
        return (
            "### Biometric Verification & Anti-Spoofing\n\n"
            "Our biometric pipeline performs multi-stage facial verification:\n\n"
            "1. **ID Face Extraction**: Isolates and normalizes the portrait photo on the document.\n"
            "2. **Live Selfie Alignment**: Detects the live applicant's face from the webcam or uploaded selfie.\n"
            "3. **Facial Feature Matching**: Measures HSV color distribution, Normalized Cross-Correlation, and structural contours.\n"
            "4. **Passive Liveness**: Inspects for electronic screen moiré artifacts (indicating someone took a picture of an iPad/phone screen) and Laplacian blur.\n\n"
            f"**Biometric Result:**\n"
            f"- Similarity: **{match_score}%** ({'✅ MATCH' if is_match else '❌ MISMATCH'})\n"
            f"- Liveness Score: **{face.get('liveness_score', 'N/A')}%**"
        )

    # Default general guidance
    return (
        f"### TrustLens AI Copilot\n\n"
        f"I am actively monitoring the screening process for this **{doc_type}**.\n\n"
        f"- **Current Composite Risk**: {risk_score}/100 ({risk_tier})\n"
        f"- **Action**: {risk.get('action', 'INSPECT')}\n\n"
        "You can ask me:\n"
        "- *'Why was this document flagged?'*\n"
        "- *'Explain the ELA tamper heatmap'* \n"
        "- *'How does the Aadhaar Verhoeff algorithm work?'*\n"
        "- *'Explain the PAN entity code rules'*\n"
        "- *'What are the biometric match findings?'*"
    )


def chat_with_copilot(
    message: str,
    screening_context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Handles user chat query regarding document screening.
    Uses Gemini 2.5 Flash if GEMINI_API_KEY is configured, else falls back to offline knowledge engine.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")

    if gemini_key and gemini_key != "your_gemini_api_key_here" and GEMINI_AVAILABLE:
        try:
            client = genai.Client(api_key=gemini_key)
            context_summary = ""
            if screening_context:
                ocr = screening_context.get("ocr", {})
                risk = screening_context.get("risk", {})
                validation = screening_context.get("validation", {})
                tamper = screening_context.get("tamper", {})
                face = screening_context.get("face", {})
                context_summary = f"""
CURRENT SCREENING DOSSIER:
- Document Type: {ocr.get('document_type')}
- Extracted ID Number: {ocr.get('document_number')}
- Extracted Full Name: {ocr.get('full_name')}
- Extracted DOB: {ocr.get('dob')}
- Risk Score: {risk.get('risk_score')}/100 ({risk.get('risk_tier')})
- Risk Flags: {risk.get('flags')}
- Validation Checks: {validation.get('checks')}
- Tamper Score: {tamper.get('tamper_score')}/100 (Is Tampered: {tamper.get('is_tampered')})
- Tamper Findings: {tamper.get('findings')}
- Face Match Score: {face.get('match_score')}% (Is Match: {face.get('is_match')})
- Liveness Score: {face.get('liveness_score')}%
"""
            full_prompt = f"{SYSTEM_PROMPT}\n\n{context_summary}\n\nUser Question: {message}"
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=full_prompt,
            )
            return {
                "reply": response.text.strip(),
                "engine": "Gemini 2.5 Flash AI"
            }
        except Exception as e:
            print(f"Gemini copilot error: {e}, using offline engine.")

    # Offline engine
    reply = generate_offline_response(message, screening_context)
    return {
        "reply": reply,
        "engine": "TrustLens Offline Compliance Engine"
    }
