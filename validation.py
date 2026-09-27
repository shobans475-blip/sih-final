"""
Document Validation & Checksum Verification Service
Includes Aadhaar Verhoeff algorithm, PAN syntax & entity logic, Passport validation,
DOB/age verification, and cross-field fuzzy consistency checks.
"""

import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Verhoeff algorithm multiplication table d
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

# Verhoeff algorithm permutation table p
VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

# Verhoeff inverse table inv
VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

PAN_ENTITY_TYPES = {
    'P': 'Individual / Person',
    'C': 'Company',
    'H': 'Hindu Undivided Family (HUF)',
    'F': 'Partnership Firm / LLP',
    'A': 'Association of Persons (AOP)',
    'T': 'Trust',
    'B': 'Body of Individuals (BOI)',
    'L': 'Local Authority',
    'J': 'Artificial Juridical Person',
    'G': 'Government Agency'
}


def validate_verhoeff(num_str: str) -> bool:
    """
    Validate a number string with checksum using Verhoeff algorithm (UIDAI Aadhaar standard).
    """
    clean_num = re.sub(r'\D', '', num_str)
    if not clean_num or len(clean_num) != 12:
        return False
    # UIDAI specification: Aadhaar numbers do not begin with 0 or 1
    if clean_num[0] in ('0', '1'):
        return False

    c = 0
    # Process digits in reverse order
    for i, digit in enumerate(reversed(clean_num)):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


def generate_verhoeff(num_str: str) -> str:
    """
    Generate the Verhoeff checksum digit for an 11-digit base number to make a valid 12-digit Aadhaar.
    """
    clean_num = re.sub(r'\D', '', num_str)
    c = 0
    for i, digit in enumerate(reversed(clean_num)):
        c = VERHOEFF_D[c][VERHOEFF_P[(i + 1) % 8][int(digit)]]
    return str(VERHOEFF_INV[c])


def fuzzy_string_similarity(str1: str, str2: str) -> float:
    """
    Calculate normalized Levenshtein token similarity between two strings (0.0 to 1.0).
    """
    if not str1 or not str2:
        return 0.0
    
    s1 = re.sub(r'[^a-zA-Z0-9\s]', '', str1.lower()).strip()
    s2 = re.sub(r'[^a-zA-Z0-9\s]', '', str2.lower()).strip()

    if s1 == s2:
        return 1.0
    
    # Token matching
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if tokens1 and tokens2:
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        jaccard = len(intersection) / len(union)
    else:
        jaccard = 0.0

    # Basic Levenshtein distance
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    
    max_len = max(m, n)
    lev_score = 1.0 - (dp[m][n] / max_len) if max_len > 0 else 1.0
    
    # Combined score
    return round((lev_score * 0.7) + (jaccard * 0.3), 3)


def parse_and_validate_date(date_str: str) -> Tuple[bool, Optional[datetime], str]:
    """
    Tries multiple date formats and verifies whether the date is realistic for an ID document.
    """
    if not date_str:
        return False, None, "Date string is empty"
    
    clean_date = date_str.strip()
    patterns = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y",
        "%d %b %Y", "%d %B %Y", "%Y/%m/%d"
    ]
    parsed = None
    for p in patterns:
        try:
            parsed = datetime.strptime(clean_date, p)
            break
        except ValueError:
            continue
    
    if not parsed:
        # Check if year only (common in older Aadhaar cards e.g. "Year of Birth: 1985")
        year_match = re.search(r'\b(19\d{2}|20[0-2]\d)\b', clean_date)
        if year_match:
            try:
                parsed = datetime(int(year_match.group(1)), 1, 1)
                return True, parsed, "Year-only birth record detected"
            except Exception:
                pass
        return False, None, f"Unrecognized date format: '{date_str}'"

    current_year = datetime.now().year
    if parsed.year < 1900:
        return False, parsed, f"Year {parsed.year} is too far in the past"
    if parsed > datetime.now():
        return False, parsed, "Date is in the future"

    age = current_year - parsed.year
    if age < 0 or age > 120:
        return False, parsed, f"Calculated age ({age}) is out of reasonable human range"

    return True, parsed, f"Valid date (Age ~{age} years)"


def validate_pan(pan_number: str, holder_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate Indian Permanent Account Number (PAN) structure and logic.
    Format: 5 uppercase letters, 4 digits, 1 uppercase letter.
    4th letter = Entity Status.
    5th letter = First character of applicant's surname (for individual 'P').
    """
    clean_pan = pan_number.strip().upper()
    checks = []
    is_valid = True
    entity_desc = "Unknown"

    pattern = r'^[A-Z]{3}([A-Z])([A-Z])[0-9]{4}[A-Z]$'
    match = re.match(pattern, clean_pan)

    if not match:
        return {
            "is_valid": False,
            "entity_type": "Invalid Format",
            "surname_matched": None,
            "checks": [{
                "field": "PAN Syntax",
                "passed": False,
                "rule": "10-character alphanumeric (AAAAA0000A)",
                "details": f"Value '{clean_pan}' does not adhere to standard Indian Income Tax PAN syntax"
            }]
        }

    status_char = match.group(1)
    surname_char = match.group(2)

    # 1. Entity type check
    if status_char in PAN_ENTITY_TYPES:
        entity_desc = PAN_ENTITY_TYPES[status_char]
        checks.append({
            "field": "Entity Status (4th char)",
            "passed": True,
            "rule": "Valid entity code",
            "details": f"Character '{status_char}' corresponds to: {entity_desc}"
        })
    else:
        is_valid = False
        checks.append({
            "field": "Entity Status (4th char)",
            "passed": False,
            "rule": "Valid entity code",
            "details": f"Invalid entity indicator '{status_char}'"
        })

    # 2. Individual surname check
    surname_matched = None
    if status_char == 'P' and holder_name:
        name_parts = holder_name.strip().split()
        if name_parts:
            surname = name_parts[-1].upper()
            if surname and surname[0] == surname_char:
                surname_matched = True
                checks.append({
                    "field": "Surname Alignment (5th char)",
                    "passed": True,
                    "rule": "5th char matches 1st letter of surname",
                    "details": f"5th character '{surname_char}' matches surname '{surname}'"
                })
            else:
                surname_matched = False
                checks.append({
                    "field": "Surname Alignment (5th char)",
                    "passed": False,
                    "rule": "5th char matches 1st letter of surname",
                    "details": f"Warning: 5th character '{surname_char}' does not match expected surname initial '{surname[0] if surname else '?'}'"
                })

    return {
        "is_valid": is_valid,
        "clean_pan": clean_pan,
        "entity_type": entity_desc,
        "surname_matched": surname_matched,
        "checks": checks
    }


def validate_document_data(
    doc_type: str,
    extracted_fields: Dict[str, Any],
    claimed_identity: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive multi-field validation for extracted document data.
    """
    doc_type_upper = (doc_type or "").upper()
    validation_checks: List[Dict[str, Any]] = []
    flags: List[str] = []
    overall_score = 100.0

    # 1. Document ID / Number validation
    doc_num = extracted_fields.get("document_number", "")
    clean_num = re.sub(r'[\s-]', '', str(doc_num)).strip()

    if "AADHAAR" in doc_type_upper:
        digits_only = re.sub(r'\D', '', clean_num)
        if len(digits_only) == 12:
            is_verhoeff = validate_verhoeff(digits_only)
            if is_verhoeff:
                validation_checks.append({
                    "field": "Aadhaar Verhoeff Checksum",
                    "passed": True,
                    "rule": "UIDAI Dihedral D5 Checksum",
                    "details": "Mathematical checksum passed. Number sequence is verified authentic."
                })
            else:
                overall_score -= 50
                flags.append("Aadhaar Verhoeff checksum algorithm failed! Number is mathematically invalid or fraudulent.")
                validation_checks.append({
                    "field": "Aadhaar Verhoeff Checksum",
                    "passed": False,
                    "rule": "UIDAI Dihedral D5 Checksum",
                    "details": "FAILED: Digits violate UIDAI D5 checksum algorithm. Number is fabricated."
                })
        else:
            overall_score -= 35
            flags.append(f"Aadhaar requires exactly 12 digits, found {len(digits_only)}")
            validation_checks.append({
                "field": "Aadhaar Length Check",
                "passed": False,
                "rule": "Exact 12 digits",
                "details": f"Invalid digit count: {len(digits_only)}"
            })

    elif "PAN" in doc_type_upper:
        pan_result = validate_pan(clean_num, extracted_fields.get("full_name"))
        validation_checks.extend(pan_result["checks"])
        if not pan_result["is_valid"]:
            overall_score -= 40
            flags.append("Invalid PAN card format or entity character.")
        if pan_result.get("surname_matched") is False:
            overall_score -= 15
            flags.append("PAN 5th letter does not match extracted cardholder surname.")

    elif "PASSPORT" in doc_type_upper:
        passport_match = re.match(r'^[A-Z][0-9]{7}$', clean_num)
        if passport_match:
            validation_checks.append({
                "field": "Passport Syntax Check",
                "passed": True,
                "rule": "1 Uppercase letter followed by 7 digits",
                "details": "Standard ICAO compliant Indian passport number format passed."
            })
        else:
            overall_score -= 30
            flags.append(f"Passport number '{clean_num}' does not adhere to standard 8-character format.")
            validation_checks.append({
                "field": "Passport Syntax Check",
                "passed": False,
                "rule": "1 Uppercase letter followed by 7 digits",
                "details": f"Invalid format: '{clean_num}'"
            })

    # 2. Date of Birth Check
    dob_val = extracted_fields.get("dob")
    if dob_val:
        valid_dob, parsed_dt, dob_msg = parse_and_validate_date(str(dob_val))
        validation_checks.append({
            "field": "Date of Birth Integrity",
            "passed": valid_dob,
            "rule": "Valid calendar date within 0-120 years",
            "details": dob_msg
        })
        if not valid_dob:
            overall_score -= 20
            flags.append(f"DOB integrity check issue: {dob_msg}")
    else:
        validation_checks.append({
            "field": "Date of Birth",
            "passed": False,
            "rule": "Presence check",
            "details": "DOB could not be reliably extracted from document."
        })
        overall_score -= 10

    # 3. Cross-verification with Claimed Identity if provided
    name_similarity = None
    if claimed_identity:
        claimed_name = claimed_identity.get("name")
        extracted_name = extracted_fields.get("full_name")
        if claimed_name and extracted_name:
            name_similarity = fuzzy_string_similarity(claimed_name, extracted_name)
            is_match = name_similarity >= 0.70
            validation_checks.append({
                "field": "Claimed Name Match",
                "passed": is_match,
                "rule": "Fuzzy token similarity >= 70%",
                "details": f"Similarity score: {int(name_similarity * 100)}% (Claimed: '{claimed_name}', Found: '{extracted_name}')"
            })
            if not is_match:
                overall_score -= 25
                flags.append(f"Applicant name mismatch: Claimed '{claimed_name}' vs Document '{extracted_name}' ({int(name_similarity*100)}% match).")

        claimed_dob = claimed_identity.get("dob")
        if claimed_dob and dob_val:
            _, parsed_claimed, _ = parse_and_validate_date(claimed_dob)
            _, parsed_extracted, _ = parse_and_validate_date(str(dob_val))
            if parsed_claimed and parsed_extracted:
                dob_match = (parsed_claimed.year == parsed_extracted.year and
                             parsed_claimed.month == parsed_extracted.month and
                             parsed_claimed.day == parsed_extracted.day)
                validation_checks.append({
                    "field": "Claimed DOB Match",
                    "passed": dob_match,
                    "rule": "Exact calendar date match",
                    "details": f"Claimed '{claimed_dob}' vs Extracted '{dob_val}'"
                })
                if not dob_match:
                    overall_score -= 25
                    flags.append(f"DOB mismatch: Claimed '{claimed_dob}' vs Extracted '{dob_val}'.")

    final_score = max(0.0, min(100.0, overall_score))
    return {
        "validation_score": round(final_score, 1),
        "is_valid": final_score >= 65.0 and len([c for c in validation_checks if not c["passed"] and "Checksum" in c["field"]]) == 0,
        "checks": validation_checks,
        "flags": flags,
        "name_similarity": name_similarity
    }
