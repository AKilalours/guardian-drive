"""
policy/medical_guardrail.py
Guardian Drive -- Medical Grade Guardrail

Enforces honest boundaries between research prototype
and clinical medical device.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

SYSTEM_CLAIMS = {
    "medical_grade":           False,
    "clinically_validated":    False,
    "fda_cleared":             False,
    "ce_marked":               False,
    "iso_13485_certified":     False,
    "iec_62304_compliant":     False,
    "intended_use":            "Research prototype and demonstration only",
    "not_intended_for":        [
        "Clinical diagnosis",
        "Treatment decisions",
        "Emergency dispatch",
        "Vehicle control",
        "Life-critical decisions",
    ],
    "regulatory_status":       "Not submitted to FDA or any regulatory body",
    "data_disclaimer":         (
        "WESAD AUC 0.9738 uses window-level split with known leakage. "
        "LOSO AUC 0.6712 is the honest generalization estimate. "
        "PTBDB Task A AUC 0.6378 is binary normal/abnormal, not multi-class."
    ),
    "emergency_disclaimer":    (
        "Emergency response workflow is prototype simulation only. "
        "Not connected to real emergency services. "
        "911 button is UI mockup. Autopilot banner is display only."
    ),
}

def check_claim(claim: str) -> bool:
    """Return True only for verified claims."""
    return bool(SYSTEM_CLAIMS.get(claim, False))

def get_disclaimer() -> str:
    return (
        "Guardian Drive is a research prototype. "
        "Not a medical device. Not clinically validated. "
        f"Intended use: {SYSTEM_CLAIMS['intended_use']}. "
        f"{SYSTEM_CLAIMS['data_disclaimer']}"
    )

def validate_output(output: dict) -> dict:
    """
    Add mandatory disclaimer to any pipeline output.
    Ensures no output is presented without context.
    """
    output["medical_grade"]  = False
    output["disclaimer"]     = get_disclaimer()
    output["regulatory"]     = SYSTEM_CLAIMS["regulatory_status"]
    return output

if __name__ == "__main__":
    import json
    print("Guardian Drive -- Medical Guardrail Status")
    print(json.dumps(SYSTEM_CLAIMS, indent=2))
    print("\nDisclaimer:")
    print(get_disclaimer())
