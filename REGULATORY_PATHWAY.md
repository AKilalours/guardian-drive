# Guardian Drive -- Regulatory Pathway Analysis

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis

## Current Status
Guardian Drive is a RESEARCH PROTOTYPE.
medical_grade: False
clinically_validated: False

## If Commercialized -- Regulatory Requirements

### United States (FDA)
Classification: Software as a Medical Device (SaMD)
Likely class: Class II (moderate risk)
Pathway: 510(k) premarket notification
Requirements:
- Substantial equivalence to predicate device
- Clinical validation study (prospective, multi-site)
- Quality Management System (ISO 13485)
- Software lifecycle documentation (IEC 62304)
- Cybersecurity documentation (FDA 2023 guidance)
- Post-market surveillance plan

Timeline estimate: 12-18 months
Cost estimate: $200k-$500k

### European Union (EU MDR 2017/745)
Classification: Class IIa (medium risk)
Pathway: CE marking via Notified Body
Requirements:
- Technical documentation
- Clinical evaluation report
- Post-market clinical follow-up
- UDI registration
- Authorized Representative in EU

### NOT Required For:
- Research use only (RUO) -- current status
- Internal development tool
- Academic publication

## Why This Matters for Guardian Drive
Task A arrhythmia screening and Task B drowsiness detection
produce physiological risk signals. If used for clinical
decision-making, FDA/MDR oversight would apply.

Current mitigation: explicit medical_grade: False guardrail,
disclaimer in every output, no clinical deployment.

## Path to Clinical Validation (Future Work)
1. IRB approval for prospective study
2. 500+ patient enrollment
3. Blinded comparison vs cardiologist ground truth
4. Sensitivity/specificity targets: >90%/>85%
5. External validation at second institution
6. FDA Pre-Submission meeting
7. 510(k) submission
