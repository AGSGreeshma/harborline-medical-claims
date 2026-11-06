"""Masking must catch Indian identifiers and health data, and must NOT touch amounts or the hospital name."""
from harborline.pii import has_placeholder, mask, restore

DOC = """Sunrise Multispeciality Hospital
INPATIENT FINAL BILL
Patient Name: Rohan Mehta        Age/Sex: 42/M
UHID: SMH-2024-88123   Policy No: HLI-55821-09
Aadhaar: 4821 7733 9012   PAN: ABCPM1234K
Mobile: +91 98765 43210   Email: rohan.mehta@gmail.com
Bank A/c No: 50100234567812
Bill Date: 14/03/2026    DOB: 05 Aug 1983
Consultant: Dr. Kavita Sharma
Diagnosis: Dengue fever with thrombocytopenia
Room charges (3 days)   12,000.00
Grand Total   48,825.00"""

SECRETS = ["Rohan Mehta", "SMH-2024-88123", "HLI-55821-09", "4821 7733 9012", "ABCPM1234K", "98765 43210",
           "rohan.mehta@gmail.com", "50100234567812", "14/03/2026", "05 Aug 1983", "Kavita Sharma", "Dengue fever"]


def test_every_identifier_is_masked():
    m = mask(DOC)
    for secret in SECRETS:
        assert secret not in m.masked_text, f"leaked: {secret}"
    for t in ["IN_AADHAAR", "IN_PAN", "IN_MOBILE", "EMAIL_ADDRESS", "BANK_ACCOUNT", "POLICY_NUMBER", "PATIENT_ID",
              "PERSON", "DATE_TIME", "MEDICAL_CONDITION"]:
        assert m.counts.get(t), f"no {t} detected"


def test_amounts_hospital_and_labels_are_kept():
    m = mask(DOC)
    for keep in ["Grand Total   48,825.00", "12,000.00", "Sunrise Multispeciality Hospital", "INPATIENT FINAL BILL",
                 "(3 days)", "Age/Sex: 42/M"]:
        assert keep in m.masked_text, f"over-masked: {keep}"


def test_restore_round_trip_and_consistent_placeholders():
    m = mask(DOC + "\nPatient Name: Rohan Mehta")
    assert m.masked_text.count("<PERSON_1>") == 2  # same value -> same placeholder
    assert restore("Patient <PERSON_1>", m.vault) == "Patient Rohan Mehta"
    assert restore("<PAN_99>", m.vault) == "<PAN_99>"  # unknown placeholders are left visible
    assert has_placeholder("<PERSON_1>") and not has_placeholder("Rohan")


def test_lab_test_names_are_not_people():
    m = mask("Test Name: Complete Blood Count    Sample Type: Blood")
    assert "Complete Blood Count" in m.masked_text
