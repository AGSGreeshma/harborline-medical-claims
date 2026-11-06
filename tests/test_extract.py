from harborline.extract_rules import classify, extract, find_total

BILL = """Lotus Care Hospital
INPATIENT FINAL BILL
Patient Name: <PERSON_1>    Age/Sex: 74/F
Bill Date: <DATE_TIME_1>    Consultant: <PERSON_2>
Diagnosis: <MEDICAL_CONDITION_1>
Room charges    14,982.86
Sub Total    49,942.86
GST 5%    2,497.14
Grand Total    52,440.00
Advance paid    10,000.00
Balance payable    42,440.00
Amount in words: Rupees Fifty Two Thousand Four Hundred Forty Only"""


def test_grand_total_beats_subtotal_tax_and_balance():
    assert find_total(BILL) == (52440.0, "Grand Total")


def test_net_payable_used_when_no_grand_total():
    assert find_total("Sub Total 46,000.00\nCGST 1,150.00\nNet Payable 48,300.00")[0] == 48300.0


def test_subtotal_alone_is_never_the_total():
    assert find_total("Sub Total 46,000.00\nGST 2,300.00")[0] is None


def test_classify():
    assert classify(BILL) == "Hospital Bill"
    assert classify("PRESCRIPTION\nRx\nTab. Paracetamol 650 mg 1-0-1") == "Prescription"
    assert classify("LABORATORY TEST REPORT\nInvestigation Result Units Reference Range") == "Lab Report"


def test_extract_bill_fields_and_label_boundaries():
    o = extract(BILL)
    assert o.document_type == "Hospital Bill"
    assert o.patient_name == "<PERSON_1>"           # stops before "Age/Sex"
    assert o.bill_date == "<DATE_TIME_1>"           # stops before "Consultant"
    assert o.total_amount == 52440.0
    assert o.total_in_words.startswith("Rupees Fifty Two Thousand")
    assert o.hospital_name == "Lotus Care Hospital"


def test_lab_report_test_name_stops_at_next_label():
    o = extract("CityCare Pathology Lab\nLABORATORY TEST REPORT\nTest Name: Lipid Profile Sample Type: Blood\nReference Range")
    assert o.diagnosis_or_test == "Lipid Profile"
    assert o.total_amount is None
