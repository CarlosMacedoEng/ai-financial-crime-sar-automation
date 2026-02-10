import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from foundation_sar import (
    AccountData,
    CaseData,
    CustomerData,
    ExplainabilityLogger,
    TransactionData,
)
from risk_analyst_agent import RiskAnalystAgent


class _FakeChoiceMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeChoiceMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def _build_case(case_id: str, transaction_type: str, amount: float, description: str, counterparty: str = None) -> CaseData:
    customer = CustomerData(
        customer_id=f"{case_id}_CUST",
        name="Test Subject",
        date_of_birth="1980-01-01",
        ssn_last_4="1234",
        address="123 Test St",
        customer_since="2020-01-01",
        risk_rating="Medium",
    )
    account = AccountData(
        account_id=f"{case_id}_ACC",
        customer_id=f"{case_id}_CUST",
        account_type="Checking",
        opening_date="2020-01-01",
        current_balance=50000.0,
        average_monthly_balance=35000.0,
        status="Active",
    )
    txn = TransactionData(
        transaction_id=f"{case_id}_TXN_1",
        account_id=f"{case_id}_ACC",
        transaction_date="2025-01-01",
        transaction_type=transaction_type,
        amount=amount,
        description=description,
        method="Wire" if "Wire" in transaction_type else "Cash",
        counterparty=counterparty,
        location="ONLINE",
    )
    return CaseData(
        case_id=case_id,
        customer=customer,
        accounts=[account],
        transactions=[txn],
        case_created_at=datetime.now().isoformat(),
        data_sources={"source": "typology_test"},
    )


def _fake_completion_payload(label: str, confidence: float, risk_level: str) -> str:
    payload = {
        "classification": label,
        "confidence_score": confidence,
        "reasoning": f"Signals align with {label} typology.",
        "analysis_steps": [
            "1) Data Review: Reviewed customer, account, and transaction context.",
            "2) Pattern Recognition: Identified red flags and behavioral anomalies.",
            "3) Regulatory Mapping: Mapped pattern to AML typology definitions.",
            "4) Risk Quantification: Calibrated confidence and risk level.",
            f"5) Classification Decision: Selected {label} as best fit.",
        ],
        "key_indicators": [f"{label.lower()} indicator"],
        "risk_level": risk_level,
    }
    return json.dumps(payload)


def test_typology_coverage_five_categories(tmp_path):
    cases = [
        ("CASE_STRUCTURING", "Structuring", _build_case("CASE_STRUCTURING", "Cash_Deposit", 9900.0, "Repeated near-threshold cash deposit")),
        ("CASE_SANCTIONS", "Sanctions", _build_case("CASE_SANCTIONS", "Wire_Transfer_Debit", 55000.0, "Wire transfer to restricted entity", "SANCTIONED ENTITY")),
        ("CASE_FRAUD", "Fraud", _build_case("CASE_FRAUD", "ACH_Debit", 7200.0, "Unusual online debit from unfamiliar device")),
        ("CASE_ML", "Money_Laundering", _build_case("CASE_ML", "Wire_Transfer", 78000.0, "Layered transfer through offshore counterparties", "OFFSHORE HOLDINGS")),
        ("CASE_OTHER", "Other", _build_case("CASE_OTHER", "Online_Transfer", 1800.0, "Irregular activity not fitting standard typologies")),
    ]

    response_map = {
        "CASE_STRUCTURING": _fake_completion_payload("Structuring", 0.87, "High"),
        "CASE_SANCTIONS": _fake_completion_payload("Sanctions", 0.91, "Critical"),
        "CASE_FRAUD": _fake_completion_payload("Fraud", 0.79, "High"),
        "CASE_ML": _fake_completion_payload("Money_Laundering", 0.88, "Critical"),
        "CASE_OTHER": _fake_completion_payload("Other", 0.62, "Medium"),
    }

    mock_client = Mock()

    def _create_completion(model, temperature, max_tokens, messages):
        prompt = messages[1]["content"]
        case_id = "UNKNOWN"
        for cid in response_map.keys():
            if cid in prompt:
                case_id = cid
                break
        return _FakeResponse(response_map[case_id])

    mock_client.chat.completions.create.side_effect = _create_completion
    logger = ExplainabilityLogger(str(tmp_path / "typology_coverage_test_audit.jsonl"))
    agent = RiskAnalystAgent(mock_client, logger, model="gpt-4")

    for case_id, expected_label, case_data in cases:
        result = agent.analyze_case(case_data)
        assert result.classification == expected_label
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.risk_level in {"Low", "Medium", "High", "Critical"}
        assert len(result.analysis_steps) == 5
