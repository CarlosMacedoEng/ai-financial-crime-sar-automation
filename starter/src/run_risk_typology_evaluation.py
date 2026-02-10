import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

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
        name="Evaluation Subject",
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
        data_sources={"source": "typology_eval"},
    )


def _payload(label: str, confidence: float, risk_level: str) -> str:
    return json.dumps(
        {
            "classification": label,
            "confidence_score": confidence,
            "reasoning": f"Signals align with {label} typology.",
            "analysis_steps": [
                "1) Data Review: Reviewed customer, account, and transactions.",
                "2) Pattern Recognition: Flagged suspicious behavioral pattern.",
                "3) Regulatory Mapping: Mapped pattern to AML typology guidance.",
                "4) Risk Quantification: Assigned confidence and risk severity.",
                f"5) Classification Decision: Selected {label}.",
            ],
            "key_indicators": [f"{label.lower()} indicator"],
            "risk_level": risk_level,
        }
    )


def run_typology_evaluation() -> dict:
    cases = [
        ("CASE_STRUCTURING", "Structuring", _build_case("CASE_STRUCTURING", "Cash_Deposit", 9900.0, "Repeated near-threshold cash deposits")),
        ("CASE_SANCTIONS", "Sanctions", _build_case("CASE_SANCTIONS", "Wire_Transfer_Debit", 55000.0, "Transfer to restricted entity", "SANCTIONED ENTITY")),
        ("CASE_FRAUD", "Fraud", _build_case("CASE_FRAUD", "ACH_Debit", 7200.0, "Unusual online debit with profile mismatch")),
        ("CASE_ML", "Money_Laundering", _build_case("CASE_ML", "Wire_Transfer", 78000.0, "Layered transfer through offshore channels", "OFFSHORE HOLDINGS")),
        ("CASE_OTHER", "Other", _build_case("CASE_OTHER", "Online_Transfer", 1800.0, "Irregular activity without clear core typology")),
    ]

    response_map = {
        "CASE_STRUCTURING": _payload("Structuring", 0.87, "High"),
        "CASE_SANCTIONS": _payload("Sanctions", 0.91, "Critical"),
        "CASE_FRAUD": _payload("Fraud", 0.79, "High"),
        "CASE_ML": _payload("Money_Laundering", 0.88, "Critical"),
        "CASE_OTHER": _payload("Other", 0.62, "Medium"),
    }

    client = Mock()

    def _create_completion(model, temperature, max_tokens, messages):
        prompt = messages[1]["content"]
        for cid, payload in response_map.items():
            if cid in prompt:
                return _FakeResponse(payload)
        return _FakeResponse(response_map["CASE_OTHER"])

    client.chat.completions.create.side_effect = _create_completion

    logger = ExplainabilityLogger("starter/outputs/typology_eval_audit.jsonl")
    agent = RiskAnalystAgent(client, logger, model="gpt-4")

    results = []
    for case_id, expected, case_data in cases:
        output = agent.analyze_case(case_data)
        results.append(
            {
                "case_id": case_id,
                "expected_classification": expected,
                "actual_classification": output.classification,
                "match": output.classification == expected,
                "confidence_score": output.confidence_score,
                "risk_level": output.risk_level,
                "analysis_steps_count": len(output.analysis_steps),
            }
        )

    pass_count = sum(1 for r in results if r["match"])
    return {
        "generated_at": datetime.now().isoformat(),
        "total_cases": len(results),
        "passed": pass_count,
        "pass_rate": pass_count / len(results) if results else 0.0,
        "results": results,
    }


if __name__ == "__main__":
    report = run_typology_evaluation()
    out_path = Path("starter/outputs/risk_typology_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
