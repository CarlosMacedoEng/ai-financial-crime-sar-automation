import json
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compliance_officer_agent import ComplianceOfficerAgent
from foundation_sar import (
    AccountData,
    CaseData,
    CustomerData,
    ExplainabilityLogger,
    RiskAnalystOutput,
    TransactionData,
)
from risk_analyst_agent import RiskAnalystAgent


def _build_case(case_id: str = "CASE_TEST") -> CaseData:
    customer = CustomerData(
        customer_id=f"{case_id}_CUST",
        name="Test Customer",
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
        current_balance=15000.0,
        average_monthly_balance=12000.0,
        status="Active",
    )
    txn = TransactionData(
        transaction_id=f"{case_id}_TXN",
        account_id=f"{case_id}_ACC",
        transaction_date="2025-01-01",
        transaction_type="Cash_Deposit",
        amount=9900.0,
        description="Cash deposit",
        method="Cash",
    )
    return CaseData(
        case_id=case_id,
        customer=customer,
        accounts=[account],
        transactions=[txn],
        case_created_at=datetime.now().isoformat(),
        data_sources={"source": "unit_test"},
    )


def _risk_analysis() -> RiskAnalystOutput:
    return RiskAnalystOutput(
        classification="Structuring",
        confidence_score=0.82,
        reasoning="Pattern matches structuring behavior.",
        analysis_steps=[
            "1) Data Review: checked case facts.",
            "2) Pattern Recognition: near-threshold deposits.",
            "3) Regulatory Mapping: mapped to structuring red flags.",
            "4) Risk Quantification: high risk with strong signal.",
            "5) Classification Decision: Structuring.",
        ],
        key_indicators=["threshold avoidance"],
        risk_level="High",
    )


def _response_with_content(content: str):
    response = Mock()
    choice = Mock()
    message = Mock()
    message.content = content
    choice.message = message
    response.choices = [choice]
    return response


def test_risk_agent_logs_and_raises_on_api_failure(tmp_path):
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("network timeout")
    logger = ExplainabilityLogger(str(tmp_path / "test_risk_api_failure.jsonl"))
    agent = RiskAnalystAgent(client, logger)

    with pytest.raises(RuntimeError, match="RiskAnalyst API call failed"):
        agent.analyze_case(_build_case("CASE_API_FAIL"))

    assert len(logger.entries) == 1
    assert logger.entries[0]["agent_type"] == "RiskAnalyst"
    assert logger.entries[0]["success"] is False
    assert "network timeout" in (logger.entries[0]["error_message"] or "")


def test_risk_agent_repair_retry_recovers_invalid_json(tmp_path):
    client = Mock()
    invalid_response = _response_with_content("not valid json")
    repaired_payload = {
        "classification": "Structuring",
        "confidence_score": 0.83,
        "reasoning": "Repaired output.",
        "analysis_steps": [
            "1) Data Review: checked facts.",
            "2) Pattern Recognition: near-threshold behavior.",
            "3) Regulatory Mapping: matched structuring indicators.",
            "4) Risk Quantification: high confidence and risk.",
            "5) Classification Decision: Structuring selected.",
        ],
        "key_indicators": ["threshold avoidance"],
        "risk_level": "High",
    }
    repaired_response = _response_with_content(json.dumps(repaired_payload))
    client.chat.completions.create.side_effect = [invalid_response, repaired_response]

    logger = ExplainabilityLogger(str(tmp_path / "test_risk_retry.jsonl"))
    agent = RiskAnalystAgent(client, logger)
    result = agent.analyze_case(_build_case("CASE_REPAIR"))

    assert result.classification == "Structuring"
    assert len(result.analysis_steps) == 5
    assert len(logger.entries) == 1
    assert logger.entries[0]["success"] is True
    assert "recovered_via_repair_retry" in (logger.entries[0]["reasoning"] or "")


def test_compliance_agent_logs_and_raises_on_api_failure(tmp_path):
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("auth failed")
    logger = ExplainabilityLogger(str(tmp_path / "test_compliance_api_failure.jsonl"))
    agent = ComplianceOfficerAgent(client, logger)

    with pytest.raises(RuntimeError, match="ComplianceOfficer API call failed"):
        agent.generate_compliance_narrative(_build_case("CASE_COMP_API_FAIL"), _risk_analysis())

    assert len(logger.entries) == 1
    assert logger.entries[0]["agent_type"] == "ComplianceOfficer"
    assert logger.entries[0]["success"] is False
    assert "auth failed" in (logger.entries[0]["error_message"] or "")


def test_compliance_agent_repair_retry_recovers_invalid_json(tmp_path):
    client = Mock()
    invalid_response = _response_with_content("malformed")
    repaired_payload = {
        "narrative": "Test Customer (CASE_COMP_REPAIR_CUST) made a $9,900.00 cash deposit on 2025-01-01 with suspicious structuring indicators requiring SAR review.",
        "narrative_reasoning": "Repaired JSON and preserved key compliance facts.",
        "regulatory_citations": ["31 CFR 1020.320"],
        "completeness_check": True,
    }
    repaired_response = _response_with_content(json.dumps(repaired_payload))
    client.chat.completions.create.side_effect = [invalid_response, repaired_response]

    logger = ExplainabilityLogger(str(tmp_path / "test_compliance_retry.jsonl"))
    agent = ComplianceOfficerAgent(client, logger)
    output = agent.generate_compliance_narrative(_build_case("CASE_COMP_REPAIR"), _risk_analysis())

    assert output.completeness_check is True
    assert "suspicious" in output.narrative.lower()
    assert len(logger.entries) == 1
    assert logger.entries[0]["success"] is True
    assert "recovered_via_repair_retry" in (logger.entries[0]["reasoning"] or "")


def test_compliance_agent_content_repair_adds_missing_citation(tmp_path):
    client = Mock()
    initial_payload = {
        "narrative": "Test Customer (CASE_COMP_REPAIR_CUST) made a $9,900.00 cash deposit on 2025-01-01 with suspicious structuring indicators requiring SAR review.",
        "narrative_reasoning": "Initial output missing citation.",
        "regulatory_citations": [],
        "completeness_check": False,
    }
    repaired_payload = {
        "narrative": "Test Customer (CASE_COMP_REPAIR_CUST) made a $9,900.00 cash deposit on 2025-01-01 with suspicious structuring indicators requiring SAR review.",
        "narrative_reasoning": "Added mandatory citation.",
        "regulatory_citations": ["31 CFR 1020.320"],
        "completeness_check": True,
    }
    client.chat.completions.create.side_effect = [
        _response_with_content(json.dumps(initial_payload)),
        _response_with_content(json.dumps(repaired_payload)),
    ]

    logger = ExplainabilityLogger(str(tmp_path / "test_compliance_content_repair.jsonl"))
    agent = ComplianceOfficerAgent(client, logger)
    output = agent.generate_compliance_narrative(_build_case("CASE_COMP_REPAIR"), _risk_analysis())

    assert output.completeness_check is True
    assert output.regulatory_citations


def test_compliance_agent_rejects_speculative_tone_after_retry(tmp_path):
    client = Mock()
    speculative_payload = {
        "narrative": "I think Test Customer (CASE_COMP_REPAIR_CUST) maybe made a $9,900.00 cash deposit on 2025-01-01 and could be suspicious.",
        "narrative_reasoning": "Speculative wording.",
        "regulatory_citations": ["31 CFR 1020.320"],
        "completeness_check": True,
    }
    client.chat.completions.create.side_effect = [
        _response_with_content(json.dumps(speculative_payload)),
        _response_with_content(json.dumps(speculative_payload)),
    ]

    logger = ExplainabilityLogger(str(tmp_path / "test_compliance_speculative_fail.jsonl"))
    agent = ComplianceOfficerAgent(client, logger)
    with pytest.raises(ValueError, match="Compliance validation failed"):
        agent.generate_compliance_narrative(_build_case("CASE_COMP_REPAIR"), _risk_analysis())
