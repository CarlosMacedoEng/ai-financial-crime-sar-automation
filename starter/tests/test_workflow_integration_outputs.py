import json
import sys
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from foundation_sar import ExplainabilityLogger
from workflow_integration import create_sar_document, log_human_decision, run_workflow_and_generate_outputs


def _load_first_case():
    from foundation_sar import DataLoader

    temp_log = Path(tempfile.gettempdir()) / "test_workflow_loader.jsonl"
    logger = ExplainabilityLogger(str(temp_log))
    loader = DataLoader(logger)
    cases = loader.create_cases_from_csv(data_dir="starter/data", skip_customers_without_transactions=True)
    return cases[0]


def _risk_result_for(case_data):
    from foundation_sar import RiskAnalystOutput

    return RiskAnalystOutput(
        classification="Structuring",
        confidence_score=0.85,
        reasoning="Pattern aligns with near-threshold behavior.",
        analysis_steps=[
            "1) Data Review: Checked case data.",
            "2) Pattern Recognition: Found suspicious pattern.",
            "3) Regulatory Mapping: Matched AML typology.",
            "4) Risk Quantification: High confidence.",
            "5) Classification Decision: Structuring.",
        ],
        key_indicators=["threshold avoidance"],
        risk_level="High",
    )


def _compliance_result_for(case_data):
    from foundation_sar import ComplianceOfficerOutput

    first_date = case_data.transactions[0].transaction_date
    return ComplianceOfficerOutput(
        narrative=(
            f"Customer {case_data.customer.customer_id} showed suspicious activity on {first_date} "
            f"including transactions around $9,900.00 consistent with structuring indicators."
        ),
        narrative_reasoning="Objective narrative with required anchors.",
        regulatory_citations=["31 CFR 1020.320"],
        completeness_check=True,
    )


def test_create_sar_document_includes_required_metadata():
    case_data = _load_first_case()
    risk = _risk_result_for(case_data)
    compliance = _compliance_result_for(case_data)

    sar = create_sar_document(case_data, risk, compliance, human_decision="APPROVED")

    assert sar["sar_metadata"]["case_id"] == case_data.case_id
    assert "generated_at" in sar["sar_metadata"]
    assert sar["decision_summary"]["risk_analyst"]["classification"] == "Structuring"
    assert sar["decision_summary"]["human_gate"]["decision"] == "APPROVED"


def test_log_human_decision_writes_structured_event(tmp_path):
    case_data = _load_first_case()
    risk = _risk_result_for(case_data)

    logger = ExplainabilityLogger(str(tmp_path / "workflow_integration.jsonl"))
    log_human_decision(
        logger=logger,
        case_data=case_data,
        risk_analysis=risk,
        reviewer_input="Approved by reviewer",
        approved=True,
    )

    assert logger.entries
    entry = logger.entries[-1]
    assert entry["agent_type"] == "HumanReviewer"
    assert entry["action"] == "review_case"
    assert entry["success"] is True


def test_workflow_outputs_are_organized_and_named(tmp_path):
    outputs_dir = tmp_path / "outputs"
    metrics = run_workflow_and_generate_outputs(
        data_dir="starter/data",
        outputs_dir=str(outputs_dir),
        case_limit=3,
        use_fake_clients=True,
    )

    filed_dir = outputs_dir / "filed_sars"
    audit_dir = outputs_dir / "audit_logs"
    audit_log = audit_dir / "workflow_integration.jsonl"
    metrics_file = audit_dir / "workflow_metrics.json"

    assert filed_dir.exists()
    assert audit_dir.exists()
    assert audit_log.exists()
    assert metrics_file.exists()
    assert metrics["processed_cases"] == 3

    sar_files = sorted(filed_dir.glob("SAR-*.json"))
    assert sar_files, "Expected at least one filed SAR with SAR-<ID>.json naming"

    sample = json.loads(sar_files[0].read_text(encoding="utf-8"))
    assert "decision_summary" in sample
    assert "case_id" in sample["sar_metadata"]
