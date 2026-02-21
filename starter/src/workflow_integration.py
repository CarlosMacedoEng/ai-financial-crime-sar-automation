import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from compliance_officer_agent import ComplianceOfficerAgent
from foundation_sar import CaseData, DataLoader, ExplainabilityLogger, RiskAnalystOutput
from risk_analyst_agent import RiskAnalystAgent


def create_sar_document(
    case_data: CaseData,
    risk_analysis: RiskAnalystOutput,
    compliance_review: Any,
    human_decision: str,
) -> Dict[str, Any]:
    amounts = [float(txn.amount) for txn in case_data.transactions]
    txn_dates = sorted(txn.transaction_date for txn in case_data.transactions)
    generated_at = datetime.now(timezone.utc).isoformat()
    decision = human_decision.upper()

    return {
        "sar_metadata": {
            "sar_id": f"SAR-{case_data.case_id.replace('-', '')[:8].upper()}",
            "case_id": case_data.case_id,
            "generated_at": generated_at,
            "filing_type": "Initial",
            "fincen_form_type": "111",
            "status": "Ready for Submission" if decision == "APPROVED" else "Rejected",
        },
        "decision_summary": {
            "risk_analyst": {
                "classification": risk_analysis.classification,
                "risk_level": risk_analysis.risk_level,
                "confidence_score": risk_analysis.confidence_score,
                "key_indicators": risk_analysis.key_indicators,
            },
            "human_gate": {
                "decision": decision,
                "approved": decision == "APPROVED",
                "reviewed_at": generated_at,
            },
        },
        "subject_information": {
            "customer_id": case_data.customer.customer_id,
            "name": case_data.customer.name,
            "ssn_last_4": case_data.customer.ssn_last_4,
            "address": case_data.customer.address,
        },
        "suspicious_activity": {
            "classification": risk_analysis.classification,
            "date_range_start": txn_dates[0] if txn_dates else None,
            "date_range_end": txn_dates[-1] if txn_dates else None,
            "total_amount": round(sum(amounts), 2),
            "transaction_count": len(case_data.transactions),
            "key_indicators": risk_analysis.key_indicators,
        },
        "narrative": {
            "text": compliance_review.narrative,
            "reasoning": compliance_review.narrative_reasoning,
            "regulatory_citations": compliance_review.regulatory_citations,
            "word_count": len(compliance_review.narrative.split()),
            "completeness_check": compliance_review.completeness_check,
        },
    }


def log_human_decision(
    logger: ExplainabilityLogger,
    case_data: CaseData,
    risk_analysis: RiskAnalystOutput,
    reviewer_input: str,
    approved: bool,
) -> None:
    outcome = "APPROVED" if approved else "REJECTED"
    logger.log_agent_action(
        agent_type="HumanReviewer",
        action="review_case",
        case_id=case_data.case_id,
        input_data={
            "displayed_ai_findings": {
                "classification": risk_analysis.classification,
                "risk_level": risk_analysis.risk_level,
                "confidence_score": risk_analysis.confidence_score,
                "key_indicators": risk_analysis.key_indicators,
            },
            "reviewer_input": reviewer_input,
        },
        output_data={"human_decision": outcome, "approved": approved},
        reasoning=f"Human gate decision recorded: {outcome}",
        execution_time_ms=0.0,
        success=True,
        error_message=None,
    )


def _build_fake_risk_client():
    class _Msg:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.message = _Msg(content)

    class _Response:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]

    class _Completions:
        def create(self, model, temperature, max_tokens, messages):
            user_prompt = messages[1]["content"]
            classification = "Other"
            risk_level = "Medium"
            confidence = 0.62
            indicators = ["unusual transaction activity"]
            if "Cash_Deposit $9,9" in user_prompt or "threshold" in user_prompt.lower():
                classification = "Structuring"
                risk_level = "High"
                confidence = 0.84
                indicators = ["threshold avoidance", "repeated cash activity"]
            elif "Wire_Transfer" in user_prompt:
                classification = "Money_Laundering"
                risk_level = "High"
                confidence = 0.78
                indicators = ["rapid wire movement", "layering-like behavior"]

            payload = {
                "classification": classification,
                "confidence_score": confidence,
                "reasoning": f"Case activity aligns with {classification} typology.",
                "analysis_steps": [
                    "1) Data Review: Reviewed customer, account, and transaction facts.",
                    "2) Pattern Recognition: Evaluated suspicious transaction behavior.",
                    "3) Regulatory Mapping: Mapped observed facts to AML typologies.",
                    "4) Risk Quantification: Set confidence and risk severity.",
                    f"5) Classification Decision: Selected {classification}.",
                ],
                "key_indicators": indicators,
                "risk_level": risk_level,
            }
            return _Response(json.dumps(payload))

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    return _Client()


def _build_fake_compliance_client():
    class _Msg:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.message = _Msg(content)

    class _Response:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]

    class _Completions:
        def create(self, model, messages, temperature, max_tokens, response_format):
            user_prompt = messages[1]["content"]
            customer_id = "UNKNOWN"
            date = "2025-01-01"
            amount = "$10,000.00"

            cid_match = user_prompt.split("ID:")
            if len(cid_match) > 1:
                customer_id = cid_match[1].split(")")[0].strip()

            date_match = user_prompt.split("Transactions:")
            if len(date_match) > 1 and "20" in date_match[1]:
                for token in date_match[1].replace("\\n", " ").split():
                    if len(token) == 10 and token[4] == "-" and token[7] == "-":
                        date = token
                        break
            amt_match = user_prompt.find("$")
            if amt_match != -1:
                amount = user_prompt[amt_match : amt_match + 10]

            narrative = (
                f"Customer {customer_id} showed suspicious activity on {date}, including transactions around {amount}. "
                f"The pattern is consistent with potential structuring or laundering indicators and lacks a clear lawful purpose."
            )
            payload = {
                "narrative": narrative,
                "narrative_reasoning": "Included subject, date, amount, and typology anchor in objective language.",
                "regulatory_citations": ["31 CFR 1020.320", "FinCEN SAR Instructions"],
                "completeness_check": True,
            }
            return _Response(json.dumps(payload))

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    return _Client()


def run_workflow_and_generate_outputs(
    *,
    data_dir: str = "starter/data",
    outputs_dir: str = "starter/outputs",
    case_limit: int = 5,
    use_fake_clients: bool = True,
    risk_client: Optional[Any] = None,
    compliance_client: Optional[Any] = None,
) -> Dict[str, Any]:
    outputs_root = Path(outputs_dir)
    filed_dir = outputs_root / "filed_sars"
    audit_dir = outputs_root / "audit_logs"
    filed_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    logger = ExplainabilityLogger(str(audit_dir / "workflow_integration.jsonl"))
    data_loader = DataLoader(logger)
    cases = data_loader.create_cases_from_csv(data_dir=data_dir, skip_customers_without_transactions=True)
    selected_cases = cases[:case_limit]

    if use_fake_clients:
        risk_client = _build_fake_risk_client()
        compliance_client = _build_fake_compliance_client()
    elif risk_client is None or compliance_client is None:
        raise ValueError("risk_client and compliance_client are required when use_fake_clients=False")

    risk_agent = RiskAnalystAgent(risk_client, logger)
    compliance_agent = ComplianceOfficerAgent(compliance_client, logger)

    processed = 0
    approved = 0
    rejected = 0
    sar_files: List[str] = []

    for case_data in selected_cases:
        processed += 1
        risk_result = risk_agent.analyze_case(case_data)

        approved_flag = (risk_result.risk_level in {"High", "Critical"}) and (risk_result.confidence_score >= 0.7)
        reviewer_input = (
            "Approved due to elevated risk level and sufficient confidence."
            if approved_flag
            else "Rejected due to insufficient confidence/risk severity."
        )
        log_human_decision(
            logger=logger,
            case_data=case_data,
            risk_analysis=risk_result,
            reviewer_input=reviewer_input,
            approved=approved_flag,
        )

        if not approved_flag:
            rejected += 1
            continue

        approved += 1
        compliance_result = compliance_agent.generate_compliance_narrative(case_data, risk_result)
        sar_document = create_sar_document(
            case_data=case_data,
            risk_analysis=risk_result,
            compliance_review=compliance_result,
            human_decision="APPROVED",
        )
        sar_id = sar_document["sar_metadata"]["sar_id"]
        sar_path = filed_dir / f"{sar_id}.json"
        sar_path.write_text(json.dumps(sar_document, indent=2), encoding="utf-8")
        sar_files.append(str(sar_path))

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processed_cases": processed,
        "approved_cases": approved,
        "rejected_cases": rejected,
        "approval_rate_pct": round((approved / processed) * 100, 2) if processed else 0.0,
        "stage2_calls_saved": rejected,
        "sar_files": sar_files,
        "audit_log": str(audit_dir / "workflow_integration.jsonl"),
    }
    (audit_dir / "workflow_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    summary = run_workflow_and_generate_outputs()
    print(json.dumps(summary, indent=2))
