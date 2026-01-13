# Risk Analyst Agent - Chain-of-Thought Implementation

"""
Risk Analyst Agent Module

This agent performs suspicious activity classification using Chain-of-Thought reasoning.
It analyzes customer profiles, account behavior, and transaction patterns to identify
potential financial crimes.

YOUR TASKS:
- Study Chain-of-Thought prompting methodology
- Design system prompt with structured reasoning framework
- Implement case analysis with proper error handling
- Parse and validate structured JSON responses
- Create comprehensive audit logging
"""

import json
import openai
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# TODO: Import your foundation components
from foundation_sar import (
    CustomerData,
    AccountData,
    TransactionData,
    CaseData,
    ExplainabilityLogger,
    RiskAnalystOutput,
    ExplainabilityLogger,
    CaseData
)

# Load environment variables
load_dotenv()

class RiskAnalystAgent:
    """
    Risk Analyst agent using Chain-of-Thought reasoning.
    
    TODO: Implement agent that:
    - Uses systematic Chain-of-Thought prompting
    - Classifies suspicious activity patterns
    - Returns structured JSON output
    - Handles errors gracefully
    - Logs all operations for audit
    """
    
    def __init__(self, openai_client, explainability_logger, model="gpt-4"):
        """Initialize the Risk Analyst Agent
        
        Args:
            openai_client: OpenAI client instance
            explainability_logger: Logger for audit trails
            model: OpenAI model to use
        """
        self.client = openai_client
        self.logger = explainability_logger
        self.model = model

        self.system_prompt = (
            "You are a Risk Analyst specializing in Financial Crime detection and SAR triage.\n"
            "Use Chain-of-Thought, step-by-step reasoning internally to analyze the case.\n\n"
            "Goal:\n"
            "- Classify suspicious activity and provide a concise professional rationale.\n\n"
            "Classification categories (use exactly one):\n"
            "- Structuring: Transactions designed to avoid reporting thresholds.\n"
            "- Sanctions: Potential sanctions violations or prohibited parties.\n"
            "- Fraud: Fraudulent transactions, identity misuse, or account takeover patterns.\n"
            "- Money_Laundering: Layering/integration typologies obscuring illicit source of funds.\n"
            "- Other: Suspicious patterns not fitting standard categories.\n\n"
            "Reasoning Framework (think step-by-step):\n"
            "1) Data Review: Summarize key facts from customer/accounts/transactions.\n"
            "2) Pattern Recognition: Identify typologies and anomalies.\n"
            "3) Regulatory Mapping: Map facts to common AML/CTF red flags.\n"
            "4) Risk Quantification: Determine severity and confidence.\n"
            "5) Classification Decision: Select the best category and justify.\n\n"
            "Output MUST be valid JSON only (no extra text). Schema:\n"
            "{\n"
            '  "classification": "Structuring|Sanctions|Fraud|Money_Laundering|Other",\n'
            '  "confidence_score": 0.0-1.0,\n'
            '  "reasoning": "string (concise, professional)",\n'
            '  "key_indicators": ["string", "string", ...],\n'
            '  "risk_level": "Low|Medium|High|Critical"\n'
            "}\n\n"
            "Constraints:\n"
            "- Return JSON only.\n"
            "- Do not include markdown.\n"
            "- Ensure confidence_score is a number between 0 and 1.\n"
            "- key_indicators must be a JSON array of strings.\n"
        )

    def analyze_case(self, case_data: Any) -> RiskAnalystOutput:
        """
        Analyze a case and return a validated RiskAnalystOutput.

        Test-driven requirements:
        - Must call client.chat.completions.create with:
          model, temperature=0.3, max_tokens=1000, messages=[system,user] (exactly 2 messages)
        - Must parse JSON robustly
        - Must raise:
          * ValueError("No JSON content found") when response has no JSON
          * ValueError("Failed to parse Risk Analyst JSON output") on invalid JSON/fields
        - Must log to logger.entries with agent_type="RiskAnalyst" and success True/False
        """
        started_at = datetime.now(timezone.utc)

        if case_data is None:
            raise ValueError("case_data is required")

        case_id = getattr(case_data, "case_id", None) or "UNKNOWN"

        user_prompt = self._format_case_for_prompt(case_data)

        # OpenAI call (tests assert these parameters)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Defensive extraction of response content
        try:
            response_content = response.choices[0].message.content
        except Exception as e:
            finished_at = datetime.now(timezone.utc)
            self._log_action(
                agent_type="RiskAnalyst",
                action="analyze_case",
                case_id=case_id,
                input_data={"case_id": case_id},
                output_data={},
                reasoning=f"Model response missing expected structure: {e}",
                started_at=started_at,
                finished_at=finished_at,
                success=False,
                error_message=str(e),
            )
            raise

        # Parse JSON
        try:
            json_str = self._extract_json_from_response(response_content)
            parsed = json.loads(json_str)
        except Exception as e:
            finished_at = datetime.now(timezone.utc)
            self._log_action(
                agent_type="RiskAnalyst",
                action="analyze_case",
                case_id=case_id,
                input_data={"case_id": case_id},
                output_data={},
                reasoning=f"JSON parsing failed: {e}",
                started_at=started_at,
                finished_at=finished_at,
                success=False,
                error_message=str(e),
            )
            # IMPORTANT: tests expect this exact message for any parsing failure
            raise ValueError("Failed to parse Risk Analyst JSON output")

        # Validate required fields (unit tests rely on failure path)
        required = ["classification", "confidence_score", "reasoning", "key_indicators", "risk_level"]
        if not isinstance(parsed, dict) or any(k not in parsed for k in required):
            finished_at = datetime.now(timezone.utc)
            self._log_action(
                agent_type="RiskAnalyst",
                action="analyze_case",
                case_id=case_id,
                input_data={"case_id": case_id},
                output_data=parsed if isinstance(parsed, dict) else {},
                reasoning="JSON parsing failed: Missing required fields in output",
                started_at=started_at,
                finished_at=finished_at,
                success=False,
                error_message="Missing required fields",
            )
            raise ValueError("Failed to parse Risk Analyst JSON output")

        # Normalize types
        try:
            classification = str(parsed["classification"])
            risk_level = str(parsed["risk_level"])
            reasoning = str(parsed["reasoning"])
            confidence_score = float(parsed["confidence_score"])
            key_indicators_raw = parsed["key_indicators"]
            if not isinstance(key_indicators_raw, list):
                key_indicators = [str(key_indicators_raw)] if key_indicators_raw is not None else []
            else:
                key_indicators = [str(x) for x in key_indicators_raw]
        except Exception as e:
            finished_at = datetime.now(timezone.utc)
            self._log_action(
                agent_type="RiskAnalyst",
                action="analyze_case",
                case_id=case_id,
                input_data={"case_id": case_id},
                output_data=parsed if isinstance(parsed, dict) else {},
                reasoning=f"JSON parsing failed: {e}",
                started_at=started_at,
                finished_at=finished_at,
                success=False,
                error_message=str(e),
            )
            raise ValueError("Failed to parse Risk Analyst JSON output")

        # Build Pydantic output (foundation_sar schema enforces enums/constraints)
        try:
            result = RiskAnalystOutput(
                classification=classification,
                confidence_score=confidence_score,
                reasoning=reasoning,
                key_indicators=key_indicators,
                risk_level=risk_level,
            )
        except Exception as e:
            finished_at = datetime.now(timezone.utc)
            self._log_action(
                agent_type="RiskAnalyst",
                action="analyze_case",
                case_id=case_id,
                input_data={"case_id": case_id},
                output_data=parsed,
                reasoning=f"JSON parsing failed: Output validation failed: {e}",
                started_at=started_at,
                finished_at=finished_at,
                success=False,
                error_message=str(e),
            )
            raise ValueError("Failed to parse Risk Analyst JSON output")

        # Success log
        finished_at = datetime.now(timezone.utc)
        self._log_action(
            agent_type="RiskAnalyst",
            action="analyze_case",
            case_id=case_id,
            input_data={"case_id": case_id},
            output_data=result.model_dump() if hasattr(result, "model_dump") else dict(result),
            reasoning=reasoning,
            started_at=started_at,
            finished_at=finished_at,
            success=True,
            error_message=None,
        )

        return result

    def _extract_json_from_response(self, response_content: str) -> str:
        """
        Extract JSON content from LLM response.

        Handles:
        - JSON in fenced code blocks (```json ... ```)
        - JSON in plain text
        - Empty/malformed responses

        Tests require:
        - empty/no JSON -> ValueError("No JSON content found")
        """
        if response_content is None or str(response_content).strip() == "":
            raise ValueError("No JSON content found")

        text = str(response_content).strip()

        # Prefer fenced block ```json ... ```
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            # fallback ``` ... ```
            m = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                return candidate

        # Fallback: first {...} span
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = text[first : last + 1].strip()
            if candidate:
                return candidate

        raise ValueError("No JSON content found")

    def _format_case_for_prompt(self, case_data: Any) -> str:
        """
        Format case data (preferably foundation_sar.CaseData) into a user prompt.
        """
        case_id = getattr(case_data, "case_id", "UNKNOWN_CASE")
        customer = getattr(case_data, "customer", None)
        accounts = getattr(case_data, "accounts", []) or []
        transactions = getattr(case_data, "transactions", []) or []
        created_at = getattr(case_data, "case_created_at", None)
        data_sources = getattr(case_data, "data_sources", None)

        customer_lines: List[str] = []
        if customer is not None:
            customer_lines = [
                f"customer_id: {getattr(customer, 'customer_id', '')}",
                f"name: {getattr(customer, 'name', '')}",
                f"date_of_birth: {getattr(customer, 'date_of_birth', '')}",
                f"ssn_last_4: {getattr(customer, 'ssn_last_4', '')}",
                f"address: {getattr(customer, 'address', '')}",
                f"customer_since: {getattr(customer, 'customer_since', '')}",
                f"risk_rating: {getattr(customer, 'risk_rating', '')}",
            ]

        accounts_block = self._format_accounts(accounts) if accounts else "No accounts provided."
        transactions_block = self._format_transactions(transactions) if transactions else "No transactions provided."

        # lightweight summary stats
        amounts: List[float] = []
        for t in transactions:
            try:
                amounts.append(float(getattr(t, "amount", 0.0)))
            except Exception:
                pass
        total_amount = sum(amounts) if amounts else 0.0

        prompt = (
            f"Case ID: {case_id}\n"
            f"Case Created At: {created_at}\n"
            f"Data Sources: {data_sources}\n\n"
            "Customer Profile:\n"
            + ("\n".join(f"- {line}" for line in customer_lines) if customer_lines else "- No customer profile provided.")
            + "\n\nAccounts:\n"
            + accounts_block
            + "\n\nTransactions:\n"
            + transactions_block
            + "\n\nSummary Stats:\n"
            + f"- transaction_count: {len(transactions)}\n"
            + f"- total_transaction_amount: ${total_amount:,.2f}\n\n"
            "Return JSON only, strictly following the schema from the system instructions."
        )
        return prompt

    def _format_accounts(self, accounts: List[Any]) -> str:
        """
        Tests require currency formatting like "$15,000.50".
        """
        def fmt_money(v: Any) -> str:
            try:
                return f"${float(v):,.2f}"
            except Exception:
                return "$0.00"

        lines: List[str] = []
        for idx, acc in enumerate(accounts, start=1):
            lines.append(
                f"{idx}. {getattr(acc, 'account_id', '')} | {getattr(acc, 'account_type', '')} | "
                f"opened: {getattr(acc, 'opening_date', '')} | "
                f"current_balance: {fmt_money(getattr(acc, 'current_balance', 0.0))} | "
                f"avg_monthly_balance: {fmt_money(getattr(acc, 'average_monthly_balance', 0.0))} | "
                f"status: {getattr(acc, 'status', '')}"
            )
        return "\n".join(lines) if lines else "No accounts provided."

    def _format_transactions(self, transactions: List[Any]) -> str:
        """
        Tests require lines like:
        '1. 2025-01-01: Cash_Deposit $9,900.00 - ...'
        """
        lines: List[str] = []
        for idx, txn in enumerate(transactions, start=1):
            date = getattr(txn, "transaction_date", "")
            txn_type = getattr(txn, "transaction_type", "")
            desc = getattr(txn, "description", "")
            method = getattr(txn, "method", "")
            location = getattr(txn, "location", None)
            account_id = getattr(txn, "account_id", "")

            try:
                amount_str = f"${float(getattr(txn, 'amount', 0.0)):,.2f}"
            except Exception:
                amount_str = "$0.00"

            extras: List[str] = []
            if account_id:
                extras.append(f"account_id={account_id}")
            if method:
                extras.append(f"method={method}")
            if location:
                extras.append(f"location={location}")

            extra_txt = (" | " + " | ".join(extras)) if extras else ""
            lines.append(f"{idx}. {date}: {txn_type} {amount_str} - {desc}{extra_txt}")

        return "\n".join(lines) if lines else "No transactions provided."
    
     # ===== Logging bridge to foundation_sar.ExplainabilityLogger =====

    def _log_action(
        self,
        *,
        agent_type: str,
        action: str,
        case_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        reasoning: str,
        started_at: datetime,
        finished_at: datetime,
        success: bool,
        error_message: Optional[str],
    ) -> None:
        """
        Prefer using foundation_sar.ExplainabilityLogger.log_agent_action(), but keep a fallback
        compatible with tests that assert:
          - logger.entries exists (list)
          - entries[0]["agent_type"] == "RiskAnalyst"
          - entries[0]["success"] == True/False

        Your foundation logger stores keys:
          timestamp, case_id, agent_type, action, input_summary, output_summary,
          reasoning, execution_time_ms, success, error_message
        """
        exec_ms = (finished_at - started_at).total_seconds() * 1000.0

        # Preferred path: use the foundation logger API when available.
        if hasattr(self.logger, "log_agent_action") and callable(getattr(self.logger, "log_agent_action")):
            try:
                self.logger.log_agent_action(
                    agent_type=agent_type,
                    action=action,
                    case_id=case_id,
                    input_data=input_data,
                    output_data=output_data,
                    reasoning=reasoning,
                    execution_time_ms=exec_ms,
                    success=success,
                    error_message=error_message,
                )
                return
            except Exception:
                # Fall back to minimal append below
                pass

        # Fallback entry structure (keeps "agent_type" and "success" keys for tests)
        entry = {
            "timestamp": finished_at.isoformat(),
            "case_id": case_id,
            "agent_type": agent_type,
            "action": action,
            "input_summary": str(input_data),
            "output_summary": str(output_data),
            "reasoning": reasoning,
            "execution_time_ms": exec_ms,
            "success": success,
            "error_message": error_message,
        }

        if hasattr(self.logger, "entries") and isinstance(getattr(self.logger, "entries"), list):
            self.logger.entries.append(entry)
            return

        # Last-resort no-op for unknown logger types
        return

# ===== PROMPT ENGINEERING HELPERS =====

def create_chain_of_thought_framework() -> Dict[str, str]:
    return {
        "step_1": "Data Review - Examine all available information",
        "step_2": "Pattern Recognition - Identify suspicious indicators",
        "step_3": "Regulatory Mapping - Connect to known typologies",
        "step_4": "Risk Quantification - Assess severity level",
        "step_5": "Classification Decision - Determine final category",
    }

def get_classification_categories() -> Dict[str, str]:
    return {
        "Structuring": "Transactions designed to avoid reporting thresholds",
        "Sanctions": "Potential sanctions violations or prohibited parties",
        "Fraud": "Fraudulent transactions or identity-related crimes",
        "Money_Laundering": "Complex schemes to obscure illicit fund sources",
        "Other": "Suspicious patterns not fitting standard categories",
    }

# ===== TESTING UTILITIES =====

def test_agent_with_sample_case():
    """
    Smoke test local do RiskAnalystAgent usando os schemas do foundation_sar.py.

    O objetivo aqui NÃO é substituir o pytest (test_risk_analyst.py), e sim permitir
    que você rode este módulo manualmente e valide rapidamente:
    - criação de CaseData com CustomerData/AccountData/TransactionData
    - chamada do agent.analyze_case()
    - validação do RiskAnalystOutput (Pydantic)
    - escrita de log via ExplainabilityLogger (entries + arquivo jsonl)
    """
    print("🧪 Testing Risk Analyst Agent (smoke test)")

    # ---- 1) Create sample data consistent with the schemas. ----
    customer = CustomerData(
        customer_id="CUST_0001",
        name="John Doe",
        date_of_birth="1985-04-12",
        ssn_last_4="1234",
        address="123 Main St, Miami, FL 33101",
        customer_since="2018-06-01",
        risk_rating="Medium",
        phone="+1-305-555-0101",
        occupation="Small Business Owner",
        annual_income=95000.0,
    )

    accounts = [
        AccountData(
            account_id="CUST_0001_ACC_1",
            customer_id="CUST_0001",
            account_type="Checking",
            opening_date="2018-06-01",
            current_balance=15250.75,
            average_monthly_balance=12000.50,
            status="Active",
        )
    ]

    transactions = [
        TransactionData(
            transaction_id="TXN_00000001",
            account_id="CUST_0001_ACC_1",
            transaction_date="2025-01-01",
            transaction_type="Cash_Deposit",
            amount=9900.00,
            description="Cash deposit at branch",
            method="Branch",
            counterparty=None,
            location="Miami, FL",
        ),
        TransactionData(
            transaction_id="TXN_00000002",
            account_id="CUST_0001_ACC_1",
            transaction_date="2025-01-03",
            transaction_type="Cash_Deposit",
            amount=9800.00,
            description="Cash deposit at branch",
            method="Branch",
            counterparty=None,
            location="Miami, FL",
        ),
        TransactionData(
            transaction_id="TXN_00000003",
            account_id="CUST_0001_ACC_1",
            transaction_date="2025-01-05",
            transaction_type="Wire_Transfer_Debit",
            amount=-15000.00,
            description="Outgoing wire transfer",
            method="Wire",
            counterparty="ACME IMPORTS LTD",
            location=None,
        ),
    ]

    case = CaseData(
        case_id="CASE_SAMPLE_0001",
        customer=customer,
        accounts=accounts,
        transactions=transactions,
        case_created_at="2026-01-12T00:00:00Z",
        data_sources={
            "customer_source": "manual_sample",
            "account_source": "manual_sample",
            "transaction_source": "manual_sample",
        },
    )

    # ---- 2) Create a mock OpenAI client (to avoid relying on a real API) ----
    # If you want to test with a real API, replace this FakeOpenAIClient with the real OpenAI client.
    class _FakeChoiceMsg:
        def __init__(self, content: str):
            self.content = content

    class _FakeChoice:
        def __init__(self, content: str):
            self.message = _FakeChoiceMsg(content)

    class _FakeResponse:
        def __init__(self, content: str):
            self.choices = [_FakeChoice(content)]

    class _FakeChatCompletions:
        def create(self, model, temperature, max_tokens, messages):
            output = RiskAnalystOutput(
                classification="Structuring",
                confidence_score=0.82,
                reasoning="Repeated cash deposits close to reporting thresholds indicate possible structuring.",
                key_indicators=[
                    "near-threshold cash deposits",
                    "repeated deposits"
                ],
                risk_level="High",
            )

            return _FakeResponse(
                json.dumps(output.model_dump())
            )

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeChatCompletions()

    class FakeOpenAIClient:
        def __init__(self):
            self.chat = _FakeChat()

    # ---- 3) Initialize logger and agent ----
    logger = ExplainabilityLogger(log_file="sar_audit_smoke_test.jsonl")
    client = FakeOpenAIClient()

    agent = RiskAnalystAgent(
        openai_client=client,
        explainability_logger=logger,
        model="gpt-4",
    )

    # ---- 4) Executar análise ----
    result = agent.analyze_case(case)

    # ---- 5) Validar resultados (Pydantic já valida no construtor) ----
    print("\n✅ RiskAnalystOutput:")
    if hasattr(result, "model_dump"):
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(result)

    # ---- 6) Validar log ----
    print("\n🧾 Audit log entries:")
    print(f"- entries_count: {len(getattr(logger, 'entries', []))}")
    if getattr(logger, "entries", None):
        print(json.dumps(logger.entries[-1], indent=2))

    print("\n✅ Smoke test concluído com sucesso.")


if __name__ == "__main__":
    print("🔍 Risk Analyst Agent Module")
    print("Chain-of-Thought reasoning for suspicious activity classification")
    print("\nRunning smoke test...\n")
    test_agent_with_sample_case()
