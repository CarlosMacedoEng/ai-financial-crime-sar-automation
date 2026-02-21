# Compliance Officer Agent - ReACT Implementation
# TODO: Implement Compliance Officer Agent using ReACT prompting

"""
Compliance Officer Agent Module

This agent generates regulatory-compliant SAR narratives using ReACT prompting.
It takes risk analysis results and creates structured documentation for
FinCEN submission.

YOUR TASKS:
- Study ReACT (Reasoning + Action) prompting methodology
- Design system prompt with Reasoning/Action framework
- Implement narrative generation with word limits
- Validate regulatory compliance requirements
- Create proper audit logging and error handling
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

from foundation_sar import (
    ComplianceOfficerOutput,
    CaseData,
    RiskAnalystOutput,
    TransactionData,
)


class ComplianceOfficerAgent:
    """
    Compliance Officer agent using ReACT prompting framework.

    TODO: Implement agent that:
    - Uses Reasoning + Action structured prompting
    - Generates regulatory-compliant SAR narratives
    - Enforces word limits and terminology
    - Includes regulatory citations
    - Validates narrative completeness
    """

    def __init__(self, openai_client, explainability_logger, model="gpt-4o"):
        self.client = openai_client
        self.logger = explainability_logger
        self.model = model
        self.max_repair_attempts = 1
        self.max_compliance_repair_attempts = 1
        self.speculative_phrases = [
            "i think",
            "maybe",
            "probably",
            "possibly",
            "might be",
            "could be",
            "appears to",
        ]
        self.allowed_citation_patterns = [
            r"31\s*CFR\s*1020\.320",
            r"31\s*USC\s*5318\s*\(?g\)?",
            r"31\s*USC\s*5324",
            r"FinCEN\s+SAR\s+Instructions",
            r"12\s*CFR\s*21\.11",
        ]

        self.system_prompt = """
            You are a Senior Compliance Officer specializing in BSA/AML regulations.
            Your task is to generate a regulatory-compliant SAR (Suspicious Activity Report) narrative based on risk analysis findings.

            You must use the **ReACT (Reasoning + Action)** framework:

            **REASONING Phase:**
            1. Analyze the provided risk analysis and case data.
            2. Identify the specific suspicious pattern (e.g., Structuring, Money Laundering, Fraud).
            3. Select relevant regulatory keywords (e.g., "Bank Secrecy Act", "31 USC 5324", "no apparent business purpose").
            4. Verify if the activity meets filing thresholds.

            **ACTION Phase:**
            Generate a JSON response containing the SAR narrative and metadata.

            **CRITICAL CONSTRAINTS:**
            1. **Word Limit:** The narrative MUST be **120 words or less**. Concise and direct.
            2. **Format:** Standard FinCEN format (WHO, WHAT, WHEN, WHERE, WHY).
            3. **Tone:** Formal, objective, regulatory.
            4. **Citations:** Include specific regulatory citations (e.g., 31 CFR 1020.320).

            **OUTPUT FORMAT:**
            You must return ONLY a JSON object with this structure:
            {
                "narrative": "The actual text of the SAR narrative...",
                "narrative_reasoning": "Brief explanation of your approach...",
                "regulatory_citations": ["List of laws/regs cited"],
                "completeness_check": true
            }
        """

    def generate_compliance_narrative(self, case_data, risk_analysis) -> "ComplianceOfficerOutput":
        start_time = datetime.now()
        parse_error = None

        try:
            risk_summary = self._format_risk_analysis_for_prompt(risk_analysis)
            transactions_summary = self._format_transactions_for_compliance(case_data.transactions)

            case_summary = f"""
            Customer: {case_data.customer.name} (ID: {case_data.customer.customer_id})
            Transactions:
            {transactions_summary}
            """

            user_prompt = f"""
                Please generate a SAR narrative for the following case:

                --- CASE DATA ---
                {case_summary}

                --- RISK ANALYSIS ---
                {risk_summary}

                Remember: Maximum 120 words. Return JSON only.
            """

            try:
                response = self._request_narrative_completion(user_prompt)
            except Exception as e:
                raise RuntimeError(f"ComplianceOfficer API call failed for case '{case_data.case_id}': {e}") from e

            if not response.choices or not response.choices[0].message:
                raise ValueError("OpenAI returned an empty response structure.")

            raw_content = response.choices[0].message.content
            parsed_json, used_repair, parse_error = self._parse_with_repair(raw_content)
            if parsed_json is None:
                raise ValueError("Failed to parse Compliance Officer JSON output")

            narrative_text = parsed_json.get("narrative", "")
            self._validate_narrative_compliance(narrative_text)
            parsed_json, used_content_repair = self._validate_or_repair_content(
                parsed_json=parsed_json,
                case_data=case_data,
                risk_analysis=risk_analysis,
                case_summary=case_summary,
                risk_summary=risk_summary,
            )
            narrative_text = parsed_json.get("narrative", "")

            output = ComplianceOfficerOutput(
                narrative=narrative_text,
                narrative_reasoning=parsed_json.get("narrative_reasoning", ""),
                regulatory_citations=parsed_json.get("regulatory_citations", []),
                completeness_check=True,
            )

            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="ComplianceOfficer",
                action="generate_narrative",
                case_id=case_data.case_id,
                input_data={
                    "risk_level": risk_analysis.risk_level,
                    "classification": risk_analysis.classification,
                },
                output_data=parsed_json,
                reasoning=(
                    parsed_json.get("narrative_reasoning", "No reasoning provided")
                    if not (used_repair or used_content_repair)
                    else f"{parsed_json.get('narrative_reasoning', 'No reasoning provided')} [recovered_via_repair_retry]"
                ),
                execution_time_ms=execution_time_ms,
                success=True,
            )

            return output

        except Exception as e:
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            reasoning_msg = f"Error: {str(e)}"
            if "Failed to parse Compliance Officer JSON output" in str(e):
                reasoning_msg = "JSON parsing failed"
            elif "API call failed" in str(e):
                reasoning_msg = "OpenAI API call failed"
            elif parse_error:
                reasoning_msg = f"JSON parsing failed after repair fallback: {parse_error}"

            try:
                self.logger.log_agent_action(
                    agent_type="ComplianceOfficer",
                    action="generate_narrative_error",
                    case_id=case_data.case_id if case_data else "UNKNOWN",
                    input_data={},
                    output_data={},
                    reasoning=reasoning_msg,
                    execution_time_ms=execution_time_ms,
                    success=False,
                    error_message=str(e),
                )
            except Exception:
                pass

            raise

    def _request_narrative_completion(self, user_prompt: str):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

    def _request_repair_completion(self, broken_response: str):
        repair_system_prompt = (
            "You are a strict JSON repair assistant. "
            "Return exactly one valid JSON object and nothing else."
        )
        repair_user_prompt = (
            "Repair the following output into valid JSON using this schema exactly:\n"
            "{"
            '"narrative":"string",'
            '"narrative_reasoning":"string",'
            '"regulatory_citations":["string"],'
            '"completeness_check":true'
            "}\n\n"
            f"Original output:\n{broken_response}"
        )
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": repair_system_prompt},
                {"role": "user", "content": repair_user_prompt},
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

    def _parse_with_repair(self, raw_content: str):
        try:
            json_str = self._extract_json_from_response(raw_content)
            parsed_json = json.loads(json_str)
            if parsed_json is None:
                raise ValueError("Parsed JSON is None")
            return parsed_json, False, None
        except Exception as first_error:
            for attempt in range(self.max_repair_attempts):
                try:
                    repair_response = self._request_repair_completion(raw_content)
                    repair_content = repair_response.choices[0].message.content
                    json_str = self._extract_json_from_response(repair_content)
                    repaired = json.loads(json_str)
                    if repaired is None:
                        raise ValueError("Repair output JSON is None")
                    return repaired, True, None
                except Exception as retry_error:
                    if attempt == self.max_repair_attempts - 1:
                        return None, False, f"Initial parse error: {first_error}; repair error: {retry_error}"
            return None, False, str(first_error)

    def _validate_or_repair_content(
        self,
        parsed_json: Dict[str, Any],
        case_data: CaseData,
        risk_analysis: RiskAnalystOutput,
        case_summary: str,
        risk_summary: str,
    ) -> Tuple[Dict[str, Any], bool]:
        errors = self._compliance_content_errors(parsed_json, case_data, risk_analysis)
        if not errors:
            parsed_json["completeness_check"] = True
            return parsed_json, False

        for attempt in range(self.max_compliance_repair_attempts):
            try:
                repaired_response = self._request_compliance_repair(
                    case_summary=case_summary,
                    risk_summary=risk_summary,
                    current_output=parsed_json,
                    errors=errors,
                )
                repaired_content = repaired_response.choices[0].message.content
                repaired_json, _, parse_error = self._parse_with_repair(repaired_content)
                if repaired_json is None:
                    raise ValueError(parse_error or "Compliance repair returned invalid JSON")
                repaired_errors = self._compliance_content_errors(repaired_json, case_data, risk_analysis)
                if repaired_errors:
                    raise ValueError("; ".join(repaired_errors))
                repaired_json["completeness_check"] = True
                return repaired_json, True
            except Exception:
                if attempt == self.max_compliance_repair_attempts - 1:
                    break

        raise ValueError(f"Compliance validation failed: {'; '.join(errors)}")

    def _request_compliance_repair(
        self,
        case_summary: str,
        risk_summary: str,
        current_output: Dict[str, Any],
        errors: List[str],
    ):
        repair_system_prompt = (
            "You are a compliance QA reviewer. "
            "Return exactly one corrected JSON object and nothing else."
        )
        repair_user_prompt = (
            "The JSON output below failed compliance checks. "
            "Repair it while keeping narrative under 120 words and preserving factual objectivity.\n\n"
            f"Missing/invalid items: {errors}\n\n"
            "Case data:\n"
            f"{case_summary}\n\n"
            "Risk analysis:\n"
            f"{risk_summary}\n\n"
            "Current JSON:\n"
            f"{json.dumps(current_output)}\n\n"
            "Return JSON schema exactly:\n"
            "{"
            '"narrative":"string",'
            '"narrative_reasoning":"string",'
            '"regulatory_citations":["string"],'
            '"completeness_check":true'
            "}"
        )
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": repair_system_prompt},
                {"role": "user", "content": repair_user_prompt},
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

    def _compliance_content_errors(
        self,
        parsed_json: Dict[str, Any],
        case_data: CaseData,
        risk_analysis: RiskAnalystOutput,
    ) -> List[str]:
        errors: List[str] = []
        narrative = str(parsed_json.get("narrative", "")).strip()
        citations_raw = parsed_json.get("regulatory_citations", [])
        citations = citations_raw if isinstance(citations_raw, list) else [citations_raw]
        citations = [str(c).strip() for c in citations if str(c).strip()]

        if not self._has_subject_identifier(narrative, case_data):
            errors.append("missing subject identifier (customer name or customer ID)")
        if not self._has_date_reference(narrative):
            errors.append("missing date or timeframe reference")
        if not self._has_amount_reference(narrative):
            errors.append("missing amount or total reference")
        if not self._has_indicator_reference(narrative, risk_analysis):
            errors.append("missing suspicious indicator or typology reference")
        citation_errors = self._citation_errors(citations)
        errors.extend(citation_errors)
        if self._has_speculative_language(narrative):
            errors.append("contains speculative/non-objective language")

        return errors

    def _has_subject_identifier(self, narrative: str, case_data: CaseData) -> bool:
        text = narrative.lower()
        name = (case_data.customer.name or "").lower()
        customer_id = (case_data.customer.customer_id or "").lower()
        return bool((name and name in text) or (customer_id and customer_id in text))

    def _has_date_reference(self, narrative: str) -> bool:
        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            r"\bfrom\b.*\bto\b",
            r"\bover\b.*\bdays?\b",
        ]
        return any(re.search(pattern, narrative, flags=re.IGNORECASE) for pattern in date_patterns)

    def _has_amount_reference(self, narrative: str) -> bool:
        return bool(
            re.search(r"\$\s?\d{1,3}(,\d{3})*(\.\d{2})?", narrative)
            or re.search(r"\btotal(?:ing)?\s+\d", narrative, flags=re.IGNORECASE)
        )

    def _has_indicator_reference(self, narrative: str, risk_analysis: RiskAnalystOutput) -> bool:
        text = narrative.lower()
        typology_terms = [
            "structuring",
            "sanctions",
            "fraud",
            "money laundering",
            "suspicious activity",
            "red flag",
            "threshold",
            "layering",
        ]
        if any(term in text for term in typology_terms):
            return True
        return any(str(ind).lower() in text for ind in getattr(risk_analysis, "key_indicators", []))

    def _has_speculative_language(self, narrative: str) -> bool:
        text = narrative.lower()
        return any(phrase in text for phrase in self.speculative_phrases)

    def _citation_errors(self, citations: List[str]) -> List[str]:
        if not citations:
            return ["missing regulatory citations"]
        is_valid = False
        for citation in citations:
            if any(re.search(pattern, citation, flags=re.IGNORECASE) for pattern in self.allowed_citation_patterns):
                is_valid = True
                break
        if not is_valid:
            return ["regulatory citations missing recognized BSA/AML reference format"]
        return []

    def _extract_json_from_response(self, response_content: str) -> str:
        """
        Extract JSON string from LLM response handling Markdown code blocks.
        """
        if not response_content or not response_content.strip():
            raise ValueError("No JSON content found (empty response)")

        json_match = re.search(r"```json\s*(.*?)\s*```", response_content, re.DOTALL)

        if json_match:
            return json_match.group(1).strip()

        return response_content.strip()

    def _format_transactions_for_compliance(self, transactions: List[TransactionData]) -> str:
        """
        Format transactions list for the narrative prompt.
        """
        formatted_txns = []
        for i, txn in enumerate(transactions, 1):
            amount_str = f"${float(txn.amount):,.2f}"
            location_str = f"at {txn.location}" if txn.location else ""
            method_str = f"via {txn.method}" if txn.method else ""

            line = f"{i}. {txn.transaction_date}: {amount_str} {txn.transaction_type} {location_str} {method_str}".strip()
            line = re.sub(r"\s+", " ", line)
            formatted_txns.append(line)

        return "\\n".join(formatted_txns)

    def _format_risk_analysis_for_prompt(self, risk_analysis: RiskAnalystOutput) -> str:
        """Format risk analysis results for compliance prompt."""
        return f"""
        Classification: {risk_analysis.classification}
        Risk Level: {risk_analysis.risk_level}
        Confidence: {risk_analysis.confidence_score}
        Key Indicators: {', '.join(risk_analysis.key_indicators)}
        Analyst Reasoning: {risk_analysis.reasoning}
        """

    def _format_case_data(self, case_data: CaseData) -> str:
        """Helper to format basic case info"""
        customer = case_data.customer
        return f"""
        Customer: {customer.name} (ID: {customer.customer_id})
        Occupation/Type: {getattr(customer, 'occupation', 'Unknown')}
        Transactions Count: {len(case_data.transactions)}
        """

    def _validate_narrative_compliance(self, narrative: str) -> bool:
        """Validate narrative meets regulatory requirements (max 120 words)."""
        word_count = len(narrative.split())
        if word_count > 120:
            raise ValueError(f"Narrative exceeds 120 word limit. Count: {word_count}")
        return True


# ===== REACT PROMPTING HELPERS =====

def create_react_framework():
    """Helper function showing ReACT structure

    TODO: Study this example and adapt for compliance narratives:

    **REASONING Phase:**
    1. Review the risk analyst's findings
    2. Assess regulatory narrative requirements
    3. Identify key compliance elements
    4. Consider narrative structure

    **ACTION Phase:**
    1. Draft concise narrative (<=120 words)
    2. Include specific details and amounts
    3. Reference suspicious activity pattern
    4. Ensure regulatory language
    """
    return {
        "reasoning_phase": [
            "Review risk analysis findings",
            "Assess regulatory requirements",
            "Identify compliance elements",
            "Plan narrative structure",
        ],
        "action_phase": [
            "Draft concise narrative",
            "Include specific details",
            "Reference activity patterns",
            "Use regulatory language",
        ],
    }


def get_regulatory_requirements():
    """Key regulatory requirements for SAR narratives

    TODO: Use these requirements in your prompts:
    """
    return {
        "word_limit": 120,
        "required_elements": [
            "Customer identification",
            "Suspicious activity description",
            "Transaction amounts and dates",
            "Why activity is suspicious",
        ],
        "terminology": [
            "Suspicious activity",
            "Regulatory threshold",
            "Financial institution",
            "Money laundering",
            "Bank Secrecy Act",
        ],
        "citations": [
            "31 CFR 1020.320 (BSA)",
            "12 CFR 21.11 (SAR Filing)",
            "FinCEN SAR Instructions",
        ],
    }


# ===== TESTING UTILITIES =====

def test_narrative_generation():
    """Test the agent with sample risk analysis

    TODO: Use this function to test your implementation:
    - Create sample risk analysis results
    - Initialize compliance agent
    - Generate narrative
    - Validate compliance requirements
    """
    print("Testing Compliance Officer Agent")
    print("TODO: Implement test case")


def validate_word_count(text: str, max_words: int = 120) -> bool:
    """Helper to validate word count

    TODO: Use this utility in your validation:
    """
    word_count = len(text.split())
    return word_count <= max_words


if __name__ == "__main__":
    print("Compliance Officer Agent Module")
    print("ReACT prompting for regulatory narrative generation")
    print("\nTODO Items:")
    print("- Design ReACT system prompt")
    print("- Implement generate_compliance_narrative method")
    print("- Add narrative validation (word count, terminology)")
    print("- Create regulatory citation system")
    print("- Test with sample risk analysis results")
    print("\nKey Concepts:")
    print("- ReACT: Reasoning + Action structured prompting")
    print("- Regulatory Compliance: BSA/AML requirements")
    print("- Narrative Constraints: Word limits and terminology")
    print("- Audit Logging: Complete decision documentation")
