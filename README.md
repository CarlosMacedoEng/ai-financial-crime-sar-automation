# Automated SAR Detection & Reporting System

An end-to-end **Agentic AI system** designed to automate the detection of financial crimes and the generation of Suspicious Activity Reports (SARs). The system employs a **two-stage multi-agent architecture** (Risk Analyst and Compliance Officer) with human-in-the-loop validation to ensure regulatory compliance (FinCEN standards) while optimizing operational costs.

## Project Instructions

### 1. Foundation Architecture (Phase 1)
**File:** `src/foundation_sar.py`
- Implemented robust **Pydantic models** (`CustomerData`, `AccountData`, `TransactionData`) to enforce strict type validation on raw CSV data.
- Created the `DataLoader` class to aggregate fragmented banking data into unified `CaseData` objects.
- Developed an `ExplainabilityLogger` to maintain a comprehensive audit trail of all AI decisions.

### 2. Risk Analyst Agent (Phase 2)
**File:** `src/risk_analyst_agent.py`
- Built an AI agent using **Chain-of-Thought (CoT)** prompting to analyze financial behaviors step-by-step.
- Configured the agent to detect specific typologies (Structuring, Money Laundering, Fraud) and output structured JSON data with confidence scores.
- Implemented error handling to manage LLM response variability.

### 3. Compliance Officer Agent (Phase 3)
**File:** `src/compliance_officer_agent.py`
- Developed a specialized agent using the **ReACT (Reasoning + Action)** framework.
- Programmed the agent to generate regulatory-compliant SAR narratives adhering to FinCEN standards (max 120 words).
- Implemented validation logic to ensure all required regulatory citations are present.

### 4. Workflow Integration & SAR Generation (Phase 4)
**File:** `notebooks/phase_4_workflow_integration.ipynb`
- Orchestrated the complete **Two-Stage Workflow**: Risk Analysis → Human Decision Gate → Compliance Narrative.
- Implemented cost-optimization logic (filtering low-risk cases) and `create_sar_document` function to generate final JSON files.
- **Deliverables**:
    - Validated SAR JSON files in `outputs/filed_sars/`.
    - Automated end-to-end integration tests proving system reliability.

### Quick Testing Reference

```bash
# Test individual phases as you complete them
python -m pytest tests/test_foundation.py -v      # Phase 1: Foundation
python -m pytest tests/test_risk_analyst.py -v    # Phase 2: Risk Analyst  
python -m pytest tests/test_compliance_officer.py -v # Phase 3: Compliance Officer

# Final validation - all components working together
python -m pytest tests/ -v                        # All 30 tests should pass
```

Tests automatically skip when modules aren't implemented yet, providing clear feedback on progress.when building your README file for students.

## Getting Started

Instructions for how to get a copy of the project running on your local machine.

### Dependencies

To run this project, you will need the following libraries and tools:

```text
python >= 3.10
openai >= 1.0.0
pandas >= 2.0.0
pydantic >= 2.0.0
python-dotenv >= 1.0.0
pytest >= 7.0.0
```

## Testing

The project includes comprehensive test suites for all modules to ensure reliability and correctness.

### Running Tests

```bash
# Run all tests
cd project/solution
python -m pytest tests/ -v

# Run individual module tests
python -m pytest tests/test_foundation.py -v        # Core data structures
python -m pytest tests/test_risk_analyst.py -v     # Chain-of-Thought agent
python -m pytest tests/test_compliance_officer.py -v # ReACT agent
```

### Test Coverage

**30 comprehensive tests** across 3 modules:
- **Foundation SAR (10 tests)**: Data validation, case creation, audit logging
- **Risk Analyst Agent (10 tests)**: Agent initialization, case analysis, error handling  
- **Compliance Officer Agent (10 tests)**: Narrative generation, regulatory compliance

### Test Results

Complete validation proof demonstrating 100% success rate across all modules.

**Execution Log:**
```text
🔍 Validating Workflow Components
✅ Foundation components available
✅ Risk Analyst Agent available
✅ Compliance Officer Agent available
✅ Test modules available
🚀 All components ready - integration tests started...

============================= test session starts ==============================
tests/test_foundation.py::TestCustomerData::test_valid_customer_data ....... PASSED [ 10%]
tests/test_foundation.py::TestCustomerData::test_risk_rating_validation .... PASSED [ 20%]
tests/test_foundation.py::TestAccountData::test_valid_account_data ......... PASSED [ 30%]
tests/test_foundation.py::TestAccountData::test_balance_validation ......... PASSED [ 40%]
tests/test_foundation.py::TestTransactionData::test_valid_transaction_data . PASSED [ 50%]
tests/test_foundation.py::TestTransactionData::test_amount_validation ...... PASSED [ 60%]
tests/test_foundation.py::TestCaseData::test_valid_case_creation ........... PASSED [ 70%]
tests/test_foundation.py::TestDataLoader::test_csv_data_loading ............ PASSED [ 80%]
tests/test_foundation.py::TestExplainabilityLogger::test_log_creation ...... PASSED [ 90%]
tests/test_foundation.py::TestExplainabilityLogger::test_log_file_writing .. PASSED [100%]

tests/test_risk_analyst.py::TestRiskAnalystAgent::test_initialization ...... PASSED [ 10%]
tests/test_risk_analyst.py::TestRiskAnalystAgent::test_analyze_case_success  PASSED [ 20%]
tests/test_risk_analyst.py::TestRiskAnalystAgent::test_analyze_json_error .. PASSED [ 30%]
tests/test_risk_analyst.py::TestRiskAnalystAgent::test_extract_json_code ... PASSED [ 40%]
tests/test_risk_analyst.py::TestRiskAnalystAgent::test_extract_json_plain .. PASSED [ 50%]
tests/test_risk_analyst.py::TestRiskAnalystAgent::test_extract_json_empty .. PASSED [ 60%]
tests/test_risk_analyst.py::TestRiskAnalystAgent::test_format_accounts ..... PASSED [ 70%]
tests/test_risk_analyst.py::TestRiskAnalystAgent::test_format_transactions . PASSED [ 80%]
tests/test_risk_analyst.py::TestRiskAnalystAgent::test_prompt_structure .... PASSED [ 90%]
tests/test_risk_analyst.py::TestRiskAnalystAgent::test_api_parameters ...... PASSED [100%]

tests/test_compliance.py::TestComplianceOfficer::test_initialization ....... PASSED [ 10%]
tests/test_compliance.py::TestComplianceOfficer::test_narrative_success .... PASSED [ 20%]
tests/test_compliance.py::TestComplianceOfficer::test_word_count_limit ..... PASSED [ 30%]
tests/test_compliance.py::TestComplianceOfficer::test_json_parsing_error ... PASSED [ 40%]
tests/test_compliance.py::TestComplianceOfficer::test_extract_json_code .... PASSED [ 50%]
tests/test_compliance.py::TestComplianceOfficer::test_extract_json_plain ... PASSED [ 60%]
tests/test_compliance.py::TestComplianceOfficer::test_extract_json_empty ... PASSED [ 70%]
tests/test_compliance.py::TestComplianceOfficer::test_format_transactions .. PASSED [ 80%]
tests/test_compliance.py::TestComplianceOfficer::test_prompt_structure ..... PASSED [ 90%]
tests/test_compliance.py::TestComplianceOfficer::test_api_parameters ....... PASSED [100%]

================================================================================
📊 INTEGRATION TEST RESULTS:
   Foundation Components:    ✅ PASS
   Risk Analyst Agent:       ✅ PASS
   Compliance Officer Agent: ✅ PASS
   Overall Status:           ✅ ALL TESTS PASSED

🎉 Your system is ready for production workflow testing!
================================================================================

### Break Down Tests

**Foundation Tests**: Validate core data structures and utilities
- Customer/Account/Transaction data validation
- Case aggregation and schema compliance
- CSV data loading and audit logging

**Risk Analyst Tests**: Validate Chain-of-Thought analysis workflow  
- OpenAI API integration and response parsing
- JSON extraction from various response formats
- Error handling for malformed responses

**Compliance Officer Tests**: Validate ReACT regulatory narrative generation
- 120-word narrative limit enforcement
- Regulatory citation and terminology validation
- Multi-format response parsing and validation

## License
[License](../LICENSE.md)
