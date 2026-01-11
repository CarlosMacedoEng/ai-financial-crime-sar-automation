# Foundation SAR - Core Data Schemas and Utilities
# TODO: Implement core Pydantic schemas and data processing utilities

"""
This module contains the foundational components for SAR processing:

1. Pydantic Data Schemas:
   - CustomerData: Customer profile information
   - AccountData: Account details and balances  
   - TransactionData: Individual transaction records
   - CaseData: Unified case combining all data sources
   - RiskAnalystOutput: Risk analysis results
   - ComplianceOfficerOutput: Compliance narrative results

2. Utility Classes:
   - ExplainabilityLogger: Audit trail logging
   - DataLoader: Combines fragmented data into case objects

YOUR TASKS:
- Study the data files in data/ folder
- Design Pydantic schemas that match the CSV structure
- Implement validation rules for financial data
- Create a DataLoader that builds unified case objects
- Add proper error handling and logging
"""

import json
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator
import uuid
import os
from typing import Literal

# ===== TODO: IMPLEMENT PYDANTIC SCHEMAS =====

class CustomerData(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier like CUST_0001")
    name: str = Field(..., description="Full customer name")
    date_of_birth: str = Field(..., description="DOB in YYYY-MM-DD")
    ssn_last_4: str = Field(..., min_length=4, max_length=4, description="Last 4 SSN digits")
    address: str = Field(..., description="Full mailing address")
    customer_since: str = Field(..., description="Customer start date YYYY-MM-DD")
    risk_rating: Literal["Low", "Medium", "High"] = Field(..., description="Customer risk rating")
    phone: Optional[str] = Field(None, description="Contact phone number")
    occupation: Optional[str] = Field(None, description="Occupation/title")
    annual_income: Optional[float] = Field(None, ge=0, description="Annual income in USD")

    @field_validator("date_of_birth", "customer_since")
    @classmethod
    def validate_dates(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v

    @field_validator("ssn_last_4")
    @classmethod
    def validate_ssn(cls, v: str) -> str:
        if not v.isdigit() or len(v) != 4:
            raise ValueError("ssn_last_4 must be 4 digits")
        return v

class AccountData(BaseModel):
    account_id: str = Field(..., description="Unique account identifier like CUST_0001_ACC_1")
    customer_id: str = Field(..., description="Owning customer_id")
    account_type: Literal["Checking", "Savings", "Money_Market"] = Field(...)
    opening_date: str = Field(..., description="Opening date in YYYY-MM-DD")
    current_balance: float = Field(..., description="Current balance (can be negative for overdraft)")
    average_monthly_balance: float = Field(..., description="Average monthly balance")
    status: Literal["Active", "Closed", "Suspended"] = Field(...)

    @field_validator("opening_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v

class TransactionData(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction id like TXN_B24455F3")
    account_id: str = Field(..., description="Account id this transaction belongs to")
    transaction_date: str = Field(..., description="Date in YYYY-MM-DD")
    transaction_type: Literal[
        "ACH_Credit",
        "ACH_Debit",
        "ATM_Withdrawal",
        "Cash_Deposit",
        "Cash_Withdrawal",
        "Check_Deposit",
        "Debit_Purchase",
        "Direct_Deposit",
        "Online_Transfer",
        "Wire_Transfer",
        "Wire_Transfer_Credit",
        "Wire_Transfer_Debit",
    ] = Field(..., description="Transaction category")
    amount: float = Field(..., description="Amount can be negative for debits/withdrawals")
    description: str = Field(..., description="Transaction description")
    method: Literal["ATM", "Branch", "Cash", "Electronic", "Mobile", "Online", "Wire"] = Field(
        ..., description="Channel or method"
    )
    counterparty: Optional[str] = Field(None, description="Other party involved, if any")
    location: Optional[str] = Field(None, description="Location or branch, if any")

    @field_validator("transaction_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v

    @field_validator("counterparty", "location", mode="before")
    @classmethod
    def blank_to_none(cls, v):
        return None if v == "" else v

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v):
        # Ensure numeric and not NaN
        val = float(v)
        if val != val:  # NaN check
            raise ValueError("amount cannot be NaN")
        return val

class CaseData(BaseModel):
    case_id: str = Field(..., description="Unique case identifier")
    customer: CustomerData = Field(..., description="Customer information")
    accounts: List[AccountData] = Field(..., description="Accounts for this customer")
    transactions: List[TransactionData] = Field(..., description="Transactions tied to these accounts")
    case_created_at: str = Field(..., description="Case creation timestamp (ISO-8601)")
    data_sources: Dict[str, str] = Field(..., description="Data lineage metadata")

    @field_validator("transactions")
    @classmethod
    def validate_transactions_not_empty(cls, v: List[TransactionData]) -> List[TransactionData]:
        if not v:
            raise ValueError("transactions cannot be empty")
        return v

    @field_validator("accounts")
    @classmethod
    def validate_accounts_belong_to_customer(cls, v: List[AccountData], info):
        customer = info.data.get("customer")
        if customer:
            for acc in v:
                if acc.customer_id != customer.customer_id:
                    raise ValueError(f"Account {acc.account_id} does not belong to customer {customer.customer_id}")
        return v

    @field_validator("transactions")
    @classmethod
    def validate_transactions_belong_to_accounts(cls, v: List[TransactionData], info):
        accounts = info.data.get("accounts") or []
        account_ids = {acc.account_id for acc in accounts}
        for txn in v:
            if txn.account_id not in account_ids:
                raise ValueError(f"Transaction {txn.transaction_id} not linked to provided accounts")
        return v

class RiskAnalystOutput(BaseModel):
    classification: Literal["Structuring", "Sanctions", "Fraud", "Money_Laundering", "Other"] = Field(
        ..., description="Primary typology classification"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model/analyst confidence 0.0–1.0")
    reasoning: str = Field(..., max_length=500, description="Step-by-step rationale")
    key_indicators: List[str] = Field(..., description="Suspicious indicators found")
    risk_level: Literal["Low", "Medium", "High", "Critical"] = Field(..., description="Overall risk level")

class ComplianceOfficerOutput(BaseModel):
    """Compliance Officer agent structured output
    
    REQUIRED FIELDS (for ReACT agent output):
    - narrative: str = Regulatory narrative text (max 1000 chars for ≤200 words)
    - narrative_reasoning: str = Reasoning for narrative construction (max 500 chars)
    - regulatory_citations: List[str] = List of relevant regulations like:
      * "31 CFR 1020.320 (BSA)"
      * "12 CFR 21.11 (SAR Filing)"
      * "FinCEN SAR Instructions"
    - completeness_check: bool = Whether narrative meets all requirements
    
    HINT: Use Field(..., max_length=1000) for narrative length limit
    HINT: Use Field(..., max_length=500) for reasoning length limit
    HINT: Use bool type for completeness_check
    """
    # TODO: Implement the ComplianceOfficerOutput schema
    pass

# ===== TODO: IMPLEMENT AUDIT LOGGING =====

class ExplainabilityLogger:
    """Simple audit logging for compliance trails

    ATTRIBUTES:
    - log_file: str = Path to JSONL log file (default: "sar_audit.jsonl")
    - entries: List = In-memory storage of log entries

    METHODS:
    - log_agent_action(): Logs agent actions with structured data
    
    LOG ENTRY STRUCTURE (use this exact format):
    {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'case_id': case_id,
        'agent_type': agent_type,  # "DataLoader", "RiskAnalyst", "ComplianceOfficer"
        'action': action,          # "create_case", "analyze_case", "generate_narrative"
        'input_summary': str(input_data),
        'output_summary': str(output_data),
        'reasoning': reasoning,
        'execution_time_ms': execution_time_ms,
        'success': success,        # True/False
        'error_message': error_message  # None if success=True
    }
    
    HINT: Write each entry as JSON + newline to create JSONL format
    HINT: Use 'a' mode to append to log file
    HINT: Store entries in self.entries list AND write to file
    """
    
    def __init__(self, log_file: str = "sar_audit.jsonl"):
        # TODO: Initialize with log_file path and empty entries list
        pass
    
    def log_agent_action(self, agent_type: str, action: str, case_id: str, 
                        input_data: Dict, output_data: Dict, reasoning: str, 
                        execution_time_ms: float, success: bool = True, 
                        error_message: Optional[str] = None):
        """Log an agent action with essential context
        
        IMPLEMENTATION STEPS:
        1. Create entry dictionary with all fields (see structure above)
        2. Add entry to self.entries list
        3. Write entry to log file as JSON line
        
        HINT: Use json.dumps(entry) + '\n' for JSONL format
        HINT: Use datetime.now(timezone.utc).isoformat() for timestamp
        HINT: Convert input_data and output_data to strings with str()
        """
        # TODO: Implement logging with structured entry creation and file writing
        pass

# ===== TODO: IMPLEMENT DATA LOADER =====

class DataLoader:
    """Simple loader that creates case objects from CSV data
    
    ATTRIBUTES:
    - logger: ExplainabilityLogger = For audit logging
    
    HELPFUL METHODS:
    - create_case_from_data(): Creates CaseData from input dictionaries
    
    IMPLEMENTATION PATTERN:
    1. Start timing with start_time = datetime.now()
    2. Generate case_id with str(uuid.uuid4())
    3. Create CustomerData object from customer_data dict
    4. Filter accounts where acc['customer_id'] == customer.customer_id
    5. Get account_ids set from filtered accounts
    6. Filter transactions where txn['account_id'] in account_ids
    7. Create CaseData object with all components
    8. Calculate execution_time_ms
    9. Log success/failure with self.logger.log_agent_action()
    10. Return CaseData object (or raise exception on failure)
    """
    
    def __init__(self, explainability_logger: ExplainabilityLogger):
        # TODO: Store logger for audit trail
        pass
    
    def create_case_from_data(self, 
                            customer_data: Dict,
                            account_data: List[Dict],
                            transaction_data: List[Dict]) -> CaseData:
        """Create a unified case object from fragmented AML data

        SUGGESTED STEPS:
        1. Record start time for performance tracking
        2. Generate unique case_id using uuid.uuid4()
        3. Create CustomerData object from customer_data dictionary
        4. Filter account_data list for accounts belonging to this customer
        5. Create AccountData objects from filtered accounts
        6. Get set of account_ids from customer's accounts
        7. Filter transaction_data for transactions in customer's accounts
        8. Create TransactionData objects from filtered transactions  
        9. Create CaseData object combining all components
        10. Add case metadata (case_id, timestamp, data_sources)
        11. Calculate execution time in milliseconds
        12. Log operation with success/failure status
        13. Return CaseData object
        
        ERROR HANDLING:
        - Wrap in try/except block
        - Log failures with error message
        - Re-raise exceptions for caller
        
        DATA_SOURCES FORMAT:
        {
            'customer_source': f"csv_extract_{datetime.now().strftime('%Y%m%d')}",
            'account_source': f"csv_extract_{datetime.now().strftime('%Y%m%d')}",
            'transaction_source': f"csv_extract_{datetime.now().strftime('%Y%m%d')}"
        }
        
        HINT: Use list comprehensions for filtering
        HINT: Use set comprehension for account_ids: {acc.account_id for acc in accounts}
        HINT: Use datetime.now(timezone.utc).isoformat() for timestamps
        HINT: Calculate execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        """
        # TODO: Implement complete case creation with error handling and logging
        pass

# ===== HELPER FUNCTIONS (PROVIDED) =====

def load_csv_data(data_dir: str = "data/") -> tuple:
    """Helper function to load all CSV files
    
    Returns:
        tuple: (customers_df, accounts_df, transactions_df)
    """
    try:
        customers_df = pd.read_csv(f"{data_dir}/customers.csv")
        accounts_df = pd.read_csv(f"{data_dir}/accounts.csv") 
        transactions_df = pd.read_csv(f"{data_dir}/transactions.csv")
        return customers_df, accounts_df, transactions_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading CSV data: {e}")

if __name__ == "__main__":
    print("🏗️  Foundation SAR Module")
    print("Core data schemas and utilities for SAR processing")
    print("\n📋 TODO Items:")
    print("• Implement Pydantic schemas based on CSV data")
    print("• Create ExplainabilityLogger for audit trails")
    print("• Build DataLoader for case object creation")
    print("• Add comprehensive error handling")
    print("• Write unit tests for all components")
