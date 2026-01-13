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
import math
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
        "Deposit",
        "Test",
    ] = Field(..., description="Transaction category")
    amount: float = Field(..., description="Amount can be negative for debits/withdrawals")
    description: str = Field(..., description="Transaction description")
    method: Literal["ATM", "Branch", "Cash", "Electronic", "Mobile", "Online", "Wire", "ACH", "Test"] = Field(
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

        if not accounts:
            return v

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
    narrative: str = Field(..., max_length=1000, description="Regulatory SAR narrative (≤ ~200 words)")
    narrative_reasoning: str = Field(..., max_length=500, description="Reasoning behind the narrative")
    regulatory_citations: List[str] = Field(..., description="Relevant regulations (e.g., 31 CFR 1020.320)")
    completeness_check: bool = Field(..., description="Whether the narrative meets all requirements")


# ===== TODO: IMPLEMENT AUDIT LOGGING =====

class ExplainabilityLogger:
    def __init__(self, log_file: str = "sar_audit.jsonl"):
        self.log_file = log_file
        self.entries: List[Dict[str, Any]] = []

    def log_agent_action(
        self,
        agent_type: str,
        action: str,
        case_id: str,
        input_data: Dict,
        output_data: Dict,
        reasoning: str,
        execution_time_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "agent_type": agent_type,
            "action": action,
            "input_summary": str(input_data),
            "output_summary": str(output_data),
            "reasoning": reasoning,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "error_message": error_message,
        }
        self.entries.append(entry)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

# ===== TODO: IMPLEMENT DATA LOADER =====

class DataLoader:
    def __init__(self, explainability_logger: ExplainabilityLogger):
        self.logger = explainability_logger

    def create_case_from_data(
        self,
        customer_data: Dict,
        account_data: List[Dict],
        transaction_data: List[Dict],
    ) -> CaseData:
        start_time = datetime.now(timezone.utc)
        case_id = str(uuid.uuid4())
        try:
            customer = CustomerData(**customer_data)
            accounts = [
                AccountData(**acc) for acc in account_data if acc.get("customer_id") == customer.customer_id
            ]
            account_ids = {acc.account_id for acc in accounts}
            transactions = [
                TransactionData(**txn)
                for txn in transaction_data
                if txn.get("account_id") in account_ids
            ]
            case = CaseData(
                case_id=case_id,
                customer=customer,
                accounts=accounts,
                transactions=transactions,
                case_created_at=datetime.now(timezone.utc).isoformat(),
                data_sources={
                    "customer_source": f"csv_extract_{start_time.strftime('%Y%m%d')}",
                    "account_source": f"csv_extract_{start_time.strftime('%Y%m%d')}",
                    "transaction_source": f"csv_extract_{start_time.strftime('%Y%m%d')}",
                },
            )
            exec_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id=case.case_id,
                input_data={"customer": customer_data, "accounts": account_data, "transactions": transaction_data},
                output_data=case.model_dump(),
                reasoning="Create case from CSV fragments",
                execution_time_ms=exec_ms,
                success=True,
                error_message=None,
            )
            return case
        except Exception as e:
            exec_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id=case_id,
                input_data={"customer": customer_data, "accounts": account_data, "transactions": transaction_data},
                output_data={},
                reasoning="Failed to create case",
                execution_time_ms=exec_ms,
                success=False,
                error_message=str(e),
            )
            raise

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

def nan_to_none(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v

def normalize_transaction_dict(txn: dict) -> dict:
    d = dict(txn)
    # These fields are Optional[str] in your schema
    d["counterparty"] = nan_to_none(d.get("counterparty"))
    d["location"] = nan_to_none(d.get("location"))

    # Ensure strings if not None
    if d["counterparty"] is not None:
        d["counterparty"] = str(d["counterparty"])
    if d["location"] is not None:
        d["location"] = str(d["location"])

    # Common CSV typing issues: ensure these are strings
    for k in ["transaction_id", "account_id", "transaction_date", "transaction_type", "description", "method"]:
        if k in d and d[k] is not None:
            d[k] = str(d[k])

    # amount must be numeric
    if "amount" in d and d["amount"] is not None:
        d["amount"] = float(d["amount"])

    return d

if __name__ == "__main__":
    print("🏗️  Foundation SAR Module")
    print("Core data schemas and utilities for SAR processing")
    print("\n📋 TODO Items:")
    print("• Implement Pydantic schemas based on CSV data")
    print("• Create ExplainabilityLogger for audit trails")
    print("• Build DataLoader for case object creation")
    print("• Add comprehensive error handling")
    print("• Write unit tests for all components")
