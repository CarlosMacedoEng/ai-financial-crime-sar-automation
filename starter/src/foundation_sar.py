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
import traceback


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(pd.isna(value))


def _required_string(value: Any, field_name: str) -> str:
    if _is_missing(value):
        raise ValueError(f"{field_name} is required")
    return str(value).strip()


def _optional_string(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    return str(value).strip()

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

    @field_validator("customer_id", "name", "date_of_birth", "address", "customer_since", mode="before")
    @classmethod
    def normalize_required_strings(cls, v: Any, info) -> str:
        return _required_string(v, info.field_name)

    @field_validator("phone", "occupation", mode="before")
    @classmethod
    def normalize_optional_strings(cls, v: Any) -> Optional[str]:
        return _optional_string(v)

    @field_validator("ssn_last_4", mode="before")
    @classmethod
    def normalize_ssn(cls, v: Any) -> str:
        if _is_missing(v):
            raise ValueError("ssn_last_4 is required")
        if isinstance(v, int):
            return f"{v:04d}"
        if isinstance(v, float):
            if math.isnan(v) or not v.is_integer():
                raise ValueError("ssn_last_4 must be a 4-digit integer")
            return f"{int(v):04d}"
        return str(v).strip()

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
    account_type: Literal["Checking", "Savings", "Money_Market", "Business_Checking", "Credit_Card"] = Field(...)
    opening_date: str = Field(..., description="Opening date in YYYY-MM-DD")
    current_balance: float = Field(..., description="Current balance (can be negative for overdraft)")
    average_monthly_balance: float = Field(..., description="Average monthly balance")
    status: Literal["Active", "Closed", "Suspended"] = Field(...)

    @field_validator("account_id", "customer_id", "opening_date", mode="before")
    @classmethod
    def normalize_required_strings(cls, v: Any, info) -> str:
        return _required_string(v, info.field_name)

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
    amount: float = Field(..., gt=0, le=1_000_000, description="Amount must be within allowed transaction limits")
    description: str = Field(..., description="Transaction description")
    method: Literal["ATM", "Branch", "Cash", "Electronic", "Mobile", "Online", "Wire", "ACH", "Test"] = Field(
        ..., description="Channel or method"
    )
    counterparty: Optional[str] = Field(None, description="Other party involved, if any")
    location: Optional[str] = Field(None, description="Location or branch, if any")

    @field_validator(
        "transaction_id",
        "account_id",
        "transaction_date",
        "description",
        "transaction_type",
        "method",
        mode="before",
    )
    @classmethod
    def normalize_required_strings(cls, v: Any, info) -> str:
        return _required_string(v, info.field_name)

    @field_validator("transaction_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v

    @field_validator("counterparty", "location", mode="before")
    @classmethod
    def blank_or_nan_to_none(cls, v: Any) -> Optional[str]:
        return _optional_string(v)

    @field_validator("amount", mode="before")
    @classmethod
    def validate_amount(cls, v: Any) -> float:
        # Ensure numeric, finite, and within explicit Field bounds
        val = float(v)
        if math.isnan(val) or math.isinf(val):
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
    analysis_steps: List[str] = Field(
        default_factory=list,
        description="Visible ordered reasoning steps aligned with the analysis framework",
    )
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

class DataIngestionError(ValueError):
    """Raised when CSV/raw input cannot be normalized into valid schema objects."""


class DataLoader:
    def __init__(self, explainability_logger: ExplainabilityLogger):
        self.logger = explainability_logger
        self.last_ingestion_report: Dict[str, Any] = {}

    @staticmethod
    def _build_data_error(file_name: str, field_name: str, value: Any, reason: str) -> DataIngestionError:
        return DataIngestionError(
            f"Invalid {file_name}: field '{field_name}' has value {repr(value)}. {reason}"
        )

    def _normalize_customer_dict(self, data: Dict[str, Any], file_name: str = "customers.csv") -> Dict[str, Any]:
        d = dict(data)
        d["customer_id"] = _required_string(d.get("customer_id"), "customer_id")
        d["name"] = _required_string(d.get("name"), "name")
        d["date_of_birth"] = _required_string(d.get("date_of_birth"), "date_of_birth")
        d["address"] = _required_string(d.get("address"), "address")
        d["customer_since"] = _required_string(d.get("customer_since"), "customer_since")
        d["risk_rating"] = _required_string(d.get("risk_rating"), "risk_rating")
        d["phone"] = _optional_string(d.get("phone"))
        d["occupation"] = _optional_string(d.get("occupation"))

        ssn_raw = d.get("ssn_last_4")
        if _is_missing(ssn_raw):
            raise self._build_data_error(file_name, "ssn_last_4", ssn_raw, "Expected a 4-digit identifier")
        if isinstance(ssn_raw, int):
            d["ssn_last_4"] = f"{ssn_raw:04d}"
        elif isinstance(ssn_raw, float):
            if math.isnan(ssn_raw) or not ssn_raw.is_integer():
                raise self._build_data_error(file_name, "ssn_last_4", ssn_raw, "Expected a 4-digit integer")
            d["ssn_last_4"] = f"{int(ssn_raw):04d}"
        else:
            d["ssn_last_4"] = str(ssn_raw).strip()

        income_raw = d.get("annual_income")
        d["annual_income"] = None if _is_missing(income_raw) else float(income_raw)
        return d

    def _normalize_account_dict(self, data: Dict[str, Any], file_name: str = "accounts.csv") -> Dict[str, Any]:
        d = dict(data)
        d["account_id"] = _required_string(d.get("account_id"), "account_id")
        d["customer_id"] = _required_string(d.get("customer_id"), "customer_id")
        d["account_type"] = _required_string(d.get("account_type"), "account_type")
        d["opening_date"] = _required_string(d.get("opening_date"), "opening_date")
        d["status"] = _required_string(d.get("status"), "status")
        d["current_balance"] = float(d.get("current_balance"))
        d["average_monthly_balance"] = float(d.get("average_monthly_balance"))
        return d

    def _normalize_transaction_dict(self, data: Dict[str, Any], file_name: str = "transactions.csv") -> Dict[str, Any]:
        d = dict(data)
        for field_name in [
            "transaction_id",
            "account_id",
            "transaction_date",
            "transaction_type",
            "description",
            "method",
        ]:
            d[field_name] = _required_string(d.get(field_name), field_name)

        amount_raw = d.get("amount")
        if _is_missing(amount_raw):
            raise self._build_data_error(file_name, "amount", amount_raw, "Amount cannot be empty")
        d["amount"] = float(amount_raw)
        d["counterparty"] = _optional_string(d.get("counterparty"))
        d["location"] = _optional_string(d.get("location"))
        return d

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame, required_columns: List[str], file_name: str) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise DataIngestionError(f"Invalid {file_name}: missing required columns {missing}")

    def create_case_from_data(
        self,
        customer_data: Dict,
        account_data: List[Dict],
        transaction_data: List[Dict],
    ) -> CaseData:
        start_time = datetime.now(timezone.utc)
        case_id = str(uuid.uuid4())
        try:
            normalized_customer = self._normalize_customer_dict(customer_data, "customers.csv")
            customer = CustomerData(**normalized_customer)
            accounts = [
                AccountData(**self._normalize_account_dict(acc, "accounts.csv"))
                for acc in account_data
                if _required_string(acc.get("customer_id"), "customer_id") == customer.customer_id
            ]
            account_ids = {acc.account_id for acc in accounts}
            transactions = [
                TransactionData(**self._normalize_transaction_dict(txn, "transactions.csv"))
                for txn in transaction_data
                if _required_string(txn.get("account_id"), "account_id") in account_ids
            ]
            if not transactions:
                raise DataIngestionError(
                    f"Cannot create case for customer '{customer.customer_id}': no linked transactions found."
                )
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
            trace = traceback.format_exc()
            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id=case_id,
                input_data={"customer": customer_data, "accounts": account_data, "transactions": transaction_data},
                output_data={},
                reasoning="Failed to create case",
                execution_time_ms=exec_ms,
                success=False,
                error_message=f"{str(e)}\n{trace}",
            )
            if isinstance(e, DataIngestionError):
                raise
            raise DataIngestionError(f"Failed to create case from input data: {str(e)}") from e

    def create_cases_from_csv(
        self,
        data_dir: str = "data/",
        skip_customers_without_transactions: bool = True,
    ) -> List[CaseData]:
        """
        Supported end-to-end path: raw CSV files -> normalized rows -> validated CaseData list.
        """
        start_time = datetime.now(timezone.utc)
        action_case_id = f"batch_{start_time.strftime('%Y%m%d%H%M%S')}"

        customers_df = accounts_df = transactions_df = None
        try:
            customers_df, accounts_df, transactions_df = load_csv_data(data_dir)

            self._validate_required_columns(
                customers_df,
                [
                    "customer_id",
                    "name",
                    "date_of_birth",
                    "ssn_last_4",
                    "address",
                    "customer_since",
                    "risk_rating",
                    "phone",
                    "occupation",
                    "annual_income",
                ],
                "customers.csv",
            )
            self._validate_required_columns(
                accounts_df,
                [
                    "account_id",
                    "customer_id",
                    "account_type",
                    "opening_date",
                    "current_balance",
                    "average_monthly_balance",
                    "status",
                ],
                "accounts.csv",
            )
            self._validate_required_columns(
                transactions_df,
                [
                    "transaction_id",
                    "account_id",
                    "transaction_date",
                    "transaction_type",
                    "amount",
                    "description",
                    "counterparty",
                    "location",
                    "method",
                ],
                "transactions.csv",
            )

            normalized_accounts: List[Dict[str, Any]] = []
            for row in accounts_df.to_dict(orient="records"):
                normalized_accounts.append(self._normalize_account_dict(row, "accounts.csv"))

            normalized_transactions: List[Dict[str, Any]] = []
            for row in transactions_df.to_dict(orient="records"):
                normalized_transactions.append(self._normalize_transaction_dict(row, "transactions.csv"))

            accounts_by_customer: Dict[str, List[Dict[str, Any]]] = {}
            for acc in normalized_accounts:
                accounts_by_customer.setdefault(acc["customer_id"], []).append(acc)

            transactions_by_account: Dict[str, List[Dict[str, Any]]] = {}
            for txn in normalized_transactions:
                transactions_by_account.setdefault(txn["account_id"], []).append(txn)

            cases: List[CaseData] = []
            skipped_customers: List[Dict[str, str]] = []

            for customer_row in customers_df.to_dict(orient="records"):
                normalized_customer = self._normalize_customer_dict(customer_row, "customers.csv")
                customer_id = normalized_customer["customer_id"]
                account_rows = accounts_by_customer.get(customer_id, [])
                account_ids = {acc["account_id"] for acc in account_rows}
                transaction_rows: List[Dict[str, Any]] = []
                for account_id in account_ids:
                    transaction_rows.extend(transactions_by_account.get(account_id, []))

                if not transaction_rows and skip_customers_without_transactions:
                    skipped_customers.append(
                        {
                            "customer_id": customer_id,
                            "reason": "No linked transactions found",
                        }
                    )
                    continue

                if not transaction_rows:
                    raise DataIngestionError(
                        f"Customer '{customer_id}' has no linked transactions. "
                        "Enable skip_customers_without_transactions=True or supply transaction rows."
                    )

                case = self.create_case_from_data(
                    customer_data=normalized_customer,
                    account_data=account_rows,
                    transaction_data=transaction_rows,
                )
                cases.append(case)

            self.last_ingestion_report = {
                "data_dir": data_dir,
                "customers_total": int(len(customers_df)),
                "accounts_total": int(len(accounts_df)),
                "transactions_total": int(len(transactions_df)),
                "cases_created": int(len(cases)),
                "customers_skipped_no_transactions": int(len(skipped_customers)),
                "skipped_customers": skipped_customers,
            }

            exec_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_cases_from_csv",
                case_id=action_case_id,
                input_data={"data_dir": data_dir, "skip_customers_without_transactions": skip_customers_without_transactions},
                output_data=self.last_ingestion_report,
                reasoning="Load raw CSVs, normalize values, and create validated CaseData objects",
                execution_time_ms=exec_ms,
                success=True,
                error_message=None,
            )
            return cases
        except Exception as e:
            exec_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            trace = traceback.format_exc()
            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_cases_from_csv",
                case_id=action_case_id,
                input_data={"data_dir": data_dir, "skip_customers_without_transactions": skip_customers_without_transactions},
                output_data={},
                reasoning="Failed CSV->CaseData ingestion flow",
                execution_time_ms=exec_ms,
                success=False,
                error_message=f"{str(e)}\n{trace}",
            )
            if isinstance(e, DataIngestionError):
                raise
            raise DataIngestionError(f"Failed loading CSV pipeline at '{data_dir}': {str(e)}") from e

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
