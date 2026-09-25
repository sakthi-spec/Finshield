import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import pdf_reader, transaction_parser
from backend.rag import StatementSession, generate_answer
from datetime import datetime, timezone
from uuid import uuid4

from backend.database import (
    statements_collection,
    transactions_collection,
    questions_collection,
)


app = FastAPI(title="FinShield API")


# Holds the currently uploaded statement for this MVP.
# Later, this can be replaced with proper per-user/session storage.
current_session: StatementSession | None = None
current_statement_id: str | None = None
def restore_latest_session():
    global current_session, current_statement_id

    try:
        latest_statement = statements_collection.find_one(
            {},
            sort=[("uploaded_at", -1)]
        )

        if not latest_statement:
            return False

        statement_id = latest_statement["_id"]

        stored_transactions = list(
            transactions_collection.find(
                {"statement_id": statement_id},
                {"_id": 0}
            )
        )

        if not stored_transactions:
            return False

        current_session = StatementSession(
            stored_transactions
        )

        current_statement_id = statement_id

        return True

    except Exception as e:
        print(f"Failed to restore latest statement: {e}")
        return False

class AskRequest(BaseModel):
    question: str
    top_k: int = 3


def detect_global_scope(question: str) -> str | None:
    """Classify explicit vault/history questions before statement RAG."""
    normalized = " ".join(question.lower().split())

    global_markers = (
        "overall",
        "across statements",
        "across all statements",
        "across my statements",
        "all statements",
        "all uploaded statements",
        "all stored statements",
        "vault",
        "stored statements",
        "uploaded statements",
        "transaction history",
        "statement history",
        "entire history",
        "pdfs",
        "pdf files",
    )

    if not any(marker in normalized for marker in global_markers):
        return None

    if "statement" in normalized or "pdf" in normalized:
        if (
            "transaction" not in normalized
            and any(word in normalized for word in ("how many", "number of", "count"))
        ):
            return "global_statement_count"

    if "transaction" in normalized and any(
        word in normalized for word in ("how many", "number of", "count")
    ):
        return "global_transaction_count"

    if "average" in normalized or "avg" in normalized:
        return "global_average"

    if any(
        phrase in normalized
        for phrase in (
            "total spending",
            "total spend",
            "total amount",
            "how much did i spend",
            "how much have i spent",
            "what did i spend",
            "overall spending",
        )
    ):
        return "global_total"

    return "global_history"


def build_global_history_summary() -> dict:
    """Calculate exact aggregates across all MongoDB-stored transactions."""
    amounts = [
        float(transaction.get("amount", 0))
        for transaction in transactions_collection.find(
            {},
            {"_id": 0, "amount": 1},
        )
    ]
    total_spending = sum(amounts)

    return {
        "transaction_count": len(amounts),
        "total_spending": round(total_spending, 2),
        "average_transaction": round(
            total_spending / len(amounts),
            2,
        ) if amounts else 0.0,
        "statement_count": statements_collection.count_documents({}),
    }


def build_global_answer(question: str, scope: str) -> dict:
    """Return an exact aggregate result without touching StatementSession."""
    summary = build_global_history_summary()
    computed_results = {"global_scope": "global_history"}

    if scope == "global_statement_count":
        count = summary["statement_count"]
        computed_results["statement_count"] = count
        exact_result = f"Statements stored in FinShield Vault: {count}"
    elif scope == "global_transaction_count":
        count = summary["transaction_count"]
        computed_results["transaction_count"] = count
        exact_result = f"Transactions across all stored statements: {count}"
    elif scope == "global_average":
        average = summary["average_transaction"]
        computed_results["average_transaction"] = average
        exact_result = (
            "Average transaction across all stored statements: "
            f"₹{average:.2f}"
        )
    elif scope == "global_total":
        total = summary["total_spending"]
        computed_results["total_spending"] = total
        exact_result = f"Total spending across all stored statements: ₹{total:.2f}"
    else:
        computed_results.update(summary)
        exact_result = (
            f"FinShield Vault contains {summary['statement_count']} statements, "
            f"{summary['transaction_count']} transactions, and total spending "
            f"of ₹{summary['total_spending']:.2f}."
        )

    return {
        "question": question,
        "intent": scope,
        "scope": "global_history",
        "intents": [scope],
        "answer": generate_answer(
            question,
            [],
            exact_result=exact_result,
        ),
        "computed_results": computed_results,
        "transactions_used": [],
    }


@app.get("/")
def root():
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend",
        "index.html"
    )

    return FileResponse(frontend_path)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/upload")
async def upload_statement(file: UploadFile = File(...)):
    global current_session, current_statement_id
    statement_id = str(uuid4())
    current_statement_id = statement_id

    # 1. Validate file
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF bank statement."
        )

    temp_path = None

    try:
        # 2. Read uploaded PDF
        contents = await file.read()

        # 3. Save temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        # 4. Extract raw text
        raw_text = pdf_reader.extract_text_from_pdf(temp_path)

        # 5. Parse into structured transactions
        transactions = transaction_parser.parse_transactions(raw_text)

        if not transactions:
            raise HTTPException(
                status_code=400,
                detail="No transactions could be extracted from the PDF."
            )

        # 6. Create RAG session and compute statement-scoped anomalies.
        current_session = StatementSession(transactions)
        annotated_transactions = current_session.anomalies

        # 7. Create a unique ID for this uploaded statement
        statement_id = str(uuid4())
        uploaded_at = datetime.now(timezone.utc)

        # 9. Save statement metadata to MongoDB
        statements_collection.insert_one({
            "_id": statement_id,
            "filename": file.filename,
            "uploaded_at": uploaded_at,
            "transaction_count": len(annotated_transactions),
        })

        # 10. Save transactions to MongoDB
        transaction_documents = []

        for transaction in annotated_transactions:
            transaction_document = dict(transaction)
            transaction_document["statement_id"] = statement_id
            transaction_documents.append(transaction_document)

        if transaction_documents:
            transactions_collection.insert_many(transaction_documents)

        # 11. Return processed statement
        return {
            "statement_id": statement_id,
            "filename": file.filename,
            "transaction_count": len(annotated_transactions),
            "transactions": annotated_transactions,
            "rag_ready": True
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process statement: {str(e)}"
        )

    finally:
        # 12. Remove temporary PDF
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/ask")
async def ask_finshield(request: AskRequest):
    global_scope = detect_global_scope(request.question)

    if global_scope is not None:
        return build_global_answer(
            request.question,
            global_scope,
        )

    if current_session is None:
        restored = restore_latest_session()

        if not restored or current_session is None:
            raise HTTPException(
                status_code=400,
                detail="Please upload a bank statement first.",
            )

    return current_session.ask(
        request.question,
        top_k=request.top_k,
    )


def build_financial_overview(session: StatementSession) -> dict:
    transactions = session.transactions

    # 1. Total spending
    total_spending = sum(
        float(transaction["amount"])
        for transaction in transactions
    )

    # 2. Average transaction
    average_transaction = (
        total_spending / len(transactions)
        if transactions
        else 0.0
    )

    # 3. Largest transaction
    largest_transaction = max(
        transactions,
        key=lambda transaction: float(transaction["amount"]),
        default=None,
    )

    # 4. Category spending
    category_spending = {}

    for transaction in transactions:
        category = transaction.get("category", "Other")

        category_spending[category] = (
            category_spending.get(category, 0.0)
            + float(transaction["amount"])
        )

    # 5. Potentially unusual transactions
    # Only MEDIUM, HIGH and CRITICAL are shown
    # in the main dashboard.
    potentially_unusual = [
        transaction
        for transaction in session.anomalies
        if transaction["severity"] in {
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        }
    ]

    return {
        "transaction_count": len(transactions),
        "total_spending": round(total_spending, 2),
        "average_transaction": round(
            average_transaction,
            2,
        ),
        "largest_transaction": (
            {
                "id": largest_transaction["id"],
                "date": largest_transaction["date"],
                "merchant": largest_transaction["merchant"],
                "amount": largest_transaction["amount"],
                "category": largest_transaction["category"],
            }
            if largest_transaction
            else None
        ),
        "category_spending": {
            category: round(amount, 2)
            for category, amount in category_spending.items()
        },
        "potentially_unusual_count": len(
            potentially_unusual
        ),
    }


@app.get("/overview")
async def financial_overview():
    if current_session is None:
        restore_latest_session()

    if current_session is None:
        raise HTTPException(
            status_code=400,
            detail="No saved bank statement found. Please upload a statement."
        )

    try:
        return build_financial_overview(
            current_session
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build financial overview: {str(e)}"
        )


def _serialize_datetime(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@app.get("/transactions")
def list_all_transactions():
    """Return the global transaction history with statement metadata."""
    try:
        statements = {
            statement["_id"]: statement
            for statement in statements_collection.find(
                {},
                {
                    "_id": 1,
                    "filename": 1,
                    "uploaded_at": 1,
                },
            )
        }

        transactions = []
        for transaction in transactions_collection.find({}, {"_id": 0}):
            statement_id = transaction.get("statement_id")
            statement = statements.get(statement_id, {})
            transactions.append({
                **transaction,
                "statement_id": statement_id,
                "filename": statement.get("filename"),
                "uploaded_at": _serialize_datetime(
                    statement.get("uploaded_at")
                ),
            })

        return {
            "count": len(transactions),
            "transactions": transactions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load transaction history: {str(e)}",
        )


@app.get("/history-summary")
def history_summary():
    """Calculate aggregate metrics across every stored transaction."""
    try:
        return build_global_history_summary()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build transaction history summary: {str(e)}",
        )


@app.get("/files")
def list_statement_files():
    try:
        statements = statements_collection.find(
            {},
            {
                "_id": 1,
                "filename": 1,
                "uploaded_at": 1,
                "transaction_count": 1,
            }
        ).sort("uploaded_at", -1)

        files = []

        for statement in statements:
            statement_id = statement["_id"]

            transactions = list(
                transactions_collection.find(
                    {"statement_id": statement_id},
                    {
                        "_id": 0,
                        "amount": 1,
                        "severity": 1,
                    }
                )
            )

            total_spending = sum(
                float(transaction.get("amount", 0))
                for transaction in transactions
            )

            unusual_count = sum(
                1
                for transaction in transactions
                if transaction.get("severity") in {
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL",
                }
            )

            files.append({
                "statement_id": statement_id,
                "filename": statement.get("filename"),
                "uploaded_at": _serialize_datetime(
                    statement.get("uploaded_at")
                ),
                "transaction_count": statement.get(
                    "transaction_count",
                    len(transactions)
                ),
                "total_spending": round(total_spending, 2),
                "potentially_unusual_count": unusual_count,
            })

        return {
            "count": len(files),
            "files": files,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load statement files: {str(e)}"
        )


@app.get("/files/{statement_id}")
def get_statement_file(statement_id: str):
    try:
        statement = statements_collection.find_one(
            {"_id": statement_id}
        )

        if not statement:
            raise HTTPException(
                status_code=404,
                detail="Statement not found."
            )

        transactions = list(
            transactions_collection.find(
                {"statement_id": statement_id},
                {"_id": 0}
            )
        )

        questions = list(
            questions_collection.find(
                {"statement_id": statement_id},
                {"_id": 0}
            ).sort("asked_at", -1)
        )

        total_spending = sum(
            float(transaction.get("amount", 0))
            for transaction in transactions
        )

        unusual_transactions = [
            transaction
            for transaction in transactions
            if transaction.get("severity") in {
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            }
        ]

        largest_transaction = max(
            transactions,
            key=lambda transaction: float(
                transaction.get("amount", 0)
            ),
            default=None,
        )

        category_spending = {}

        for transaction in transactions:
            category = transaction.get("category", "Other")

            category_spending[category] = (
                category_spending.get(category, 0.0)
                + float(transaction.get("amount", 0))
            )

        return {
            "statement": {
                "statement_id": statement["_id"],
                "filename": statement.get("filename"),
                "uploaded_at": _serialize_datetime(
                    statement.get("uploaded_at")
                ),
                "transaction_count": len(transactions),
            },
            "summary": {
                "total_spending": round(total_spending, 2),
                "potentially_unusual_count": len(
                    unusual_transactions
                ),
                "largest_transaction": (
                    {
                        "id": largest_transaction.get("id"),
                        "date": largest_transaction.get("date"),
                        "merchant": largest_transaction.get("merchant"),
                        "amount": largest_transaction.get("amount"),
                        "category": largest_transaction.get("category"),
                    }
                    if largest_transaction
                    else None
                ),
                "category_spending": {
                    category: round(amount, 2)
                    for category, amount in category_spending.items()
                },
            },
            "transactions": transactions,
            "questions": questions,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load statement: {str(e)}"
        )