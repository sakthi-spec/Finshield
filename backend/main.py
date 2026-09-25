import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import pdf_reader, transaction_parser, anomaly_engine
from backend.rag import StatementSession
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

        # 6. Run layered anomaly engine
        annotated_transactions = anomaly_engine.evaluate_statement(
            transactions
        )

        # 7. Create RAG session
        #    Embeddings are created ONCE here.
        current_session = StatementSession(transactions)

        # 8. Create a unique ID for this uploaded statement
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
    if current_session is None:
        restore_latest_session()

    if current_session is None:
        raise HTTPException(
            status_code=400,
            detail="No saved bank statement found. Please upload a statement."
        )

    try:
        result = current_session.ask(
            request.question,
            top_k=request.top_k
        )

        # Save the question and answer to MongoDB
        question_document = {
            "_id": str(uuid4()),
            "statement_id": current_statement_id,
            "question": request.question,
            "top_k": request.top_k,
            "intent": result.get("intent"),
            "intents": result.get("intents", []),
            "computed_results": result.get("computed_results", {}),
            "answer": result.get("answer", ""),
            "transactions_used": result.get("transactions_used", []),
            "asked_at": datetime.now(timezone.utc),
        }

        questions_collection.insert_one(question_document)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
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