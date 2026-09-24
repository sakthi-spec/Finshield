import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import pdf_reader, transaction_parser, anomaly_engine
from backend.rag import StatementSession


app = FastAPI(title="FinShield API")


# Holds the currently uploaded statement for this MVP.
# Later, this can be replaced with proper per-user/session storage.
current_session: StatementSession | None = None


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
    global current_session

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

        # 8. Return processed statement
        return {
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
        # 9. Remove temporary PDF
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/ask")
async def ask_finshield(request: AskRequest):
    if current_session is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a bank statement before asking questions."
        )

    try:
        result = current_session.ask(
            request.question,
            top_k=request.top_k
        )

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
        raise HTTPException(
            status_code=400,
            detail="Please upload a bank statement first."
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