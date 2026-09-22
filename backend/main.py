import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException

from backend import pdf_reader, transaction_parser, anomaly_engine

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "FinShield API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/upload")
async def upload_statement(file: UploadFile = File(...)):
    # 1. Save temporarily
    contents = await file.read()

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        # 2. Extract raw text
        raw_text = pdf_reader.extract_text_from_pdf(temp_path)

        # 3. Parse into structured transactions
        transactions = transaction_parser.parse_transactions(raw_text)

        # 4. Run the layered anomaly engine
        annotated_transactions = anomaly_engine.evaluate_statement(
            transactions
        )

        # 5. Return the complete result
        return {
            "filename": file.filename,
            "transaction_count": len(annotated_transactions),
            "transactions": annotated_transactions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process statement: {str(e)}"
        )

    finally:
        # 6. Remove temporary PDF
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)