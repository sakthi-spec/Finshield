import os
from typing import List, Dict, Tuple

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from pdf_reader import extract_text_from_pdf
from transaction_parser import parse_transactions


# --------------------------------------------------
# 1. Load our API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 2. Turn a transaction into searchable text
# --------------------------------------------------

def transaction_to_text(transaction: Dict) -> str:
    return (
        f"Date: {transaction['date']}. "
        f"Merchant: {transaction['merchant']}. "
        f"Amount: ₹{transaction['amount']:.2f}."
    )


# --------------------------------------------------
# 3. Create embeddings for transactions
# --------------------------------------------------

def create_document_embeddings(
    transactions: List[Dict],
) -> List[List[float]]:

    texts = [
        transaction_to_text(transaction)
        for transaction in transactions
    ]

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )

    return [embedding.values for embedding in result.embeddings]


# --------------------------------------------------
# 4. Create an embedding for the user's question
# --------------------------------------------------

def create_query_embedding(question: str) -> List[float]:

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    return result.embeddings[0].values


# --------------------------------------------------
# 5. Compare two embeddings
# --------------------------------------------------

def cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:

    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# --------------------------------------------------
# 6. Retrieve the most relevant transactions
# --------------------------------------------------

def retrieve_transactions(
    transactions: List[Dict],
    embeddings: List[List[float]],
    question: str,
    top_k: int = 3,
) -> List[Tuple[Dict, float]]:

    query_embedding = create_query_embedding(question)

    scored_results = []

    for transaction, embedding in zip(
        transactions,
        embeddings,
    ):
        score = cosine_similarity(
            query_embedding,
            embedding,
        )

        scored_results.append(
            (transaction, score)
        )

    scored_results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_results[:top_k]


# --------------------------------------------------
# 7. Give retrieved information to the LLM
# --------------------------------------------------

def generate_answer(
    question: str,
    retrieved_transactions: List[Tuple[Dict, float]],
) -> str:

    context = "\n".join(
        transaction_to_text(transaction)
        for transaction, _ in retrieved_transactions
    )

    prompt = f"""
You are FinShield, a financial information assistant.

Answer the user's question using ONLY the transaction
information provided below.

Do not invent transactions or financial information.

If the provided information is not enough to answer the
question, clearly say that the available data is insufficient.

User question:
{question}

Retrieved transaction information:
{context}

Give a clear and simple answer.
"""

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
        store=False,
    )

    return interaction.output_text


# --------------------------------------------------
# 8. Test the complete RAG pipeline
# --------------------------------------------------

if __name__ == "__main__":

    pdf_path = "data/sample_statement.pdf"

    # Read PDF
    text = extract_text_from_pdf(pdf_path)

    # Convert PDF text into structured transactions
    transactions = parse_transactions(text)

    print("----- TRANSACTIONS -----")

    for transaction in transactions:
        print(transaction)

    print("\nCreating embeddings...")

    embeddings = create_document_embeddings(
        transactions
    )

    print("Embeddings created.")

    question = "Which transactions are related to food?"

    retrieved = retrieve_transactions(
        transactions,
        embeddings,
        question,
        top_k=3,
    )

    print("\n----- RETRIEVED TRANSACTIONS -----")

    for transaction, score in retrieved:
        print(
            transaction,
            f"| similarity = {score:.4f}"
        )

    print("\n----- AI ANSWER -----")

    answer = generate_answer(
        question,
        retrieved,
    )

    print(answer)