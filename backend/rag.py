import os
import json
import os
import urllib.request
import urllib.error
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.pdf_reader import extract_text_from_pdf
from backend.transaction_parser import parse_transactions
from backend.anomaly_engine import evaluate_statement


# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 2. Category enrichment
# --------------------------------------------------

CATEGORY_KEYWORDS = {
    "Food": [
        "swiggy",
        "zomato",
        "restaurant",
        "cafe",
        "food",
    ],
    "Transport": [
        "uber",
        "ola",
        "rapido",
        "metro",
        "fuel",
        "petrol",
    ],
    "Shopping": [
        "amazon",
        "flipkart",
        "myntra",
        "shopping",
    ],
    "Groceries": [
        "grocery",
        "supermarket",
        "bigbasket",
        "dmart",
    ],
    "Bills": [
        "electricity",
        "recharge",
        "broadband",
        "insurance",
    ],
}


def categorize_merchant(merchant: str) -> str:
    merchant_lower = merchant.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in merchant_lower for keyword in keywords):
            return category

    return "Other"


def enrich_transactions(
    transactions: List[Dict],
) -> List[Dict]:
    """
    Adds a stable transaction ID and category.
    Does not mutate the original transactions.
    """

    enriched = []

    for i, transaction in enumerate(
        transactions,
        start=1,
    ):
        enriched.append(
            {
                **transaction,
                "id": f"TXN{i:03d}",
                "category": categorize_merchant(
                    transaction["merchant"]
                ),
            }
        )

    return enriched


# --------------------------------------------------
# 3. Transaction -> searchable text
# --------------------------------------------------

def transaction_to_text(
    transaction: Dict,
) -> str:

    category = transaction.get(
        "category",
        "Uncategorized",
    )

    return (
        f"Transaction ID: {transaction.get('id', '?')}. "
        f"Date: {transaction['date']}. "
        f"Merchant: {transaction['merchant']}. "
        f"Amount: ₹{transaction['amount']:.2f}. "
        f"Category: {category}."
    )


# --------------------------------------------------
# 4. Create transaction embeddings
# --------------------------------------------------

def create_document_embeddings(
    transactions: List[Dict],
) -> List[List[float]]:

    texts = [
        transaction_to_text(transaction)
        for transaction in transactions
    ]

    if not texts:
        return []

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )

    return [
        embedding.values
        for embedding in result.embeddings
    ]


# --------------------------------------------------
# 5. Create question embedding
# --------------------------------------------------

def create_query_embedding(
    question: str,
) -> List[float]:

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    return result.embeddings[0].values


# --------------------------------------------------
# 6. Cosine similarity
# --------------------------------------------------

def cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:

    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# --------------------------------------------------
# 7. Semantic retrieval
# --------------------------------------------------

def retrieve_transactions(
    transactions: List[Dict],
    embeddings: List[List[float]],
    question: str,
    top_k: int = 3,
) -> List[Tuple[Dict, float]]:

    if not transactions or not embeddings:
        return []

    query_embedding = create_query_embedding(
        question
    )

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
# 8. Intent detection
# --------------------------------------------------

def detect_question_intents(question: str) -> List[str]:
    """
    Detect one or more intents from the user's question.

    More specific intents are detected before the broad
    overall-spending intent so questions such as
    "How much did I spend on food?" do not also become
    an overall-total question.

    Returns a list containing one or more of:
        total
        largest
        count
        category_total
        unknown_payment
        unusual_transactions
        financial_insight
        general
    """

    q = question.lower().strip()
    intents: List[str] = []

    # --------------------------------------------------
    # Unknown / unrecognized payment
    # --------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "unknown payment",
            "unknown transaction",
            "unknown merchant",
            "unrecognized payment",
            "unrecognized transaction",
            "unfamiliar payment",
            "unfamiliar transaction",
            "who is this merchant",
        ]
    ):
        intents.append("unknown_payment")

    # --------------------------------------------------
    # Unusual / anomalous transactions
    # --------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "unusual transaction",
            "unusual transactions",
            "unusual payment",
            "unusual payments",
            "suspicious transaction",
            "suspicious transactions",
            "anything suspicious",
            "anomaly",
            "anomalies",
            "flagged transaction",
            "flagged transactions",
            "was flagged",
            "were flagged",
            "transaction flagged",
            "weird transaction",
            "anything look unusual",
            "look unusual",
            "unusual",
        ]
    ):
        intents.append("unusual_transactions")

    # --------------------------------------------------
    # Largest transaction
    # --------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "largest transaction",
            "biggest transaction",
            "highest transaction",
            "most expensive transaction",
            "largest expense",
            "biggest expense",
            "highest expense",
        ]
    ):
        intents.append("largest")

    # --------------------------------------------------
    # Transaction count
    # --------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "how many transactions",
            "number of transactions",
            "transaction count",
            "how many purchases",
        ]
    ):
        intents.append("count")

    # --------------------------------------------------
    # Category spending
    # --------------------------------------------------

    # Use both phrase matching and the detected category so that
    # natural/typo-prone questions such as
    # "how much amounti spend on my food" still work.
    detected_category = detect_category(question)
    spending_words = (
        "spend",
        "spent",
        "spending",
        "amount",
        "cost",
    )

    has_spending_word = any(
        word in q
        for word in spending_words
    )

    category_spending_match = any(
        phrase in q
        for phrase in [
            "how much did i spend on",
            "how much have i spent on",
            "how much money did i spend on",
            "how much spend on",
            "how much on",
            "spending on",
            "spent on",
            "spend on",
            "amount spent on",
            "amount i spent on",
        ]
    )

    if category_spending_match or (
        detected_category is not None
        and has_spending_word
    ):
        intents.append("category_total")

    # --------------------------------------------------
    # Overall spending
    # --------------------------------------------------
    # Do not classify a category-specific question as a general total.

    if "category_total" not in intents and any(
        phrase in q
        for phrase in [
            "total spending",
            "total spend",
            "how much did i spend",
            "how much have i spent",
            "total amount spent",
            "overall spending",
            "overall spend",
        ]
    ):
        intents.append("total")

    # --------------------------------------------------
    # Broad financial insight
    # --------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "analyze my spending",
            "analyse my spending",
            "analyze this statement",
            "analyse this statement",
            "what can you tell me about my spending",
            "what patterns do you notice",
            "what patterns do you see",
            "what spending patterns",
            "spending overview",
            "overview of my spending",
            "how concentrated is my spending",
            "what stands out",
            "something useful about my spending",
        ]
    ):
        intents.append("financial_insight")

    # --------------------------------------------------
    # No known intent -> general RAG
    # --------------------------------------------------

    if not intents:
        intents.append("general")

    return intents

# --------------------------------------------------
# 9. Detect category
# --------------------------------------------------

def detect_category(
    question: str,
) -> Optional[str]:

    q = question.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():

        if category.lower() in q:
            return category

        if any(
            keyword in q
            for keyword in keywords
        ):
            return category

    return None


# --------------------------------------------------
# 10. Exact Python calculations
# --------------------------------------------------

def calculate_total(
    transactions: List[Dict],
) -> float:

    return sum(
        float(transaction["amount"])
        for transaction in transactions
    )


def find_largest_transaction(
    transactions: List[Dict],
) -> Optional[Dict]:

    if not transactions:
        return None

    return max(
        transactions,
        key=lambda transaction: float(
            transaction["amount"]
        ),
    )


def count_transactions(
    transactions: List[Dict],
) -> int:

    return len(transactions)


def filter_by_category(
    transactions: List[Dict],
    category: str,
) -> List[Dict]:

    return [
        transaction
        for transaction in transactions
        if transaction.get("category") == category
    ]


def find_unknown_payments(
    transactions: List[Dict],
) -> List[Dict]:
    """
    Finds merchants explicitly named as unknown.
    """

    return [
        transaction
        for transaction in transactions
        if "unknown"
        in transaction["merchant"].lower()
    ]


def calculate_financial_insight(
    transactions: List[Dict],
    anomalies: List[Dict],
) -> Dict:
    """Calculate lightweight, statement-scoped financial insight metrics."""
    if not transactions:
        return {
            "transaction_count": 0,
            "total_spending": 0.0,
            "average_transaction": 0.0,
            "potentially_unusual_count": 0,
        }

    amounts = [float(transaction["amount"]) for transaction in transactions]
    total_spending = sum(amounts)
    largest_transactions = sorted(
        transactions,
        key=lambda transaction: float(transaction["amount"]),
        reverse=True,
    )
    largest = largest_transactions[0]

    result = {
        "transaction_count": len(transactions),
        "total_spending": round(total_spending, 2),
        "average_transaction": round(total_spending / len(transactions), 2),
        "largest_transaction": {
            "id": largest["id"],
            "date": largest["date"],
            "merchant": largest["merchant"],
            "amount": round(float(largest["amount"]), 2),
            "category": largest.get("category", "Other"),
        },
        "potentially_unusual_count": sum(
            transaction.get("severity") in {"MEDIUM", "HIGH", "CRITICAL"}
            for transaction in anomalies
        ),
    }

    if total_spending > 0:
        result["largest_transaction_percentage"] = round(
            float(largest["amount"]) / total_spending * 100,
            2,
        )

    if len(largest_transactions) > 1:
        second_largest = largest_transactions[1]
        result["second_largest_transaction"] = {
            "id": second_largest["id"],
            "date": second_largest["date"],
            "merchant": second_largest["merchant"],
            "amount": round(float(second_largest["amount"]), 2),
            "category": second_largest.get("category", "Other"),
        }

    category_totals: Dict[str, float] = {}
    for transaction in transactions:
        category = transaction.get("category", "Other")
        category_totals[category] = category_totals.get(category, 0.0) + float(
            transaction["amount"]
        )

    if category_totals:
        top_category, top_category_amount = max(
            category_totals.items(),
            key=lambda item: item[1],
        )
        result["top_category"] = {
            "category": top_category,
            "amount": round(top_category_amount, 2),
            "percentage": round(
                top_category_amount / total_spending * 100,
                2,
            ) if total_spending > 0 else 0.0,
        }

    merchant_counts: Dict[str, int] = {}
    for transaction in transactions:
        merchant = transaction["merchant"].strip()
        merchant_counts[merchant] = merchant_counts.get(merchant, 0) + 1

    repeated_merchants = [
        {"merchant": merchant, "transaction_count": count}
        for merchant, count in merchant_counts.items()
        if count > 1
    ]
    if repeated_merchants:
        result["repeated_merchants"] = repeated_merchants

    return result


def financial_insight_result(
    question: str,
    transactions: List[Dict],
    anomalies: List[Dict],
) -> Dict:
    """Build an exact insight payload and send it through the existing LLM layer."""
    metrics = calculate_financial_insight(transactions, anomalies)
    exact_lines = [
        f"Total spending: ₹{metrics['total_spending']:.2f}",
        f"Transaction count: {metrics['transaction_count']}",
        f"Average transaction: ₹{metrics['average_transaction']:.2f}",
    ]

    largest = metrics.get("largest_transaction")
    if largest:
        exact_lines.append(
            f"Largest transaction: {largest['merchant']} — "
            f"₹{largest['amount']:.2f} on {largest['date']}"
        )
    if "largest_transaction_percentage" in metrics:
        exact_lines.append(
            "Largest transaction percentage of total spending: "
            f"{metrics['largest_transaction_percentage']:.2f}%"
        )

    top_category = metrics.get("top_category")
    if top_category:
        exact_lines.append(
            f"Top spending category: {top_category['category']} — "
            f"₹{top_category['amount']:.2f} "
            f"({top_category['percentage']:.2f}% of total spending)"
        )

    exact_lines.append(
        "Potentially unusual transactions: "
        f"{metrics['potentially_unusual_count']}"
    )

    repeated_merchants = metrics.get("repeated_merchants", [])
    if repeated_merchants:
        exact_lines.append(
            "Repeated merchants: "
            + ", ".join(
                f"{item['merchant']} ({item['transaction_count']} transactions)"
                for item in repeated_merchants
            )
        )

    ranked_transactions = sorted(
        transactions,
        key=lambda transaction: float(transaction["amount"]),
        reverse=True,
    )[:3]
    answer_text = generate_answer(
        question,
        [(transaction, 1.0) for transaction in ranked_transactions],
        exact_result="\n".join(exact_lines),
        financial_insight=True,
        financial_metrics=metrics,
    )

    return {
        "question": question,
        "intent": "financial_insight",
        "scope": "active_statement",
        "intents": ["financial_insight"],
        "answer": answer_text,
        "computed_results": metrics,
        "transactions_used": [
            {
                "id": transaction["id"],
                "date": transaction["date"],
                "merchant": transaction["merchant"],
                "amount": transaction["amount"],
                "category": transaction.get("category", "Other"),
            }
            for transaction in ranked_transactions
        ],
    }


# --------------------------------------------------
# 11. OpenRouter explanation
# --------------------------------------------------

def build_financial_insight_fallback(metrics: Dict) -> str:
    """Render a readable insight from metrics calculated by Python."""
    transaction_count = metrics.get("transaction_count", 0)
    total_spending = metrics.get("total_spending", 0.0)
    average_transaction = metrics.get("average_transaction", 0.0)

    lines = [
        "Spending Overview",
        "",
        f"You spent ₹{total_spending:,.2f} across "
        f"{transaction_count} transactions, with an average transaction of "
        f"₹{average_transaction:,.2f}.",
    ]

    top_category = metrics.get("top_category")
    if top_category:
        lines.extend([
            "",
            "Key pattern:",
            f"The {top_category['category']} category accounts for "
            f"₹{top_category['amount']:,.2f}, or "
            f"{top_category['percentage']:.2f}% of total spending.",
        ])

    largest = metrics.get("largest_transaction")
    if largest:
        largest_line = (
            f"Your largest transaction was ₹{largest['amount']:,.2f} to "
            f"{largest['merchant']}"
        )
        largest_percentage = metrics.get("largest_transaction_percentage")
        if largest_percentage is not None:
            largest_line += (
                f", representing {largest_percentage:.2f}% of total spending."
            )
        else:
            largest_line += "."
        lines.extend(["", "Largest payment:", largest_line])

    repeated_merchants = metrics.get("repeated_merchants", [])
    if repeated_merchants:
        merchant_summary = ", ".join(
            f"{item['merchant']} appears in {item['transaction_count']} "
            "transactions"
            for item in repeated_merchants
        )
        lines.extend(["", "Merchant pattern:", merchant_summary + "."])

    unusual_count = metrics.get("potentially_unusual_count", 0)
    if unusual_count == 1:
        review_line = (
            "1 transaction was identified as potentially unusual and may be "
            "worth reviewing."
        )
    else:
        review_line = (
            f"{unusual_count} transactions were identified as potentially "
            "unusual and may be worth reviewing."
        )
    lines.extend(["", "Review signal:", review_line])

    return "\n".join(lines)


def _format_anomaly_date(value: object) -> str:
    date_text = str(value)
    for date_format in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(date_text, date_format).strftime("%d %b %Y")
        except ValueError:
            continue
    return date_text


def _format_anomaly_signal(signal: object) -> str:
    return str(signal).replace("_", " ").capitalize()


def build_anomaly_fallback(
    transactions: List[Dict],
    detailed: bool = False,
) -> str:
    """Render anomaly evidence without adding analysis beyond stored fields."""
    if detailed and len(transactions) == 1:
        transaction = transactions[0]
        lines = [
            "Potentially unusual transaction",
            "",
            f"₹{float(transaction['amount']):,.2f} at "
            f"{transaction['merchant']} on "
            f"{_format_anomaly_date(transaction['date'])}",
            "",
            f"Risk: {transaction['severity']}",
            f"Risk score: {transaction['risk_score']}",
            f"Signals: {', '.join(_format_anomaly_signal(signal) for signal in transaction.get('signals', []))}",
            "",
            "Why it was flagged:",
        ]
        lines.extend(
            f"• {reason}"
            for reason in transaction.get("reasons", [])
        )
        lines.extend(["", "Review " "recommended: verify that the transaction was expected."])
        return "\n".join(lines)

    count = len(transactions)
    lines = [
        f"Potentially unusual activity was identified "
        f"in {count} "
        f"transaction{'s' if count != 1 else ''}.",
    ]

    for transaction in transactions:
        lines.extend([
            "",
            f"• {transaction['merchant']} — ₹{float(transaction['amount']):,.2f} "
            f"on {_format_anomaly_date(transaction['date'])}",
            f"  Risk: {transaction['severity']} (score {transaction['risk_score']})",
            "  Signals: "
            + ", ".join(
                _format_anomaly_signal(signal)
                for signal in transaction.get("signals", [])
            ),
            "  Reasons:",
        ])
        lines.extend(
            f"  • {reason}"
            for reason in transaction.get("reasons", [])
        )

    lines.extend(["", "Review " "recommended: verify that the transaction was expected."])
    return "\n".join(lines)

def generate_answer(
    question: str,
    retrieved_transactions: List[Tuple[Dict, float]],
    computed_total: Optional[float] = None,
    exact_result: Optional[str] = None,
    financial_insight: bool = False,
    financial_metrics: Optional[Dict] = None,
) -> str:

    fallback_answer = (
        build_financial_insight_fallback(financial_metrics)
        if financial_insight and financial_metrics is not None
        else exact_result
    ) or (
        "FinShield's AI explanation service is temporarily unavailable. "
        "The statement data and deterministic financial analysis are still "
        "available."
    )

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        if fallback_answer:
            return fallback_answer
        raise RuntimeError("OpenRouter is unavailable.")

    if retrieved_transactions:
        context = "\n".join(
            f"{transaction.get('id', '?')}: "
            f"{transaction_to_text(transaction)}"
            for transaction, _ in retrieved_transactions
        )
    else:
        context = "No relevant transactions were retrieved."

    calculation_instruction = ""

    if computed_total is not None:
        calculation_instruction += f"""
The exact amount calculated by Python is ₹{computed_total:.2f}.
Use this number exactly.
Do not recalculate it.
"""

    if exact_result is not None:
        calculation_instruction += f"""
The exact result calculated by Python is:

{exact_result}

Use this result exactly.
"""

    prompt = f"""
You are FinShield, a financial information assistant.

Answer the user's question using ONLY the financial
information provided below.

Python performs exact financial calculations.
Do not invent numbers.
Do not perform arithmetic yourself when an exact
result has already been provided.

{calculation_instruction}

User question:
{question}

Relevant transaction information:
{context}

Give a clear and simple answer.

If the available information is insufficient,
say so clearly.

Never claim that a transaction is definitely
fraudulent. Use wording such as
"potentially unusual" when discussing anomalies.
"""

    if financial_insight:
        prompt += """

For this financial insight response, explain all supplied patterns
concisely in roughly 150-300 words. Output only the final answer; do not
include analysis, planning, or a description of your reasoning. Finish each
thought in a complete sentence and do not add unsupported financial values.
"""

    payload = {
        "models": [
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "inclusionai/ling-3.0-flash-fin:free",
            "poolside/laguna-s-2.1:free"
        ],
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
        "max_tokens": 450 if financial_insight else 300,
    }

    if financial_insight:
        payload["reasoning"] = {
            "effort": "none",
            "exclude": True,
        }

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as e:
        # The financial calculation is already authoritative.
        # If the LLM is unavailable, return the exact Python result
        # instead of breaking the entire /ask feature.
        if fallback_answer:
            return fallback_answer

        raise RuntimeError("OpenRouter is unavailable.") from e

    except urllib.error.URLError as e:
        if fallback_answer:
            return fallback_answer

        raise RuntimeError("OpenRouter is unavailable.") from e

    except (json.JSONDecodeError, TimeoutError, OSError) as e:
        if fallback_answer:
            return fallback_answer

        raise RuntimeError("OpenRouter is unavailable.") from e

    choices = response_data.get("choices", [])

    if not choices:
        if fallback_answer:
            return fallback_answer
        raise RuntimeError("OpenRouter returned no usable answer.")

    answer = choices[0].get("message", {}).get("content")

    if not answer:
        if fallback_answer:
            return fallback_answer
        raise RuntimeError("OpenRouter returned no usable answer.")

    return answer.strip()


# --------------------------------------------------
# 12. StatementSession
# --------------------------------------------------

class StatementSession:
    """
    Represents one uploaded bank statement.

    Created ONCE after upload.

    Stores:
        enriched transactions
        embeddings
        anomaly results

    Questions reuse all three.
    """

    def __init__(
        self,
        transactions: List[Dict],
    ):

        # Enrich once.
        self.transactions = enrich_transactions(
            transactions
        )

        # Create embeddings once.
        self.embeddings = (
            create_document_embeddings(
                self.transactions
            )
        )

        # Run anomaly engine once.
        #
        # evaluate_statement() returns the same
        # transaction dictionaries with:
        # risk_score, severity, signals, reasons
        self.anomalies = evaluate_statement(
            self.transactions
        )

    # --------------------------------------------------
    # Ask FinShield
    # --------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> Dict:
        """
        Answer a FinShield question.

        A question can contain more than one supported intent.
        Deterministic financial operations are handled by Python;
        general open-ended questions use semantic RAG.
        """

        intents = detect_question_intents(question)

        # --------------------------------------------------
        # CASE A — Multiple intents in one question
        # --------------------------------------------------

        if len(intents) > 1:
            computed_results: Dict = {}
            exact_lines: List[str] = []
            selected_transactions: List[Dict] = []

            if "total" in intents:
                total = calculate_total(self.transactions)
                computed_results["total_spending"] = round(total, 2)
                exact_lines.append(
                    f"Total spending: ₹{total:.2f}"
                )

            if "largest" in intents:
                largest = find_largest_transaction(self.transactions)

                if largest:
                    computed_results["largest_transaction"] = {
                        "id": largest["id"],
                        "date": largest["date"],
                        "merchant": largest["merchant"],
                        "amount": largest["amount"],
                        "category": largest["category"],
                    }
                    exact_lines.append(
                        f"Largest transaction: {largest['merchant']} — "
                        f"₹{largest['amount']:.2f} on {largest['date']}"
                    )
                    selected_transactions.append(largest)

            if "count" in intents:
                count = count_transactions(self.transactions)
                computed_results["transaction_count"] = count
                exact_lines.append(
                    f"Transaction count: {count}"
                )

            if "category_total" in intents:
                category = detect_category(question)

                if category:
                    category_transactions = filter_by_category(
                        self.transactions, category
                    )
                    category_total = calculate_total(category_transactions)

                    computed_results["category_total"] = {
                        "category": category,
                        "amount": round(category_total, 2),
                    }

                    exact_lines.append(
                        f"Total {category} spending: ₹{category_total:.2f}"
                    )
                    selected_transactions.extend(category_transactions)
                else:
                    exact_lines.append(
                        "No spending category could be identified from the question."
                    )

            if "unknown_payment" in intents:
                unknown = find_unknown_payments(self.transactions)
                computed_results["unknown_payment_count"] = len(unknown)

                if unknown:
                    exact_lines.append(
                        f"Found {len(unknown)} potentially unknown payment(s)."
                    )
                    selected_transactions.extend(unknown)
                else:
                    exact_lines.append(
                        "No payments explicitly marked as unknown were found."
                    )

            if "unusual_transactions" in intents:
                flagged = [
                    transaction
                    for transaction in self.anomalies
                    if transaction["severity"] in {
                        "MEDIUM",
                        "HIGH",
                        "CRITICAL",
                    }
                ]

                computed_results["potentially_unusual_count"] = len(flagged)

                if flagged:
                    exact_lines.append(
                        f"Found {len(flagged)} potentially unusual transaction(s)."
                    )
                    exact_lines.append(build_anomaly_fallback(flagged))
                    selected_transactions.extend(flagged)
                else:
                    exact_lines.append(
                        "No potentially unusual transactions were detected."
                    )

            if "financial_insight" in intents:
                insight_metrics = calculate_financial_insight(
                    self.transactions,
                    self.anomalies,
                )
                computed_results.update(insight_metrics)
                exact_lines.extend([
                    f"Total spending: ₹{insight_metrics['total_spending']:.2f}",
                    f"Average transaction: ₹{insight_metrics['average_transaction']:.2f}",
                    "Potentially unusual transactions: "
                    f"{insight_metrics['potentially_unusual_count']}",
                ])
                insight_largest = insight_metrics.get("largest_transaction")
                if insight_largest:
                    exact_lines.append(
                        f"Largest transaction: {insight_largest['merchant']} — "
                        f"₹{insight_largest['amount']:.2f}"
                    )
                selected_transactions.extend(
                    sorted(
                        self.transactions,
                        key=lambda transaction: float(transaction["amount"]),
                        reverse=True,
                    )[:3]
                )

            # Remove duplicate transactions while preserving insertion order.
            unique_transactions = {}
            for transaction in selected_transactions:
                unique_transactions[transaction["id"]] = transaction

            selected_transactions = list(unique_transactions.values())

            exact_result = "\n".join(exact_lines)

            answer_text = generate_answer(
                question,
                [(transaction, 1.0) for transaction in selected_transactions],
                exact_result=exact_result,
                financial_insight="financial_insight" in intents,
                financial_metrics=insight_metrics
                if "financial_insight" in intents
                else None,
            )

            return {
                "question": question,
                "intent": "multiple",
                "scope": "active_statement",
                "intents": intents,
                "answer": answer_text,
                "computed_results": computed_results,
                "transactions_used": [
                    {
                        "id": transaction["id"],
                        "date": transaction["date"],
                        "merchant": transaction["merchant"],
                        "amount": transaction["amount"],
                        "category": transaction["category"],
                    }
                    for transaction in selected_transactions
                ],
            }

        # --------------------------------------------------
        # Single intent
        # --------------------------------------------------

        intent = intents[0]

        # --------------------------------------------------
        # CASE 0 — Financial insight
        # --------------------------------------------------

        if intent == "financial_insight":
            return financial_insight_result(
                question,
                self.transactions,
                self.anomalies,
            )

        # --------------------------------------------------
        # CASE 1 — Overall total
        # --------------------------------------------------

        if intent == "total":

            total = calculate_total(self.transactions)

            exact_result = f"Total spending: ₹{total:.2f}"

            answer_text = generate_answer(
                question,
                [],
                computed_total=total,
                exact_result=exact_result,
            )

            return {
                "question": question,
                "intent": intent,
                "answer": answer_text,
                "computed_total": round(total, 2),
                "transactions_used": [],
            }

        # --------------------------------------------------
        # CASE 2 — Largest transaction
        # --------------------------------------------------

        if intent == "largest":

            largest = find_largest_transaction(self.transactions)

            if not largest:
                return {
                    "question": question,
                    "intent": intent,
                    "answer": "No transactions are available.",
                    "computed_total": None,
                    "transactions_used": [],
                }

            exact_result = (
                f"Largest transaction: {largest['merchant']} — "
                f"₹{largest['amount']:.2f} on {largest['date']}"
            )

            answer_text = generate_answer(
                question,
                [(largest, 1.0)],
                computed_total=float(largest["amount"]),
                exact_result=exact_result,
            )

            return {
                "question": question,
                "intent": intent,
                "answer": answer_text,
                "computed_total": round(float(largest["amount"]), 2),
                "transactions_used": [
                    {
                        "id": largest["id"],
                        "date": largest["date"],
                        "merchant": largest["merchant"],
                        "amount": largest["amount"],
                        "category": largest["category"],
                    }
                ],
            }

        # --------------------------------------------------
        # CASE 3 — Transaction count
        # --------------------------------------------------

        if intent == "count":

            count = count_transactions(self.transactions)
            exact_result = f"Transaction count: {count}"

            answer_text = generate_answer(
                question,
                [],
                exact_result=exact_result,
            )

            return {
                "question": question,
                "intent": intent,
                "answer": answer_text,
                "computed_total": None,
                "transaction_count": count,
                "transactions_used": [],
            }

        # --------------------------------------------------
        # CASE 4 — Category total
        # --------------------------------------------------

        if intent == "category_total":

            category = detect_category(question)

            if not category:
                return {
                    "question": question,
                    "intent": intent,
                    "answer": (
                        "I couldn't identify a spending category in your question."
                    ),
                    "computed_total": None,
                    "transactions_used": [],
                }

            category_transactions = filter_by_category(
                self.transactions, category
            )

            if not category_transactions:
                return {
                    "question": question,
                    "intent": intent,
                    "category": category,
                    "answer": (
                        f"No {category.lower()} transactions were found in the statement."
                    ),
                    "computed_total": 0.0,
                    "transactions_used": [],
                }

            category_total = calculate_total(category_transactions)
            exact_result = (
                f"Total {category} spending: ₹{category_total:.2f}"
            )

            answer_text = generate_answer(
                question,
                [(transaction, 1.0) for transaction in category_transactions],
                computed_total=category_total,
                exact_result=exact_result,
            )

            return {
                "question": question,
                "intent": intent,
                "category": category,
                "answer": answer_text,
                "computed_total": round(category_total, 2),
                "transactions_used": [
                    {
                        "id": transaction["id"],
                        "date": transaction["date"],
                        "merchant": transaction["merchant"],
                        "amount": transaction["amount"],
                        "category": transaction["category"],
                    }
                    for transaction in category_transactions
                ],
            }

        # --------------------------------------------------
        # CASE 5 — Unknown payment
        # --------------------------------------------------

        if intent == "unknown_payment":

            unknown = find_unknown_payments(self.transactions)

            if not unknown:
                return {
                    "question": question,
                    "intent": intent,
                    "answer": (
                        "I didn't find any payments to merchants explicitly identified as unknown."
                    ),
                    "computed_total": 0.0,
                    "transactions_used": [],
                }

            total = calculate_total(unknown)
            exact_result = (
                "Potentially unknown payments:\n"
                + "\n".join(
                    f"{transaction['id']} — {transaction['merchant']} — "
                    f"₹{transaction['amount']:.2f} on {transaction['date']}"
                    for transaction in unknown
                )
            )

            answer_text = generate_answer(
                question,
                [(transaction, 1.0) for transaction in unknown],
                computed_total=total,
                exact_result=exact_result,
            )

            return {
                "question": question,
                "intent": intent,
                "answer": answer_text,
                "computed_total": round(total, 2),
                "transactions_used": [
                    {
                        "id": transaction["id"],
                        "date": transaction["date"],
                        "merchant": transaction["merchant"],
                        "amount": transaction["amount"],
                        "category": transaction["category"],
                    }
                    for transaction in unknown
                ],
            }

        # --------------------------------------------------
        # CASE 6 — Unusual transactions
        # --------------------------------------------------

        if intent == "unusual_transactions":

            # The main user-facing anomaly list shows MEDIUM or higher.
            # LOW signals remain available in self.anomalies for details.
            flagged = [
                transaction
                for transaction in self.anomalies
                if transaction["severity"] in {
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL",
                }
            ]

            if not flagged:
                return {
                    "question": question,
                    "intent": intent,
                    "answer": (
                        "No potentially unusual transactions were identified "
                        "in this statement based on the current risk signals."
                    ),
                    "computed_total": None,
                    "transactions_used": [],
                }

            matching_flagged = [
                transaction
                for transaction in flagged
                if transaction.get("merchant", "").lower() in question.lower()
            ]
            detail_requested = any(
                phrase in question.lower()
                for phrase in ("why", "flagged")
            )
            explanation_transactions = (
                matching_flagged
                if len(matching_flagged) == 1
                else flagged
            )

            exact_result = (
                build_anomaly_fallback(
                    explanation_transactions,
                    detailed=detail_requested and len(explanation_transactions) == 1,
                )
            )

            answer_text = generate_answer(
                question,
                [(transaction, 1.0) for transaction in flagged],
                exact_result=exact_result,
            )

            return {
                "question": question,
                "intent": intent,
                "answer": answer_text,
                "computed_total": None,
                "transactions_used": explanation_transactions,
            }

        # --------------------------------------------------
        # CASE 7 — General RAG question
        # --------------------------------------------------

        retrieved = retrieve_transactions(
            self.transactions,
            self.embeddings,
            question,
            top_k=top_k,
        )

        answer_text = generate_answer(
            question,
            retrieved,
        )

        return {
            "question": question,
            "intent": intent,
            "answer": answer_text,
            "computed_total": None,
            "transactions_used": [
                {
                    "id": transaction["id"],
                    "date": transaction["date"],
                    "merchant": transaction["merchant"],
                    "amount": transaction["amount"],
                    "category": transaction["category"],
                    "similarity": round(score, 4),
                }
                for transaction, score in retrieved
            ],
        }


# --------------------------------------------------
# 13. Test complete RAG pipeline
# --------------------------------------------------

if __name__ == "__main__":

    pdf_path = "data/sample_statement.pdf"

    # Read PDF
    text = extract_text_from_pdf(
        pdf_path
    )

    # Parse transactions
    transactions = parse_transactions(
        text
    )

    print("----- TRANSACTIONS -----")

    for transaction in transactions:
        print(transaction)

    print(
        "\nStarting statement session "
        "(creating embeddings and anomaly "
        "results once)..."
    )

    session = StatementSession(
        transactions
    )

    print(
        "\nEnriched transactions:"
    )

    for transaction in session.transactions:

        print(
            f"  {transaction['id']}  "
            f"{transaction['category']:<10}  "
            f"{transaction['merchant']}  "
            f"₹{transaction['amount']:.2f}"
        )

    # --------------------------------------------------
    # Test 1 — Category
    # --------------------------------------------------

    question = (
        "How much did I spend on food?"
    )

    print(
        f"\n----- ASKING: {question} -----"
    )

    result = session.ask(question)

    print("\n----- AI ANSWER -----")
    print(result["answer"])

    print(
        f"\nComputed total (Python): "
        f"₹{result['computed_total']:.2f}"
    )

    print("\n----- TRANSACTIONS USED -----")

    for transaction in result[
        "transactions_used"
    ]:

        print(
            f"  {transaction['id']} — "
            f"{transaction['merchant']} — "
            f"₹{transaction['amount']:.2f}"
        )

    # --------------------------------------------------
    # Test 2 — Compound question
    # --------------------------------------------------

    compound_question = (
        "How much amounti spend on my food and what is the largest transaction?"
    )

    print(
        f"\n----- ASKING: {compound_question} -----"
    )

    compound_result = session.ask(
        compound_question
    )

    print("\n----- AI ANSWER -----")
    print(compound_result["answer"])

    print(
        f"Detected intents: {compound_result.get('intents')}"
    )

    print(
        f"Computed results: {compound_result.get('computed_results')}"
    )

    # --------------------------------------------------
    # Test 3 — Largest
    # --------------------------------------------------

    second_question = (
        "What was my largest transaction?"
    )

    print(
        f"\n----- ASKING: "
        f"{second_question} -----"
    )

    result2 = session.ask(
        second_question
    )

    print("\n----- AI ANSWER -----")
    print(result2["answer"])

    # --------------------------------------------------
    # Test 4 — Count
    # --------------------------------------------------

    third_question = (
        "How many transactions did I make?"
    )

    print(
        f"\n----- ASKING: "
        f"{third_question} -----"
    )

    result3 = session.ask(
        third_question
    )

    print("\n----- AI ANSWER -----")
    print(result3["answer"])

    print(
        f"Transaction count: "
        f"{result3['transaction_count']}"
    )

    # --------------------------------------------------
    # Test 5 — Unknown payment
    # --------------------------------------------------

    fourth_question = (
        "Is there any unknown payment?"
    )

    print(
        f"\n----- ASKING: "
        f"{fourth_question} -----"
    )

    result4 = session.ask(
        fourth_question
    )

    print("\n----- AI ANSWER -----")
    print(result4["answer"])

    print(
        "\n----- UNKNOWN PAYMENTS -----"
    )

    for transaction in result4[
        "transactions_used"
    ]:

        print(
            f"  {transaction['id']} — "
            f"{transaction['merchant']} — "
            f"₹{transaction['amount']:.2f}"
        )

    # --------------------------------------------------
    # Test 6 — Unusual transactions
    # --------------------------------------------------

    fifth_question = (
        "Show me unusual transactions"
    )

    print(
        f"\n----- ASKING: "
        f"{fifth_question} -----"
    )

    result5 = session.ask(
        fifth_question
    )

    print("\n----- AI ANSWER -----")
    print(result5["answer"])

    print(
        "\n----- UNUSUAL TRANSACTIONS -----"
    )

    for transaction in result5[
        "transactions_used"
    ]:

        print(
            f"  {transaction['id']} — "
            f"{transaction['merchant']} — "
            f"₹{transaction['amount']:.2f} — "
            f"severity={transaction['severity']} — "
            f"score={transaction['risk_score']}"
        )
    # --------------------------------------------------
    # Test 6 — Multiple intents in one question
    # --------------------------------------------------

    sixth_question = (
        "How much did I spend on food and what was my largest transaction?"
    )

    print(
        f"\n----- ASKING: {sixth_question} -----"
    )

    result6 = session.ask(sixth_question)

    print("\n----- AI ANSWER -----")
    print(result6["answer"])

    print("\nDetected intents:")
    print(result6.get("intents", [result6.get("intent")]))

