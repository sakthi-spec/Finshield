"""
FinShield — Layered Anomaly Engine
===================================

Checks a transaction against multiple independent signals ("shields") and
combines them into a single weighted risk score. Each shield is deliberately
conservative in its language: nothing here claims a transaction IS fraud,
only that it may be worth a user's attention.

Shields implemented:
    1. Amount Shield            (+2)  — statistical outlier on transaction amount
    2. New Merchant Shield      (+1)  — merchant not seen earlier in this statement
    3. Frequency / Velocity     (+2)  — unusually many transactions on one day
    4. Duplicate Shield         (+2)  — near-identical transaction repeated in a
                                          short window (possible double charge)

Severity bands (sum of triggered shield weights):
    0        -> NONE
    1-2      -> LOW
    3-4      -> MEDIUM
    5-6      -> HIGH
    7+       -> CRITICAL
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Config — tweak thresholds / weights here without touching logic below
# ---------------------------------------------------------------------------

SHIELD_WEIGHTS = {
    "amount": 2,
    "new_merchant": 1,
    "frequency": 2,
    "duplicate": 2,
}

AMOUNT_STD_MULTIPLIER = 2.0          # flag if amount > mean + 2*std
MIN_TRANSACTIONS_FOR_STATS = 4       # need at least this many to trust mean/std
FREQUENCY_THRESHOLD_PER_DAY = 4      # more than this many txns/day is unusual
DUPLICATE_WINDOW_MINUTES = 120       # same merchant+amount within this window
DUPLICATE_AMOUNT_TOLERANCE = 0.01    # amounts within this fraction count as "same"

SEVERITY_BANDS = [
    (0, 0, "NONE"),
    (1, 2, "LOW"),
    (3, 4, "MEDIUM"),
    (5, 6, "HIGH"),
    (7, float("inf"), "CRITICAL"),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    date: str          # "YYYY-MM-DD" or full ISO timestamp
    merchant: str
    amount: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def dt(self) -> datetime:
        # Accept plain dates or full timestamps
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(self.date, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unrecognized date format: {self.date!r}")


@dataclass
class ShieldResult:
    name: str
    triggered: bool
    weight: int
    reason: str | None = None


@dataclass
class TransactionRiskReport:
    transaction: Transaction
    shields: list[ShieldResult]
    risk_score: int
    severity: str
    reasons: list[str]


# ---------------------------------------------------------------------------
# Individual shields
# ---------------------------------------------------------------------------

def amount_shield(txn: Transaction, all_txns: list[Transaction]) -> ShieldResult:
    """Flags a transaction whose amount is a statistical outlier."""
    amounts = [t.amount for t in all_txns]

    if len(amounts) < MIN_TRANSACTIONS_FOR_STATS:
        return ShieldResult("amount", False, SHIELD_WEIGHTS["amount"])

    mean = statistics.mean(amounts)
    std = statistics.pstdev(amounts) or 1e-9  # avoid div-by-zero on flat data
    upper_limit = mean + AMOUNT_STD_MULTIPLIER * std

    if txn.amount > upper_limit:
        return ShieldResult(
            "amount",
            True,
            SHIELD_WEIGHTS["amount"],
            f"Amount ₹{txn.amount:,.2f} is above the normal range "
            f"(mean ₹{mean:,.2f} + {AMOUNT_STD_MULTIPLIER}×std ≈ ₹{upper_limit:,.2f})",
        )
    return ShieldResult("amount", False, SHIELD_WEIGHTS["amount"])


def new_merchant_shield(txn: Transaction, prior_txns: list[Transaction]) -> ShieldResult:
    """Flags a merchant not seen earlier in the available history.

    Note: with only one statement uploaded, "new" really means
    "first-seen in this statement" — not "new to the user's life".
    """
    seen_merchants = {t.merchant.strip().lower() for t in prior_txns}
    is_new = txn.merchant.strip().lower() not in seen_merchants

    if is_new:
        return ShieldResult(
            "new_merchant",
            True,
            SHIELD_WEIGHTS["new_merchant"],
            f"'{txn.merchant}' is a first-seen merchant in this statement",
        )
    return ShieldResult("new_merchant", False, SHIELD_WEIGHTS["new_merchant"])


def frequency_shield(txn: Transaction, all_txns: list[Transaction]) -> ShieldResult:
    """Flags unusually high transaction volume on the same calendar day."""
    same_day = [t for t in all_txns if t.dt.date() == txn.dt.date()]
    count = len(same_day)

    if count > FREQUENCY_THRESHOLD_PER_DAY:
        return ShieldResult(
            "frequency",
            True,
            SHIELD_WEIGHTS["frequency"],
            f"{count} transactions occurred on {txn.dt.date()}, "
            f"above the {FREQUENCY_THRESHOLD_PER_DAY}/day threshold",
        )
    return ShieldResult("frequency", False, SHIELD_WEIGHTS["frequency"])


def duplicate_shield(txn: Transaction, all_txns: list[Transaction]) -> ShieldResult:
    """Flags a near-identical transaction (same merchant, ~same amount)
    occurring within a short time window — a common signature of a
    double charge or duplicate submission."""
    window = timedelta(minutes=DUPLICATE_WINDOW_MINUTES)

    for other in all_txns:
        if other is txn:
            continue
        if other.merchant.strip().lower() != txn.merchant.strip().lower():
            continue

        amount_close = (
            abs(other.amount - txn.amount) <= DUPLICATE_AMOUNT_TOLERANCE * max(txn.amount, 1)
        )
        time_close = abs((other.dt - txn.dt)) <= window

        if amount_close and time_close and other.dt <= txn.dt:
            return ShieldResult(
                "duplicate",
                True,
                SHIELD_WEIGHTS["duplicate"],
                f"Matches another '{txn.merchant}' charge of ~₹{txn.amount:,.2f} "
                f"within {DUPLICATE_WINDOW_MINUTES} minutes — possible duplicate",
            )

    return ShieldResult("duplicate", False, SHIELD_WEIGHTS["duplicate"])


SHIELD_FUNCTIONS = [amount_shield, new_merchant_shield, frequency_shield, duplicate_shield]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _severity_for(score: int) -> str:
    for low, high, label in SEVERITY_BANDS:
        if low <= score <= high:
            return label
    return "NONE"  # unreachable, but keeps type-checkers happy


def evaluate_transaction(
    txn: Transaction,
    all_txns: list[Transaction],
    index_in_statement: int,
) -> TransactionRiskReport:
    """Runs every shield against a single transaction and combines results."""
    prior_txns = all_txns[:index_in_statement]  # for "new merchant" ordering

    results: list[ShieldResult] = []
    for shield_fn in SHIELD_FUNCTIONS:
        if shield_fn is new_merchant_shield:
            results.append(shield_fn(txn, prior_txns))
        else:
            results.append(shield_fn(txn, all_txns))

    risk_score = sum(r.weight for r in results if r.triggered)
    severity = _severity_for(risk_score)
    reasons = [r.reason for r in results if r.triggered and r.reason]

    return TransactionRiskReport(
        transaction=txn,
        shields=results,
        risk_score=risk_score,
        severity=severity,
        reasons=reasons,
    )


def evaluate_statement(transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Entry point for FastAPI. Takes the list of transaction dicts produced by
    transaction_parser.py and returns each one annotated with its risk report.

    Input transaction dict shape (from transaction_parser.py):
        {"date": "2026-09-12", "merchant": "Swiggy", "amount": 450.0}
    """
    txn_objs = [Transaction(date=t["date"], merchant=t["merchant"], amount=t["amount"], raw=t)
                for t in transactions]

    # Sort by date so "new merchant" and "duplicate" checks respect chronology
    txn_objs.sort(key=lambda t: t.dt)

    annotated = []
    for i, txn in enumerate(txn_objs):
        report = evaluate_transaction(txn, txn_objs, i)
        annotated.append({
            **txn.raw,
            "risk_score": report.risk_score,
            "severity": report.severity,
            "signals": [r.name for r in report.shields if r.triggered],
            "reasons": report.reasons,
        })

    return annotated


# ---------------------------------------------------------------------------
# Quick manual test using the FinShield sample statement
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = [
        {"date": "2026-09-12", "merchant": "Swiggy", "amount": 450.0},
        {"date": "2026-09-13", "merchant": "Amazon", "amount": 2499.0},
        {"date": "2026-09-14", "merchant": "Uber", "amount": 320.0},
        {"date": "2026-09-15", "merchant": "Zomato", "amount": 510.0},
        {"date": "2026-09-16", "merchant": "Grocery Store", "amount": 1800.0},
        {"date": "2026-09-17", "merchant": "Unknown Merchant", "amount": 18500.0},
        {"date": "2026-09-18", "merchant": "Swiggy", "amount": 620.0},
    ]

    results = evaluate_statement(sample)
    for r in results:
        print(f"{r['date']}  {r['merchant']:<18} ₹{r['amount']:>9,.2f}  "
              f"score={r['risk_score']}  severity={r['severity']:<8}  "
              f"signals={r['signals']}")
        for reason in r["reasons"]:
            print(f"    - {reason}")