import os

from pymongo import MongoClient
from pymongo.server_api import ServerApi


MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set.")


client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi(
        version="1",
        strict=True,
        deprecation_errors=True,
    ),
)

db = client["finshield"]

statements_collection = db["statements"]
transactions_collection = db["transactions"]
questions_collection = db["questions"]


def test_connection() -> bool:
    """Check whether MongoDB Atlas is reachable."""
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False