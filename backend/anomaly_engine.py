from statistics import quantiles


def detect_anomalies(transactions: list[dict]) -> list[dict]:
    """
    Detect unusually large transactions using the IQR method.

    Returns a list containing only potentially unusual transactions.
    """

    if len(transactions) < 4:
        return []

    amounts = [transaction["amount"] for transaction in transactions]

    # Calculate the first and third quartiles.
    q1, _, q3 = quantiles(amounts, n=4, method="inclusive")

    # IQR = middle 50% spread of the data.
    iqr = q3 - q1

    # Anything above this is considered an outlier.
    upper_limit = q3 + (1.5 * iqr)

    anomalies = []

    for transaction in transactions:
        if transaction["amount"] > upper_limit:
            anomaly = transaction.copy()

            anomaly["reason"] = (
                "The transaction amount is significantly higher "
                "than the user's normal spending range."
            )

            anomaly["upper_limit"] = round(upper_limit, 2)

            anomalies.append(anomaly)

    return anomalies


if __name__ == "__main__":
    from transaction_parser import parse_transactions
    from pdf_reader import extract_text_from_pdf

    pdf_path = "data/sample_statement.pdf"

    text = extract_text_from_pdf(pdf_path)
    transactions = parse_transactions(text)

    anomalies = detect_anomalies(transactions)

    print("----- POTENTIALLY UNUSUAL TRANSACTIONS -----")

    if not anomalies:
        print("No unusual transactions detected.")
    else:
        for anomaly in anomalies:
            print(anomaly)