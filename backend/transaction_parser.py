import re


DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
AMOUNT_PATTERN = re.compile(r"^Rs\.\s*([\d,]+(?:\.\d{2})?)$")

# Supports transactions where date + merchant + amount are on one line.
TRANSACTION_LINE_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+(.+?)\s+Rs\.\s*([\d,]+(?:\.\d{2})?)$"
)


def parse_transactions(text: str) -> list[dict]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    transactions = []
    i = 0

    while i < len(lines):

        # --------------------------------------------------
        # Format 1:
        # date + merchant + amount on the same line
        # --------------------------------------------------
        line_match = TRANSACTION_LINE_PATTERN.match(lines[i])

        if line_match:
            date = line_match.group(1)
            merchant = line_match.group(2).strip()
            amount = float(
                line_match.group(3).replace(",", "")
            )

            transactions.append(
                {
                    "date": date,
                    "merchant": merchant,
                    "amount": amount,
                }
            )

            i += 1
            continue

        # --------------------------------------------------
        # Format 2:
        # date
        # merchant
        # amount
        # --------------------------------------------------
        if DATE_PATTERN.match(lines[i]):
            date = lines[i]

            if i + 2 < len(lines):
                merchant = lines[i + 1]
                amount_match = AMOUNT_PATTERN.match(lines[i + 2])

                if amount_match:
                    amount = float(
                        amount_match.group(1).replace(",", "")
                    )

                    transactions.append(
                        {
                            "date": date,
                            "merchant": merchant,
                            "amount": amount,
                        }
                    )

                    i += 3
                    continue

        i += 1

    return transactions


if __name__ == "__main__":
    from pdf_reader import extract_text_from_pdf

    pdf_path = "data/sample_statement.pdf"

    text = extract_text_from_pdf(pdf_path)

    transactions = parse_transactions(text)

    print("----- PARSED TRANSACTIONS -----")

    for transaction in transactions:
        print(transaction)