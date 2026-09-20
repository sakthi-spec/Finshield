from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"

    return text


if __name__ == "__main__":
    pdf_path = "data/sample_statement.pdf"
    text = extract_text_from_pdf(pdf_path)

    print("----- EXTRACTED TEXT -----")
    print(text)