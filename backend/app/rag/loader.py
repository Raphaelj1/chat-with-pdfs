from pathlib import Path
from typing import BinaryIO
import fitz
from app.models.document import Document


def _parse_pdf(
    pdf: fitz.Document,
    document_name: str,
) -> list[Document]:
    """
    Extract text from every page of a PDF.
    """

    documents: list[Document] = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text().strip()

        if not text:
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "source": document_name,
                    "page": page_number,
                },
            )
        )

    return documents


def load_pdf_file(
    pdf_path: Path,
) -> list[Document]:
    """
    Load a PDF from disk.
    """

    pdf = fitz.open(pdf_path)

    try:
        return _parse_pdf(
            pdf=pdf,
            document_name=pdf_path.name,
        )
    finally:
        pdf.close()


def load_pdf_stream(
    file: BinaryIO,
    filename: str,
) -> list[Document]:
    """
    Load a PDF from an uploaded file.
    """

    pdf = fitz.open(
        stream=file.read(),
        filetype="pdf",
    )

    try:
        return _parse_pdf(
            pdf=pdf,
            document_name=filename,
        )
    finally:
        pdf.close()


def load_directory(
    directory: Path,
) -> list[Document]:
    """
    Load every PDF inside a directory.
    """

    documents: list[Document] = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        documents.extend(
            load_pdf_file(pdf_path)
        )

    return documents