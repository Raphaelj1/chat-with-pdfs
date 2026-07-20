from pathlib import Path
import fitz
from app.models.document import Document


def load_pdf(file_path: Path) -> list[Document]:
    """
    Load a PDF and return one Document per page.
    """

    documents: list[Document] = []

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if not text:
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source": file_path.name,
                        "page": page_number,
                    },
                )
            )

    return documents


def load_documents(data_dir: Path) -> list[Document]:
    """
    Load every PDF in the data directory.
    """

    documents: list[Document] = []

    for pdf_file in data_dir.glob("*.pdf"):
        documents.extend(load_pdf(pdf_file))

    return documents