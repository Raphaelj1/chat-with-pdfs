from dataclasses import dataclass, field
from typing import BinaryIO


@dataclass(slots=True)
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class UploadedDocument:
    filename: str
    file: BinaryIO