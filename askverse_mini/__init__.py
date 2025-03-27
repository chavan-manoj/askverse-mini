"""
AskVerse: A multi-agent AI question answering system
"""

__version__ = "0.1.0"

from .document_processor import DocumentProcessor
from .qa_system import AskVerse

__all__ = ["DocumentProcessor", "AskVerse"] 