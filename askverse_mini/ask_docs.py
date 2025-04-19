"""
AskVerse Mini
"""

import logging
from askverse_mini.askverse_base import AskVerseBase
from askverse_mini.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class AskDocs(AskVerseBase):

    def __init__(self, document_processor: DocumentProcessor, retriever_kind: str = "dense"):
        super().__init__()
        self.document_processor = document_processor
        self.retriever_kind = retriever_kind
    
    def _prepare_retriever(self):
        retriever = self.document_processor.get_retriever(self.retriever_kind)
        logger.info(f"Retriever:\n{retriever}")
        return retriever
