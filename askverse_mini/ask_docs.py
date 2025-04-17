"""
AskVerse Mini
"""

import logging
from askverse_mini.askverse_base import AskVerseBase
from askverse_mini.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class AskDocs(AskVerseBase):
    
    def initialize(self, document_processor: DocumentProcessor, retriever_kind: str = "ensemble"):
        self.llm = self._prepare_llm()
        self.retriever = self._prepare_retriever(document_processor, retriever_kind)
        self.prompt = self._prepare_prompt()
        self.graph = self._prepare_graph()
        
    def _prepare_retriever(self, document_processor: DocumentProcessor, retriever_kind: str = "ensemble"):
        retriever = document_processor.get_retriever(retriever_kind)
        logger.info(f"Retriever:\n{retriever}")
        return retriever
