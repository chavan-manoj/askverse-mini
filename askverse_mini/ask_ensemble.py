"""
AskVerse Mini
"""

import logging
from askverse_mini.askverse_base import AskVerseBase
from askverse_mini.document_processor import DocumentProcessor
from langchain.retrievers import EnsembleRetriever
from askverse_mini.ask_docs import AskDocs
from askverse_mini.ask_wiki import AskWiki
from askverse_mini.ask_arxiv import AskArxiv
from askverse_mini.ask_tavily import AskTavily

logger = logging.getLogger(__name__)

class AskEnsemble(AskVerseBase):
    
    def initialize(self, document_processor: DocumentProcessor, retriever_kind: str = "ensemble"):
        self.llm = self._prepare_llm()
        self.retriever = self._prepare_retriever(document_processor, retriever_kind)
        self.prompt = self._prepare_prompt()
        self.graph = self._prepare_graph()
        
    def _prepare_retriever(self, document_processor: DocumentProcessor, retriever_kind: str = "ensemble"):
        docs_retriever = AskDocs()._prepare_retriever(document_processor=document_processor, retriever_kind = retriever_kind)
        wiki_retriever = AskWiki()._prepare_retriever()
        arxiv_retriever = AskArxiv()._prepare_retriever()
        tavily_retriever = AskTavily()._prepare_retriever()

        ensemble_retriever = EnsembleRetriever(
                retrievers=[docs_retriever, wiki_retriever, arxiv_retriever, tavily_retriever],
                weights=[0.4, 0.2, 0.2, 0.2],
                c=0
            )
        logger.info(f"Retriever:\n{ensemble_retriever}")
        return ensemble_retriever

    def _massage_retrieved_docs(self, retrieved_docs):
        return AskArxiv()._massage_retrieved_docs(retrieved_docs)