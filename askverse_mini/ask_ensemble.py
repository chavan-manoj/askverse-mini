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
    
    def __init__(self, document_processor: DocumentProcessor, retriever_kind: str = "dense"):
        super().__init__()
        self.document_processor = document_processor
        self.retriever_kind = retriever_kind

    def _prepare_retriever(self):
        docs_retriever = AskDocs(document_processor=self.document_processor, retriever_kind = self.retriever_kind)._prepare_retriever()
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