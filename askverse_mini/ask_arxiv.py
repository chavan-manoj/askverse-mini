"""
Ask Arxiv
"""

import logging
from askverse_mini.askverse_base import AskVerseBase
from langchain_community.retrievers import ArxivRetriever

logger = logging.getLogger(__name__)

class AskArxiv(AskVerseBase):
    
    def _prepare_retriever(self):
        retriever = ArxivRetriever(load_max_docs=10, get_full_documents=False, load_all_available_meta=False)
        logger.info(f"Retriever:\n{retriever}")
        return retriever
    
    def _massage_retrieved_docs(self, retrieved_docs):
        for doc in retrieved_docs:
            if not doc.metadata.get("source"):
                if doc.metadata.get("Entry ID") and doc.metadata.get("Title"):
                    doc.metadata["source"] = doc.metadata["Entry ID"] + " - " + doc.metadata["Title"]
        return retrieved_docs