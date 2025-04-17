"""
Ask Wiki
"""

import logging
from askverse_mini.askverse_base import AskVerseBase
from langchain_community.retrievers import WikipediaRetriever

logger = logging.getLogger(__name__)

class AskWiki(AskVerseBase):
    
    def _prepare_retriever(self):
        retriever = WikipediaRetriever(top_k_results=10, doc_content_chars_max=2000)
        logger.info(f"Retriever:\n{retriever}")
        return retriever
        