"""
Ask Wiki
"""

import logging
from askverse_mini.askverse_base import AskVerseBase
from langchain_community.retrievers import TavilySearchAPIRetriever

logger = logging.getLogger(__name__)

class AskTavily(AskVerseBase):
    
    def _prepare_retriever(self):
        retriever = TavilySearchAPIRetriever(k=10)
        logger.info(f"Retriever:\n{retriever}")
        return retriever
