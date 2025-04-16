"""
Ask Wiki
"""

import os
import logging
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

logger = logging.getLogger(__name__)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class AskWiki:
    
    def initialize(self):
        model_name = os.getenv("MODEL_NAME")
        self.llm = ChatOpenAI(model_name=model_name, temperature=0, streaming=True)
        self.retriever = self._prepare_retriever()
        self.prompt = self._prepare_prompt()
        self.graph = self._prepare_graph()
        
    def ask(self, question: str):
        result = self.graph.invoke({"question": question})
        sources = [doc.metadata["source"] for doc in result["context"]]
        return {
            "sources": sources,
            "answer": result["answer"]
        }
        
    def _retrieve(self, state: State):
        retrieved_docs = self.retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    def _generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}
    
    def _prepare_prompt(self):
        system_prompt = (
            "You're a helpful AI assistant. Given a user question "
            "and some Wikipedia article snippets, answer the user "
            "question. If none of the articles answer the question, "
            "just say you don't know."
            "\n\nHere are the Wikipedia articles: "
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )
        logger.info(f"Prompt:\n{prompt.pretty_repr()}")
        return prompt
    
    def _prepare_retriever(self):
        retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)
        logger.info(f"Retriever:\n{retriever}")
        return retriever
        
    def _prepare_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", self._retrieve)
        graph_builder.add_node("generate", self._generate)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph = graph_builder.compile()
        logger.info(f"Graph (Mermaid format):\n{graph.get_graph().draw_mermaid()}")
        return graph

