"""
Document processing module for AskVerse
"""

import os
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class DocumentProcessor:
    """Handles document loading, processing, and retrieval setup"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_function = OpenAIEmbeddings()
        self.dense_documents = []
        self.sparse_documents = []
        self.vectorstore = None
        self.ensemble_retriever = None
        self.document_metadata = {}  # Track metadata for each document
        
    def load_pdf(self, pdf_path: str, doc_id: str = None) -> None:
        """
        Load and process a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            doc_id: Optional unique identifier for the document
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = os.path.basename(pdf_path)
            
        # Extract text from PDF
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # Split text into chunks
        splits = self.text_splitter.split_text(text)
        
        # Create documents for both dense and sparse retrieval with metadata
        start_idx = len(self.dense_documents)
        for i, text in enumerate(splits):
            metadata = {
                "id": str(start_idx + i),
                "source": "dense",
                "doc_id": doc_id,
                "file_name": os.path.basename(pdf_path)
            }
            self.dense_documents.append(Document(page_content=text, metadata=metadata))
            
            metadata = {
                "id": str(start_idx + i),
                "source": "sparse",
                "doc_id": doc_id,
                "file_name": os.path.basename(pdf_path)
            }
            self.sparse_documents.append(Document(page_content=text, metadata=metadata))
            
        # Store document metadata
        self.document_metadata[doc_id] = {
            "file_path": pdf_path,
            "file_name": os.path.basename(pdf_path),
            "num_chunks": len(splits),
            "total_pages": len(pdf_reader.pages)
        }
        
    def setup_retrievers(self, collection_name: str = "askverse_docs") -> None:
        """
        Set up the retrieval system with both dense and sparse retrievers
        
        Args:
            collection_name: Name for the Chroma collection
        """
        if not self.dense_documents or not self.sparse_documents:
            raise ValueError("No documents loaded. Call load_pdf() first.")
            
        # Set up Chroma vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.dense_documents,
            embedding=self.embedding_function,
            collection_name=collection_name
        )
        
        # Create retrievers
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        sparse_retriever = BM25Retriever.from_documents(self.sparse_documents, k=10)
        
        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5],
            c=0
        )
        
    def get_retriever(self):
        """Get the ensemble retriever"""
        if not self.ensemble_retriever:
            raise ValueError("Retrievers not set up. Call setup_retrievers() first.")
        return self.ensemble_retriever
        
    def get_document_info(self) -> Dict[str, Any]:
        """
        Get information about loaded documents
        
        Returns:
            Dict containing information about loaded documents
        """
        return {
            "num_documents": len(self.document_metadata),
            "total_chunks": len(self.dense_documents),
            "documents": self.document_metadata
        } 