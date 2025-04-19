"""
Document processing module for AskVerse
"""

import os
import json
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
        
    def _load_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Load metadata from JSON file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict containing document metadata
        """
        # Get the metadata file path
        metadata_path = os.path.splitext(pdf_path)[0] + "_metadata.json"
        
        # Default metadata
        metadata = {
            "title": os.path.splitext(os.path.basename(pdf_path))[0].replace("-", " ").replace("_", " ").title()
        }
        
        # Load metadata from JSON if exists
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    json_metadata = json.load(f)
                    metadata.update(json_metadata)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse metadata file {metadata_path}: {e}")
            except Exception as e:
                print(f"Warning: Error loading metadata file {metadata_path}: {e}")
                
        return metadata
        
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
            
        # Load metadata from JSON
        metadata = self._load_metadata(pdf_path)
            
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
            # Combine JSON metadata with required fields
            chunk_metadata = {
                **metadata,  # Include all JSON metadata
                "source": doc_id + " Chunk " + str(i)
            }
            self.dense_documents.append(Document(page_content=text, metadata=chunk_metadata))
            
            chunk_metadata = {
                **metadata,  # Include all JSON metadata
                "source": doc_id + " Chunk " + str(i)
            }
            self.sparse_documents.append(Document(page_content=text, metadata=chunk_metadata))
            
        # Store document metadata
        self.document_metadata[doc_id] = {
            "file_path": pdf_path,
            "file_name": os.path.basename(pdf_path),
            "num_chunks": len(splits),
            "total_pages": len(pdf_reader.pages),
            **metadata  # Include all JSON metadata
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
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        self.sparse_retriever = BM25Retriever.from_documents(self.sparse_documents, k=10)
        
    def get_retriever(self, kind: str = "ensemble"):
        """Get the retriever based on the kind
        
        Args:
            kind: Type of retriever to return (ensemble, dense, sparse)
        """

        if kind == "ensemble":
            return EnsembleRetriever(
                retrievers=[self.dense_retriever, self.sparse_retriever],
                weights=[0.5, 0.5],
                c=0
            )
        elif kind == "dense":
            return self.dense_retriever
        elif kind == "sparse":
            return self.sparse_retriever
        else:
            raise ValueError(f"Invalid retriever kind: {kind}")
        
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
    
    def setup(self, pdf_dir: str = "pdfs") -> None:
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Loading PDF: {pdf_file}")
            self.load_pdf(pdf_path)
        
        print("Storing documents in VectorDB...")
        self.setup_retrievers()
        
        doc_info = self.get_document_info()
        for doc_id, info in doc_info['documents'].items():
            print(f"Loaded document: {doc_id} with {info['total_pages']} pages and {info['num_chunks']} chunks")
