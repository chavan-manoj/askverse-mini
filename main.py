"""
Entry point for AskVerse Mini
"""

import os
from dotenv import load_dotenv
from askverse_mini.document_processor import DocumentProcessor
from askverse_mini.qa_system import AskVerse

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Load multiple PDF documents
    pdf_dir = "pdfs"  # Directory containing PDF files
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Loading PDF: {pdf_file}")
            processor.load_pdf(pdf_path)
    
    # Set up retrieval system
    processor.setup_retrievers()
    
    # Print document information
    doc_info = processor.get_document_info()
    print("\nLoaded Documents:")
    print(f"Total documents: {doc_info['num_documents']}")
    print(f"Total chunks: {doc_info['total_chunks']}")
    for doc_id, info in doc_info['documents'].items():
        print(f"\nDocument: {doc_id}")
        print(f"File: {info['file_name']}")
        print(f"Pages: {info['total_pages']}")
        print(f"Chunks: {info['num_chunks']}")
    
    # Initialize QA system
    qa_system = AskVerse()
    qa_system.initialize(processor)
    
    print("\nAskVerse Mini is ready! Ask questions like 'What is Microsft's environment policy?', 'How is Google helping people make more sustainable choices'")
    print("\nType 'quit' to exit.")
    print("-" * 80)
    
    while True:
        # Get user input
        question = input("\nEnter your question: ").strip()
        
        # Check for quit command
        if question.lower() == "quit":
            print("\nThank you for using AskVerse Mini!")
            break
            
        # Skip empty questions
        if not question:
            continue
            
        try:
            # Get answer
            answer = qa_system.ask(question)
            
            # Print answer
            print("\nAnswer:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try rephrasing your question or ask a different one.")

if __name__ == "__main__":
    main() 