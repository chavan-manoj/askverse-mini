"""
Entry point for AskVerse Mini
"""

import os
import time
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
        if pdf_file.endswith(".pdf") and pdf_file.startswith("google"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Loading PDF: {pdf_file}")
            processor.load_pdf(pdf_path)
    
    # Set up retrieval system
    processor.setup_retrievers()
    
    # Print document information
    doc_info = processor.get_document_info()
    for doc_id, info in doc_info['documents'].items():
        print(f"\nLoaded document: {doc_id} with {info['num_chunks']} chunks and {info['total_pages']} pages.")
    
    # Initialize QA system
    qa_system = AskVerse()
    qa_system.initialize(processor)

    qa_system_wo_web = AskVerse()
    qa_system_wo_web.initialize(processor, use_web_search=False)


    print("\nAskVerse Mini is ready! Ask questions like ""What is Google's environment policy?"", ""How is Google helping people make more sustainable choices through its products?""")
    print("\nType 'quit' at anytime to exit.")
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
            start_time = time.time()
            answer_wo_web = qa_system_wo_web.ask(question)
            
            # Print answer
            print(f"\nAnswer (without web search) (time taken: {time.time() - start_time} seconds):")
            print("-" * 80)
            print(answer_wo_web)
            print("-" * 80)

            start_time = time.time()
            answer = qa_system.ask(question)
            
            # Print answer
            print(f"\nAnswer (with web search) (time taken: {time.time() - start_time} seconds):")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try rephrasing your question or ask a different one.")

if __name__ == "__main__":
    main() 