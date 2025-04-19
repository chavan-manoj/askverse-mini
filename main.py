"""
Entry point for AskVerse Mini
"""

import os
import time
import logging
from colorama import Fore, Style
from dotenv import load_dotenv
from askverse_mini.ask_ensemble import AskEnsemble
from askverse_mini.document_processor import DocumentProcessor
from askverse_mini.qa_system import AskVerse
from askverse_mini.ask_wiki import AskWiki
from askverse_mini.ask_tavily import AskTavily
from askverse_mini.ask_docs import AskDocs
from askverse_mini.ask_arxiv import AskArxiv

def setup_document_processor():
    processor = DocumentProcessor()
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
    for doc_id, info in doc_info['documents'].items():
        print(f"\nLoaded document: {doc_id} with {info['num_chunks']} chunks and {info['total_pages']} pages.")
    return processor


def main_askverse():
    processor = setup_document_processor()
    
    # Initialize QA systems with different retriever strategies
    qa_systems = {
        "dense_only": AskVerse(),
        "sparse_only": AskVerse(),
        "ensemble": AskVerse(),
        "web_ensemble": AskVerse()
    }
    
    qa_systems["dense_only"].initialize(processor, use_web_search=False, retriever_kind="dense")
    qa_systems["sparse_only"].initialize(processor, use_web_search=False, retriever_kind="sparse")
    qa_systems["ensemble"].initialize(processor, use_web_search=False, retriever_kind="ensemble")
    qa_systems["web_ensemble"].initialize(processor, use_web_search=True, retriever_kind="ensemble")
    

    current_system = "dense_only"  # Default system

    print("\nAskVerse Mini is ready! Ask questions like \"What is Google's environment policy?\", \"How is Google helping people make more sustainable choices through its products?\"")
    print("\nType 'quit' at anytime to exit.")
    print("Type 'use dense', 'use sparse', or 'use ensemble' or 'use web ensemble' to switch between retrieval strategies.")
    print("-" * 80)
    
    while True:
        # Get user input
        question = input("\nEnter your question: ").strip()
        
        # Check for quit command
        if question.lower() == "quit":
            print("\nThank you for using AskVerse Mini!")
            break
            
        # Check for system switch commands
        if question.lower() == "use dense":
            current_system = "dense_only"
            print("\nSwitched to dense retriever only")
            continue
        elif question.lower() == "use sparse":
            current_system = "sparse_only"
            print("\nSwitched to sparse retriever only")
            continue
        elif question.lower() == "use ensemble":
            current_system = "ensemble"
            print("\nSwitched to ensemble retriever (dense + sparse)")
            continue
        elif question.lower() == "use web ensemble":
            current_system = "web_ensemble"
            print("\nSwitched to web ensemble retriever (dense + sparse + web search)")
            continue

        # Skip empty questions
        if not question:
            continue
            
        try:
            # Get answer from current system
            start_time = time.time()
            answer = qa_systems[current_system].ask_with_metrics(question)
            
            # Print answer
            print(f"\nAnswer (using {current_system}) (time taken: {round(time.time() - start_time, 2)} seconds):")
            print("-" * 80)
            print(answer["content"])
            print("-" * 80)

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try rephrasing your question or ask a different one.")

def main_askverse_mini(system: str = "wiki"):
    if system == "wiki":
        askverse_mini = AskWiki()
        askverse_mini.initialize()
    elif system == "tavily":
        askverse_mini = AskTavily()
        askverse_mini.initialize()
    elif system == "arxiv":
        askverse_mini = AskArxiv()
        askverse_mini.initialize()
    elif system == "docs":
        askverse_mini = AskDocs()
        askverse_mini.initialize(document_processor=setup_document_processor(), retriever_kind="dense")
    elif system == "ensemble":
        askverse_mini = AskEnsemble()
        askverse_mini.initialize(document_processor=setup_document_processor(), retriever_kind="ensemble")

    print("\nAskVerse", system, "is ready!\nType 'quit' at anytime to exit.")
    print("-" * 80)
    
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() == "quit":
            print("\nThank you for using AskVerse Mini!")
            break

        answer = askverse_mini.ask(question)
        print("Answer:")
        print(Fore.LIGHTBLUE_EX, answer["answer"], Style.RESET_ALL)
        print("Sources:")

        sorted_sources = sorted(answer["sources"])
        print(Fore.LIGHTBLUE_EX, end="")
        for idx, source in enumerate(sorted_sources, start=1):
            print(f"{idx}. {source}")
        print(Style.RESET_ALL)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    load_dotenv()

    while True:
        system = input("\nChoose the system (wiki|tavily|arxiv|docs|ensemble) or quit to quit: ").strip().lower()
        
        if system == "quit":
            print("\nThank you for using AskVerse Mini!")
            break
        elif system not in ("wiki", "tavily", "arxiv", "docs", "ensemble"):
            print("Invalid choice.")
            continue
        else:
            main_askverse_mini(system)
