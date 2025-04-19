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

document_processor = None
def setup_document_processor():
    global document_processor
    if document_processor is not None:
        return document_processor
    
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
    
    document_processor = processor
    return document_processor

askverse_systems = {}
def setup_askverse_system(system: str):
    global askverse_systems
    if system in askverse_systems:
        return askverse_systems[system]

    if system == "wiki":
        askverse_system = AskWiki()
        askverse_system.initialize()
    elif system == "tavily":
        askverse_system = AskTavily()
        askverse_system.initialize()
    elif system == "arxiv":
        askverse_system = AskArxiv()
        askverse_system.initialize()
    elif system == "docs":
        askverse_system = AskDocs()
        askverse_system.initialize(document_processor=setup_document_processor())
    elif system == "ensemble":
        askverse_system = AskEnsemble()
        askverse_system.initialize(document_processor=setup_document_processor())

    askverse_systems[system] = askverse_system
    return askverse_system

def run_system(system: str = "wiki"):
    askverse_system = setup_askverse_system(system)
    print("-" * 80)
    
    while True:
        question = input("\nEnter your question, q|quit to exit: ").strip()
        if question.lower() in ("q", "quit"):
            break

        start_time = time.time()
        answer = askverse_system.ask(question)
        print(f"Answer (using {system}) (time taken: {round(time.time() - start_time, 2)} seconds):")
        print(Fore.LIGHTBLUE_EX)
        print(answer["answer"], Style.RESET_ALL)

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
        system = input("\nChoose the system (wiki|tavily|arxiv|docs|ensemble) or quit|q to exit: ").strip().lower()
        
        if system in ("q", "quit"):
            print("Thank you for using AskVerse Mini!")
            break
        elif system not in ("wiki", "tavily", "arxiv", "docs", "ensemble"):
            print("Invalid choice.")
            continue
        else:
            run_system(system)
