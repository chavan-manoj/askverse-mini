"""
Entry point for AskVerse Mini
"""

import os
import time
import logging
from colorama import Fore, Style
from dotenv import load_dotenv
from askverse_mini.askverse_system import *

def run_system(system: str = "wiki"):
    askverse_system = setup_askverse_system(system)
    print("-" * 80)
    
    while True:
        question = input("\nEnter your question, (or q|quit to swtich system): ").strip()
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

def main():
    logging.basicConfig(level=logging.WARN)
    load_dotenv()

    while True:
        system = input("\nChoose the system (wiki|tavily|arxiv|docs|ensemble) or (quit|q) to exit: ").strip().lower()
        
        if system in ("q", "quit"):
            print("Thank you for using AskVerse Mini!")
            break
        elif system not in ("wiki", "tavily", "arxiv", "docs", "ensemble"):
            print("Invalid choice.")
            continue
        else:
            run_system(system)

if __name__ == "__main__":
    main()
