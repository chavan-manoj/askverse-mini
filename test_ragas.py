"""
Test script for RAGAS evaluation metrics
"""

import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
)
from ragas import evaluate
from langchain_openai import ChatOpenAI
from datasets import Dataset
from askverse_mini.qa_system import AskVerse
from askverse_mini.document_processor import DocumentProcessor

# Load environment variables
load_dotenv()

def setup_document_processor(pdf_dir):
    processor = DocumentProcessor()
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Loading PDF: {pdf_file}")
            processor.load_pdf(pdf_path)
    
    processor.setup_retrievers()
    return processor

def generate_test_data(test_dataset_path):
    """Generate test data"""

    df = pd.read_excel(test_dataset_path)
    questions = df["Question"].tolist()
    references = df["Expected Answer"].tolist()
    answers = []
    contexts = []

    qa_systems = {
        "dense_only": AskVerse(),
        "sparse_only": AskVerse(),
        "ensemble": AskVerse(),
        "web_ensemble": AskVerse()
    }
    
    pdf_dir = "pdfs"  # Directory containing PDF files
    processor = setup_document_processor(pdf_dir)
    qa_systems["dense_only"].initialize(processor, use_web_search=False, retriever_kind="dense")
    qa_systems["sparse_only"].initialize(processor, use_web_search=False, retriever_kind="sparse")
    qa_systems["ensemble"].initialize(processor, use_web_search=False, retriever_kind="ensemble")
    qa_systems["web_ensemble"].initialize(processor, use_web_search=True, retriever_kind="ensemble")
    
    current_system = "dense_only" 

    print("Asking questions to AskVerse...")
    for question in tqdm(questions):
        answer = qa_systems[current_system].ask_with_metrics(question)
        answers.append(answer["content"])
        contexts.append([answer["retrieved_docs"]])
    
    test_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": references,
    }
    
    return test_data, df

def main():
    """Main function to run RAGAS evaluation"""
    test_data, test_dataset_df = generate_test_data("tests/ragas_test_dataset.xlsx")
    
    # Convert to Dataset format required by Ragas
    dataset = Dataset.from_dict(test_data)
    
    # Configure OpenAI LLM
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    print(f"\nUsing LLM model: {model_name}")
    
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0
    )
    
    print("\nRunning Ragas evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_entity_recall
        ],
        llm=llm
    )
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 30)
    df = result.to_pandas()
    df["Question Category"] = test_dataset_df["Question Category"]
    print(df.to_string())
    df.to_excel("tests/ragas_evaluation_results.xlsx", index=False)

if __name__ == "__main__":
    main()