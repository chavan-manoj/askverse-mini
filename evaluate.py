"""
Evaluate the RAG systems using RAGAS evaluation framework.

Usage:
python evaluate.py

Here are the metrics used for evaluation of Retriever component of RAG:
- Context Precision: Measures the proportion of relevant chunks in the retrieved contexts.
    This metric is computed using the question and the contexts.
- Context Recall: How many of the relevant documents (or pieces of information) were successfully retrieved.
    It focuses on not missing important results. Calculating context recall always requires a reference to compare against.
    This metric is computed using the reference answer (ground truth) and the contexts.
- Context Entity Recall: Measure of what fraction of entities are recalled from reference.
    This metric is computed using the entities in the reference answer (ground truth) and the contexts.

Here are the metrics used for evaluation of Generator component of RAG:
- Faithfulness: Measures how factually consistent a response is with the retrieved context.
    This metric uses the question, retrieved contexts and the generated answer.
- Answer Relevancy: Measures the relevance of the answer to the question.
    This metric uses the question and the generated answer.

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
from askverse_mini.askverse_system import *

def generate_test_data(test_dataset_path):
    """Generate test data"""

    print("Initializing AskVerse docs...")
    askverse_system = setup_askverse_system("docs")

    print("Loading test data...")
    df = pd.read_excel(test_dataset_path)
    questions = df["Question"].tolist()
    references = df["Expected Answer"].tolist()
    answers = []
    contexts = []

    print("Asking questions to AskVerse...")
    for question in tqdm(questions):
        response = askverse_system.ask(question)
        answers.append(response["answer"])
        contexts.append(response["retrieved_docs"])
    
    test_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": references,
    }
    return test_data, df

def main():
    """Main function to evaluate the RAG systems using RAGAS evaluation framework"""

    load_dotenv()
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
    
    output_file = "tests/ragas_evaluation_results.xlsx"
    df = result.to_pandas()
    df["Question Category"] = test_dataset_df["Question Category"]
    df.to_excel(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 30)
    print(df)

if __name__ == "__main__":
    main()