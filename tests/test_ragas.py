"""
Test script for RAGAS evaluation metrics
"""

import os
import pandas as pd
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()

def load_test_data():
    """Load test data from CSV file"""
    # Create sample test data
    test_data = {
        "question": [
            "What are Microsoft's environmental goals for 2025?",
            "What is Google's carbon footprint?",
            "What are Microsoft's sustainability initiatives?",
            "What is Google's renewable energy usage?",
            "What are Microsoft's water conservation efforts?",
            "What is Google's waste management strategy?",
            "What are Microsoft's biodiversity commitments?",
            "What is Google's circular economy approach?"
        ],
        "answer": [
            "Microsoft aims to be carbon negative by 2030 and remove all historical carbon emissions by 2050. They are investing in renewable energy and sustainable practices.",
            "Google has been carbon neutral since 2007 and aims to operate on 24/7 carbon-free energy by 2030.",
            "Microsoft focuses on carbon reduction, water conservation, and waste management through various programs.",
            "Google has achieved 100% renewable energy for global operations and continues to invest in clean energy projects.",
            "Microsoft has committed to being water positive by 2030 through conservation and replenishment programs.",
            "Google has achieved zero waste to landfill across all data centers and offices.",
            "Microsoft has pledged to protect more land than they use by 2025.",
            "Google implements circular economy principles in their hardware and data center operations."
        ],
        "contexts": [
            ["Microsoft's environmental report shows their commitment to sustainability."],
            ["Google's environmental report details their carbon reduction efforts."],
            ["Microsoft has various sustainability programs across their operations."],
            ["Google's renewable energy initiatives are outlined in their report."],
            ["Microsoft's water conservation programs are described in detail."],
            ["Google's waste management approach is documented in their report."],
            ["Microsoft's biodiversity commitments are outlined in their report."],
            ["Google's circular economy initiatives are described in their report."]
        ],
        "reference": [
            "Microsoft aims to be carbon negative by 2030 and remove all historical carbon emissions by 2050. They are investing in renewable energy and sustainable practices.",
            "Google has been carbon neutral since 2007 and aims to operate on 24/7 carbon-free energy by 2030.",
            "Microsoft focuses on carbon reduction, water conservation, and waste management through various programs.",
            "Google has achieved 100% renewable energy for global operations and continues to invest in clean energy projects.",
            "Microsoft has committed to being water positive by 2030 through conservation and replenishment programs.",
            "Google has achieved zero waste to landfill across all data centers and offices.",
            "Microsoft has pledged to protect more land than they use by 2025.",
            "Google implements circular economy principles in their hardware and data center operations."
        ]
    }
    
    return test_data

def main():
    """Main function to run RAGAS evaluation"""
    print("Loading test data...")
    test_data = load_test_data()
    
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
    print(df.to_string())
    df.to_csv("tests/ragas_evaluation_results.csv", index=False)

if __name__ == "__main__":
    main()