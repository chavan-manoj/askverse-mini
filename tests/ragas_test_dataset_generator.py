import os
import pandas as pd
from PyPDF2 import PdfReader
from tqdm import tqdm
from ragas.testset import TestsetGenerator
from datasets import Dataset
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pdfs(pdf_dir):
    """Load all PDFs in the directory and return as documents."""
    documents = None
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        if documents is None:
            documents = PyPDFLoader(pdf_path).load()
        else:
            documents.extend(PyPDFLoader(pdf_path).load())
    return documents

def generate_test_dataset(pdf_dir, num_questions=10, output_csv="ragas_test_dataset.csv"):
    """Generate a test dataset using RAGAS."""
    logger.info(f"Generating test dataset with target of {num_questions} questions")
    
    documents = load_pdfs(pdf_dir)
    
    llm = ChatOpenAI(model=os.getenv("MODEL_NAME"), temperature=0.1)
    embedding_model = OpenAIEmbeddings()
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embedding_model)
    
    # Generate test dataset
    test_data = generator.generate_with_langchain_docs(
        documents,
        num_questions
    )
    
    # Convert to pandas DataFrame for easier handling
    df = pd.DataFrame({
        "question": test_data["question"],
        "ground_truth": test_data["ground_truth"],
        "context": test_data["contexts"],
        "source": [ctx[0]["metadata"]["source"] if ctx and len(ctx) > 0 else "unknown" for ctx in test_data["contexts"]]
    })
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Created test dataset with {len(df)} questions saved to {output_csv}")
    
    return df

def main():
    pdf_dir = "pdfs"
    output_csv = "ragas_test_dataset.csv"
    num_questions = 10
    
    generate_test_dataset(pdf_dir, num_questions, output_csv)

if __name__ == "__main__":
    main()