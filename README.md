# AskVerse Mini

A lightweight multi-agent AI system for document question answering, built with LangChain and LangGraph.

## Features

- Multi-agent architecture for robust question answering
- Support for multiple PDF documents
- Hybrid retrieval system (dense + sparse)
- Interactive command-line interface
- Configurable logging

## Architecture

The system uses a multi-agent architecture with the following components:

1. **Document Processor**
   - PDF text extraction
   - Text chunking
   - Hybrid retrieval setup (Chroma + BM25)

2. **QA System**
   - Multi-agent workflow using LangGraph
   - Specialized agents for different tasks
   - Structured output handling

3. **Retrieval System**
   - Dense retrieval using OpenAI embeddings
   - Sparse retrieval using BM25
   - Ensemble retriever for better results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chavan-manoj/askverse-mini.git
cd askverse-mini
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Place your PDF documents in the `pdfs` directory

2. Run the main script:
```bash
python main.py
```

3. Ask questions about your documents. Type 'quit' to exit.

## Example Questions

- "What are Google's environmental initiatives?"
- "How is Google helping people make more sustainable choices?"
- "What are Google's sustainability goals?"

## Dependencies

- langchain==0.1.12
- langchain-openai==0.0.8
- langgraph==0.0.27
- chromadb==0.4.24
- PyPDF2==3.0.1
- python-dotenv==1.0.1
- openai==1.14.2
- rank_bm25==0.2.2

## License

MIT License 