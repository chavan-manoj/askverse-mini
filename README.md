# AskVerse Mini

A lightweight multi-agent AI system that combines document question answering with web search capabilities, built with LangChain and LangGraph. It can answer questions based on your local PDF documents and enhance responses with real-time web search results.

## Features

- Multi-agent architecture for robust question answering
- Support for multiple PDF documents
- Web search integration using Tavily
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

Using standard venv:
```bash
python -m venv askverse-mini-venv
source askverse-mini-venv/bin/activate  # On Windows: askverse-mini-venv\Scripts\activate
```

Or using Conda:
```bash
conda create -n askverse-mini-venv python=3.9
conda activate askverse-mini-venv
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

You can get your API keys from:
- OpenAI API key: https://platform.openai.com/api-keys
- Tavily API key: https://tavily.com/

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