# AskVerse: Design & Architecture

## 1. Introduction

AskVerse is a **Secure, Scalable, and Real-time Knowledge Management System** leveraging a **Multi-Agentic Retrieval-Augmented Generation (RAG) Pipeline**. The system consists of multiple retrievers, real-time API orchestration, document scoring, uncertainty quantification, and access control mechanisms.

This document describes the **architecture and design** of the system, including:
- The **RAG pipeline workflow**
- **Publishing OpenAPI specs to VectorDB**
- **Publishing enterprise documents to VectorDB**

---

## 2. RAG Workflow (Multi-Agentic RAG Pipeline)

```mermaid
graph TD
    %% User Input and Initial Processing
    UserQuery[User Query] -->|LLM Identifies Subqueries| QueryBreaker[Query Breakdown Module]
    
    %% Subqueries to Different Retrievers
    QueryBreaker -->|Dense Search| DenseRetriever[Dense Retriever Vector DB]
    QueryBreaker -->|Sparse Search| SparseRetriever[Sparse Retriever BM25]
    QueryBreaker -->|Ensemble Search| EnsembleRetriever[Ensemble Retriever]
    QueryBreaker -->|Real-time Web Search| WebRetriever[Real-time Web Search]
    QueryBreaker -->|API Calls| APILookup[API Lookup Vector DB with OpenAPI Specs]
    
    %% API Selection Process
    APILookup -->|Find Suitable APIs| LLM[LLM API Selector]
    LLM -->|Call API 1| API1[API Call 1]
    API1 -->|Data Response| APIRetriever[API Data Retriever]
    LLM -->|Call API 2 | API2[API Call 2]
    API2 -->|Additional Data| APIRetriever
    
    %% Combining Retrieved Information
    DenseRetriever -->|Docs + Citation| Scorer[Document Scorer]
    SparseRetriever -->|Docs + Citation| Scorer
    EnsembleRetriever -->|Docs + Citation| Scorer
    WebRetriever -->|Docs + Citation| Scorer
    APIRetriever -->|API Data + Citation| Scorer
    
    %% Document Scoring & Query Refinement
    Scorer -->|Score < Threshold?| ReThink[Rephrase & Retry Thinking Mode]
    ReThink -->|Re-execute Query| QueryBreaker
    Scorer -->|Score ≥ Threshold| ResponseProcessor[Response Processing]
    
    %% Final Processing (Citations, Confidence, Masking)
    ResponseProcessor -->|Compute Confidence Score| Confidence[Confidence Score Calculation]
    ResponseProcessor -->|Mask Sensitive Data| DataMasking[PII & Sensitive Data Handling]
    
    %% Final Response
    Confidence --> FinalResponse[Final Answer with Citations & Confidence Score]
    DataMasking --> FinalResponse
    FinalResponse --> User[User Receives Answer]
```

### **Description:**
- **Query Breakdown:** LLM decomposes complex queries into subqueries.
- **Multi-Retriever System:** Uses Dense, Sparse, Ensemble retrievers, and Web Search.
- **API Orchestration:** Finds and invokes APIs dynamically based on OpenAPI specs stored in Vector DB.
- **Document Scoring & Re-execution:** If results are poor, the system retries with rephrased queries.
- **Final Processing:** Adds citations, computes confidence scores, and masks sensitive data.

---

## 3. OpenAPI Specification Publisher

```mermaid
graph TD
    OpenAPI_Specs[OpenAPI Spec Documents] -->|Parse API Definitions| SpecProcessor[OpenAPI Spec Processor]
    SpecProcessor -->|Generate Embeddings| EmbeddingModel[OpenAI Embeddings]
    EmbeddingModel -->|Store API Vectors| VectorDB[Vector Database]
```

### **Description:**
- **Processes OpenAPI Specifications** to extract API definitions.
- **Generates vector embeddings** for each API.
- **Stores API vectors in VectorDB**, enabling dynamic API selection in the RAG pipeline.

---

## 4. Document Ingestion & Publishing to VectorDB

```mermaid
graph TD
    Confluence[Confluence Docs] -->|Fetch Content| ConfluenceListener[Confluence Listener]
    Websites[Enterprise Websites] -->|Scrape Content| WebScraper[Web Scraper]
    PDFs[PDF Documents] -->|Parse PDFs| PDFProcessor[PDF Parser]
    
    ConfluenceListener -->|Extract Text| TextProcessor[Text Processing Module]
    WebScraper -->|Extract Text| TextProcessor
    PDFProcessor -->|Extract Text| TextProcessor
    
    TextProcessor -->|Generate Embeddings| DocEmbeddingModel[OpenAI Embeddings]
    DocEmbeddingModel -->|Store in VectorDB| VectorDB_Docs[Vector Database Documents]
```

### **Description:**
- **Confluence Listener, Web Scraper, PDF Processor** extract data from various sources.
- **Text Processing Module** cleans and standardizes extracted content.
- **Embeddings are generated** using OpenAI models.
- **Processed documents are stored in VectorDB**, enabling retrieval in the RAG pipeline.

---

## 5. Conclusion

This design ensures:
- ✅ **Scalability** through multi-agent retrieval and API orchestration.
- ✅ **Security & Compliance** via access control and PII masking.
- ✅ **Accuracy & Transparency** by incorporating document scoring, citations, and confidence scores.
- ✅ **Real-time Adaptability** via OpenAPI-driven API selection and real-time web search.

This architecture supports a **highly extensible, plug-and-play knowledge management system** for enterprises.
