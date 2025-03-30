# AskVerse Mini - System Design Document

## Overview

AskVerse Mini is a sophisticated multi-agent AI system that combines document question answering with web search capabilities. The system uses a hybrid approach to provide comprehensive answers by leveraging both local document knowledge and real-time web search results.

## System Architecture

```mermaid
graph TB
    subgraph Input Layer
        PDF[PDF Documents]
        Q[User Questions]
    end

    subgraph Document Processing
        DP[Document Processor]
        CR[Chunking & Retrieval]
    end

    subgraph Search Layer
        WS[Web Search Engine]
        SR[Search Results]
    end

    subgraph Agent Layer
        RA[Research Agent]
        QA[QA Agent]
        SA[Synthesis Agent]
    end

    subgraph Output Layer
        A[Final Answer]
    end

    PDF --> DP
    DP --> CR
    Q --> RA
    RA --> WS
    WS --> SR
    CR --> QA
    SR --> QA
    QA --> SA
    SA --> A
```

## Component Details

### 1. Document Processing Layer

The document processing layer handles the ingestion and preparation of PDF documents:

```mermaid
graph LR
    subgraph Document Processing
        PDF[PDF Input] --> EX[Text Extraction]
        EX --> CH[Chunking]
        CH --> DR[Dense Retrieval]
        CH --> SR[Sparse Retrieval]
        DR --> ER[Ensemble Retriever]
        SR --> ER
    end
```

- **Text Extraction**: Converts PDF documents into plain text while preserving structure
- **Chunking**: Splits text into manageable chunks with overlap for context preservation
- **Hybrid Retrieval**:
  - Dense Retrieval: Uses OpenAI embeddings for semantic search
  - Sparse Retrieval: Uses BM25 for keyword-based search
  - Ensemble: Combines both approaches for better results

### 2. Agent Workflow

The system employs a multi-agent architecture with specialized agents:

```mermaid
sequenceDiagram
    participant User
    participant Research Agent
    participant QA Agent
    participant Synthesis Agent
    participant Web Search
    participant Document Store

    User->>Research Agent: Submit Question
    Research Agent->>Web Search: Search for relevant info
    Web Search-->>Research Agent: Return search results
    Research Agent->>QA Agent: Process with context
    QA Agent->>Document Store: Query relevant chunks
    Document Store-->>QA Agent: Return document context
    QA Agent->>Synthesis Agent: Combine information
    Synthesis Agent->>User: Return final answer
```

#### Agent Roles

1. **Research Agent**
   - Handles web search queries
   - Filters and processes search results
   - Identifies relevant information gaps

2. **QA Agent**
   - Processes document chunks
   - Extracts relevant information
   - Combines document context with search results

3. **Synthesis Agent**
   - Merges information from multiple sources
   - Ensures consistency and coherence
   - Generates final comprehensive answer

### 3. Information Flow

```mermaid
graph TD
    subgraph Input
        Q[Question]
        D[Documents]
        W[Web Search]
    end

    subgraph Processing
        RC[Research Context]
        DC[Document Context]
        IC[Information Combination]
    end

    subgraph Output
        A[Answer]
    end

    Q --> RC
    W --> RC
    D --> DC
    RC --> IC
    DC --> IC
    IC --> A
```

## Response Enhancement Process

1. **Initial Question Processing**
   - Question is analyzed for required information types
   - System determines if web search is needed

2. **Parallel Information Gathering**
   - Document retrieval from local PDFs
   - Web search for current/relevant information
   - Both processes run concurrently for efficiency

3. **Information Integration**
   - Document context provides foundational knowledge
   - Web search results add current/relevant information
   - System identifies and resolves any contradictions

4. **Response Generation**
   - Information is synthesized into coherent answer
   - Sources are properly attributed
   - Response is formatted for clarity

## Error Handling and Fallbacks

```mermaid
graph TD
    A[Start] --> B{Web Search Available?}
    B -->|Yes| C[Use Web Search]
    B -->|No| D[Document Only Mode]
    C --> E{Search Successful?}
    E -->|Yes| F[Combine Results]
    E -->|No| D
    D --> G[Generate Answer]
    F --> G
```

## Performance Considerations

1. **Caching**
   - Document embeddings are cached
   - Frequently accessed chunks are stored in memory
   - Search results are cached temporarily

2. **Parallel Processing**
   - Document processing runs in parallel
   - Web search and document retrieval are concurrent
   - Agent operations are optimized for speed

3. **Resource Management**
   - Memory usage is monitored
   - Large documents are processed in chunks
   - Search results are limited to relevant content

## Future Enhancements

1. **Planned Improvements**
   - Support for more document types
   - Enhanced caching mechanisms
   - Improved search result filtering
   - Better source attribution

2. **Potential Features**
   - Multi-language support
   - Custom agent configurations
   - Advanced visualization options
   - API endpoint for integration

## Security Considerations

1. **API Key Management**
   - Secure storage of API keys
   - Environment variable usage
   - No hardcoded credentials

2. **Data Privacy**
   - Local document processing
   - Secure API communications
   - No data storage of sensitive information

## Monitoring and Logging

```mermaid
graph LR
    subgraph Monitoring
        P[Performance Metrics]
        E[Error Logs]
        U[Usage Statistics]
    end

    subgraph Logging
        L[Log Files]
        A[Analytics]
        D[Debug Info]
    end

    P --> L
    E --> L
    U --> L
    L --> A
    L --> D
```

## Conclusion

AskVerse Mini's architecture combines the power of local document processing with real-time web search capabilities through a sophisticated multi-agent system. The design ensures efficient information processing, reliable response generation, and scalable operation while maintaining security and performance standards. 