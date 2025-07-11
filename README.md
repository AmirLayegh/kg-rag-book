# GraphRAG Book

A GraphRAG (Graph Retrieval-Augmented Generation) project using LangChain and Neo4j.

This repository contains the implementation examples and code from **"Essential GraphRAG: Knowledge Graph-Enhanced RAG"** by Tomaž Bratanič and Oskar Hane, published by [Manning Publications](https://www.manning.com/books/essential-graphrag).


## About the Book

![Essential GraphRAG Book Cover](data/kg-book.png)

**Essential GraphRAG** teaches you to implement accurate, performant, and traceable RAG by structuring the context data as a knowledge graph. The book covers how to upgrade your RAG applications with the power of knowledge graphs, delivering better performance, accuracy, traceability, and completeness compared to traditional RAG approaches.

## Overview

This project combines the power of graph databases with language models to create intelligent retrieval-augmented generation systems, following the methodologies outlined in the Essential GraphRAG book.

## Features

- **Neo4j Graph Database**: Store and query knowledge graphs efficiently
- **Hybrid Search**: Combines vector similarity search with traditional keyword search
- **Vector Embeddings**: Support for semantic search using sentence-transformers
- **Document Processing**: Parse and process various document formats (PDF, HTML, etc.)
- **GraphRAG Implementation**: Complete implementation following the book's methodologies

## Chapter Overview

### Chapter 2: Basic RAG with Vector Search and Hybrid Search
**File**: `graphrag_book/ch02.py`

Implements a foundational RAG (Retrieval-Augmented Generation) system that demonstrates:
- **Document Processing**: Downloads and chunks PDF documents (Einstein's patents paper)
- **Vector Embeddings**: Creates semantic embeddings using sentence-transformers (`all-MiniLM-L12-v2`)
- **Neo4j Storage**: Stores text chunks and embeddings in a graph database
- **Vector Search**: Implements semantic similarity search using vector indices
- **Full-Text Search**: Creates keyword-based search capabilities
- **Hybrid Search**: Combines vector and keyword search with score normalization
- **Answer Generation**: Uses GPT-4o-mini to generate answers from retrieved context

This chapter establishes the baseline RAG pipeline that subsequent chapters build upon.

### Chapter 3: Parent Document Retrieval with Step-Back Prompting
**File**: `graphrag_book/ch03.py`

Advances the RAG system with sophisticated retrieval strategies:
- **Parent-Child Architecture**: Splits documents into hierarchical chunks (large parent chunks containing smaller child chunks)
- **Section-Based Splitting**: Uses regex patterns to split documents by titles and sections for better contextual boundaries
- **Parent-Document Retrieval**: Returns larger parent chunks containing the matched children for comprehensive context
- **Child-Level Search**: Performs vector search on smaller, focused child chunks for precision
- **Parent-Document Retrieval**: Returns larger parent chunks containing the matched children for comprehensive context
- **Step-Back Prompting**: Implements query reformulation technique to generate more generic, easier-to-answer questions

This approach addresses the common RAG problem of retrieving relevant but insufficient context by ensuring complete sections are returned.

### Chapter 4: Text-to-Cypher with Schema-Aware Prompt Engineering
**File**: `graphrag_book/ch04.py`

Introduces graph database querying capabilities using natural language:
- **Movie Database Setup**: Creates a comprehensive movie graph database using predefined Cypher queries
- **Schema Introspection**: Automatically extracts and utilizes graph database schema for accurate query generation
- **Text-to-Cypher Generation**: Converts natural language questions into valid Cypher queries using LLM

This chapter establishes the foundation for natural language querying of knowledge graphs, enabling users to interact with complex graph data using conversational interfaces.

### Chapter 5: Agentic RAG with Multi-Tool Reasoning
**File**: `graphrag_book/ch05.py`

Implements an advanced agentic RAG system with autonomous reasoning capabilities:
- **Query Reformulation**: Automatically updates questions to be more atomic, specific, and easier to answer using contextual information
- **Tool Routing**: Intelligently selects appropriate tools based on question type and context using GPT-4o
- **Specialized Tools**: Dedicated functions for movie queries by title, actor search, general text-to-Cypher fallback, and answer extraction
- **Answer Critique**: Evaluates response completeness and identifies missing information for follow-up queries

This chapter represents a significant evolution from simple retrieval to autonomous agent behavior, enabling the system to break down complex questions, gather information iteratively, and provide comprehensive answers through intelligent tool orchestration.

### Chapter 6: Structured Information Extraction and Legal Contract Knowledge Graphs
**File**: `graphrag_book/ch06.py`

Demonstrates structured information extraction from legal documents and knowledge graph construction:
- **Structured Output**: Uses OpenAI's structured output feature with Pydantic models to extract contract information reliably
- **Legal Document Processing**: Processes complex legal contracts to extract parties, dates, terms, obligations, and jurisdictions
- **Entity Relationship Modeling**: Defines comprehensive schemas for contracts, organizations, and locations with proper relationships
- **Knowledge Graph Construction**: Creates detailed contract knowledge graphs in Neo4j with proper constraints and relationships
- **Legal Data Querying**: Implements querying capabilities for contract databases with type-specific searches

This chapter showcases how to handle complex, domain-specific documents (legal contracts) by combining structured AI extraction with graph database storage, enabling sophisticated querying and analysis of legal document collections.

### Chapter 7: MS GraphRAG Implementation
**File**: `graphrag_book/ch07.py`

Demonstrates large-scale MS GraphRAG implementation.


## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Set up environment variables (create a `.env` file):
   ```
   OPENAI_API_KEY=your_openai_api_key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   ```

## Usage

Run the Chapter 2 example (Einstein's Patents and Inventions):
```bash
make run-ch02
```

Run the Chapter 3 example (Parent Document Retrieval):
```bash
make run-ch03
```

Run the Chapter 4 example (Text-to-Cypher Query Generation):
```bash
make run-ch04
```

Run the Chapter 5 example (Agentic RAG with Multi-Tool Reasoning):
```bash
make run-ch05
```

Run the Chapter 6 example (Structured Information Extraction and Legal Contract Knowledge Graphs):
```bash
make run-ch06
```

## Project Structure

```
graphrag_book/
├── graphrag_book/
│   ├── __init__.py
│   ├── ch02.py              # Chapter 2: Basic RAG with Vector Search and Hybrid Search
│   ├── ch03.py              # Chapter 3: Parent Document Retrieval with Step-Back Prompting
│   ├── ch04.py              # Chapter 4: Text-to-Cypher with Schema-Aware Prompt Engineering
│   ├── ch05.py              # Chapter 5: Agentic RAG with Multi-Tool Reasoning
│   ├── ch06.py              # Chapter 6: Structured Information Extraction and Legal Contract Knowledge Graphs
│   ├── ch07.py              # Chapter 7: Large-Scale GraphRAG from Texts
│   ├── utils.py             # Utility functions for Neo4j and common operations
│   ├── schema_utils.py      # Schema introspection and chat utilities
│   └── cypher_queries.py    # Predefined Cypher queries for database setup
├── makefile                 # Commands to run chapter examples
├── pyproject.toml          # Project dependencies and configuration
├── uv.lock                 # Dependency lock file
├── README.md
└── .gitignore
```

## Development

Install development dependencies:
```bash
uv sync
```

## Book Reference

This implementation is based on **"Essential GraphRAG: Knowledge Graph-Enhanced RAG"** by Tomaž Bratanič and Oskar Hane, published by Manning Publications. 

- **Book URL**: https://www.manning.com/books/essential-graphrag
- **Authors**: Tomaž Bratanič and Oskar Hane
- **Publisher**: Manning Publications
