# GraphRAG Book

A GraphRAG (Graph Retrieval-Augmented Generation) project using LangChain and Neo4j.

This repository contains the implementation examples and code from **"Essential GraphRAG: Knowledge Graph-Enhanced RAG"** by Tomaž Bratanič and Oskar Hane, published by [Manning Publications](https://www.manning.com/books/essential-graphrag).

## About the Book

**Essential GraphRAG** teaches you to implement accurate, performant, and traceable RAG by structuring the context data as a knowledge graph. The book covers how to upgrade your RAG applications with the power of knowledge graphs, delivering better performance, accuracy, traceability, and completeness compared to traditional RAG approaches.

## Overview

This project combines the power of graph databases with language models to create intelligent retrieval-augmented generation systems, following the methodologies outlined in the Essential GraphRAG book.

## Features

- **Neo4j Graph Database**: Store and query knowledge graphs efficiently
- **Hybrid Search**: Combines vector similarity search with traditional keyword search
- **Vector Embeddings**: Support for semantic search using sentence-transformers
- **Document Processing**: Parse and process various document formats (PDF, HTML, etc.)
- **GraphRAG Implementation**: Complete implementation following the book's methodologies

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

This will:
1. Download and process a PDF document about Einstein's patents
2. Create vector embeddings using sentence-transformers
3. Store the data in Neo4j with vector and full-text indexes
4. Perform hybrid search combining semantic and keyword matching
5. Generate AI-powered answers using retrieved context

## Project Structure

```
graphrag_book/
├── graphrag_book/
│   ├── __init__.py
│   ├── ch02.py          # Chapter 2 implementation
│   └── utils.py         # Utility functions
├── pyproject.toml       # Project dependencies
├── README.md
└── .gitignore
```

## Development

Install development dependencies:
```bash
uv sync --extra dev
```

## Book Reference

This implementation is based on **"Essential GraphRAG: Knowledge Graph-Enhanced RAG"** by Tomaž Bratanič and Oskar Hane, published by Manning Publications. 

- **Book URL**: https://www.manning.com/books/essential-graphrag
- **Authors**: Tomaž Bratanič and Oskar Hane
- **Publisher**: Manning Publications
- **Publication**: September 2025 (estimated)
