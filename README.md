# GraphRAG Book

A GraphRAG (Graph Retrieval-Augmented Generation) project using LangChain and Neo4j.

## Overview

This project combines the power of graph databases with language models to create intelligent retrieval-augmented generation systems.

## Features

- **LangChain Integration**: Leverage LangChain's comprehensive ecosystem for LLM operations
- **Neo4j Graph Database**: Store and query knowledge graphs efficiently
- **Vector Embeddings**: Support for semantic search using ChromaDB and sentence-transformers
- **Document Processing**: Parse and process various document formats (PDF, HTML, etc.)

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

[Add usage examples here as you develop the project]

## Development

Install development dependencies:
```bash
uv sync --extra dev
```

Run tests:
```bash
pytest
```

Format code:
```bash
black .
isort .
```

## License

[Add your license here] 