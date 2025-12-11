# Agentic RAG Notebooks

A collection of advanced Retrieval-Augmented Generation (RAG) implementations with intelligent agents using LangChain, LangGraph, and various vector store backends.

## 📚 Project Overview

This repository demonstrates different approaches to building RAG systems with autonomous agents. Each notebook showcases a distinct implementation strategy, from basic vector search to hybrid search and cloud-based vector stores.

---

## 📔 Notebooks

### 1. Agentic RAG (FAISS Vector Store)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Saad-Shakeel/Agentic-RAG-Notebooks/blob/main/Agentic_Rag.ipynb)

**Description:**
A foundational RAG system combining document retrieval with an intelligent agent. This notebook demonstrates:
- **Document Loading & Processing**: Load PDF documents and split them into manageable chunks using recursive character splitting
- **Vector Store Setup**: Create a FAISS (Facebook AI Similarity Search) vector database for fast semantic similarity search
- **RAG Tool Implementation**: Define a retrieval tool that fetches relevant document context
- **LangGraph Agent**: Build an agentic workflow that decides when to retrieve documents and generates contextual responses
- **System-Guided Responses**: Uses system prompts to ensure answers are based on provided documents with proper citations

**Key Components:**
- LLM: ChatGroq (GPT OSS 120B)
- Embeddings: OllamaEmbeddings (nomic-embed-text:v1.5)
- Vector Store: FAISS
- Framework: LangChain + LangGraph

---

### 2. Agentic RAG - Hybrid Search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Saad-Shakeel/Agentic-RAG-Notebooks/blob/main/Agentic_RAG_Hybrid_Search.ipynb)

**Description:**
An advanced RAG system that combines dense (semantic) and sparse (keyword) search for comprehensive document retrieval. This notebook showcases:
- **Hybrid Search Strategy**: Combines semantic embeddings with BM25-based keyword search for better recall
- **Pinecone Integration**: Leverages Pinecone cloud vector database for scalable vector operations
- **BM25 Encoder**: Trains TF-IDF-based sparse vectors for keyword matching
- **Dense Embeddings**: Uses Google Gemini embeddings for semantic understanding (3072 dimensions)
- **LangGraph Workflow**: Implements multi-step agent logic with both retrieval and response generation
- **Production-Ready**: Demonstrates enterprise-grade RAG architecture

**Key Components:**
- LLM: ChatGroq (GPT OSS 120B)
- Dense Embeddings: GoogleGenerativeAIEmbeddings (Gemini)
- Sparse Search: BM25Encoder (Pinecone)
- Vector Store: Pinecone (Cloud-based)
- Framework: LangChain + LangGraph

---

### 3. Agentic RAG - MongoDB Vector Store

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Saad-Shakeel/Agentic-RAG-Notebooks/blob/main/Agentic_RAG_with_MongoDB_VectorStore.ipynb)

**Description:**
A cloud-native RAG implementation using MongoDB Atlas for vector storage and retrieval. This notebook demonstrates:
- **MongoDB Atlas Setup**: Connect to MongoDB Atlas cluster for managed vector storage
- **Vector Search Index**: Create and manage vector search indices in MongoDB (768-dimensional vectors)
- **Cloud Document Storage**: Store both documents and their embeddings in MongoDB
- **Similarity Search**: Leverage MongoDB's native similarity search capabilities
- **ReAct Agent Pattern**: Implements the Reasoning + Acting agent pattern for dynamic tool usage
- **Multi-turn Conversations**: Uses MemorySaver for maintaining conversation state across multiple turns
- **Scalable Architecture**: Built for production deployment with proper checkpointing

**Key Components:**
- LLM: ChatGroq (GPT OSS 120B)
- Embeddings: OllamaEmbeddings (nomic-embed-text:v1.5, 768 dimensions)
- Vector Store: MongoDB Atlas
- Database: MongoDB Atlas Cloud
- Framework: LangChain (create_agent utility)

---

### 4. Agentic Knowledge Graph RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Saad-Shakeel/Agentic-RAG-Notebooks/blob/main/Agentic_Knowledge_Graph_RAG.ipynb)

**Description:**
An advanced RAG system that combines knowledge graphs with vector search for enhanced reasoning and relationship discovery. This notebook demonstrates:
- **Neo4j Graph Database**: Connect to Neo4j for storing and querying structured knowledge relationships
- **Graph Data Loading**: Import CSV data to create Article, Researcher, and Topic nodes with relationships
- **Hybrid Vector Index**: Combine semantic embeddings with graph structure for comprehensive search
- **Dual Tool Architecture**: Separate tools for semantic retrieval and graph-based queries
- **Cypher Query Generation**: Automatic translation of natural language to Cypher queries
- **ReAct Agent Pattern**: Intelligent tool selection between vector search and graph queries
- **Relationship Analysis**: Discover connections between entities, authors, and research topics

**Key Components:**
- LLM: ChatGroq (GPT OSS 20B)
- Embeddings: GoogleGenerativeAIEmbeddings (Gemini, 3072 dimensions)
- Graph Database: Neo4j
- Vector Store: Neo4j Vector Index (hybrid search)
- Framework: LangChain + Neo4j Integration

---

## 📋 Prerequisites

- Python 3.9+
- API Keys:
  - `GROQ_API_KEY` - For ChatGroq LLM
  - `GOOGLE_API_KEY` - For Google Gemini embeddings (Hybrid Search & Knowledge Graph notebooks)
  - `PINECONE_API_KEY` - For Pinecone vector database (Hybrid Search notebook)
  - `MONGODB_ATLAS_CLUSTER_URI` - For MongoDB Atlas connection (MongoDB notebook)
  - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` - For Neo4j graph database (Knowledge Graph notebook)
- Installed Dependencies (see `pyproject.toml`)

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Saad-Shakeel/Agentic-RAG-Notebooks.git
cd Agentic-RAG-Notebooks

# Install dependencies
uv sync
```

---

## 📝 Environment Setup

1. **Rename the environment template file:**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   ```

2. **Add your API keys to the `.env` file:**
   ```env
   # Groq API - Get from https://console.groq.com
   GROQ_API_KEY=your_groq_api_key
   
   # Google Gemini API - Get from https://makersuite.google.com/app/apikey
   GOOGLE_API_KEY=your_google_api_key
   
   # Pinecone API - Get from https://app.pinecone.io
   PINECONE_API_KEY=your_pinecone_api_key
   
   # MongoDB Atlas Connection - Get from https://cloud.mongodb.com
   MONGODB_ATLAS_CLUSTER_URI=mongodb+srv://username:password@cluster.mongodb.net/database
   
   # Neo4j Graph Database - Get from https://neo4j.com/cloud/aura/
   NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   ```

3. **Place your PDF documents in the `data/` directory**

---

## 🎯 Quick Start

### Running Locally
```bash
# Start Jupyter
jupyter notebook

# Open desired notebook and run cells sequentially
```

### Running on Google Colab
Click the "Open in Colab" badge at the top of each notebook to run directly in your browser without local setup.

---
## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Pinecone Vector Database](https://www.pinecone.io/)
- [MongoDB Atlas](https://www.mongodb.com/products/platform/atlas)
- [Neo4j Graph Database](https://neo4j.com/)
- [Groq API](https://www.groq.com/)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## ⭐ Star History

If you find this repository helpful, please consider giving it a star! Your support motivates continued development.

---

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the maintainer.

---
