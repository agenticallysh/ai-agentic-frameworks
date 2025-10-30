#!/usr/bin/env python3
"""
Langflow RAG Pipeline Example

This example demonstrates how to build a document-based question-answering system
using Langflow components. The pipeline includes document loading, text splitting,
embedding generation, vector storage, and retrieval-augmented generation.

Requirements:
    pip install langflow openai chromadb
    
Usage:
    python rag-pipeline.py
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List

from langflow.components import (
    DocumentLoader, TextSplitter, OpenAIEmbeddings,
    ChromaVectorStore, VectorRetriever, PromptTemplate,
    OpenAIModel, ChatInput, ChatOutput
)
from langflow.graph import Graph


class RAGPipeline:
    """
    A comprehensive RAG (Retrieval-Augmented Generation) pipeline using Langflow.
    
    This pipeline:
    1. Loads documents from a directory
    2. Splits them into chunks
    3. Creates embeddings
    4. Stores them in a vector database
    5. Retrieves relevant context for queries
    6. Generates answers using an LLM
    """
    
    def __init__(
        self,
        document_path: str = "./documents",
        collection_name: str = "knowledge_base",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model: str = "gpt-4"
    ):
        self.document_path = document_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.graph = None
        self._setup_components()
    
    def _setup_components(self):
        """Initialize and configure all pipeline components."""
        
        # Document processing components
        self.document_loader = DocumentLoader(
            path=self.document_path,
            file_types=[".txt", ".md", ".pdf", ".docx"],
            recursive=True
        )
        
        self.text_splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n\n"
        )
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            chunk_size=1000
        )
        
        # Vector storage and retrieval
        self.vector_store = ChromaVectorStore(
            collection_name=self.collection_name,
            persist_directory="./chroma_db"
        )
        
        self.retriever = VectorRetriever(
            vector_store=self.vector_store,
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # LLM components
        self.prompt_template = PromptTemplate(
            template="""You are a helpful AI assistant that answers questions based on the provided context.
Use only the information from the context to answer the question. If the answer is not in the context,
say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Provide a comprehensive and accurate answer based on the context above.

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.llm = OpenAIModel(
            model=self.model,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Chat interface components
        self.chat_input = ChatInput()
        self.chat_output = ChatOutput()
    
    def build_graph(self) -> Graph:
        """Build the complete RAG pipeline graph."""
        
        self.graph = Graph(name="RAG Pipeline")
        
        # Add components to graph
        components = {
            "loader": self.document_loader,
            "splitter": self.text_splitter,
            "embeddings": self.embeddings,
            "vector_store": self.vector_store,
            "retriever": self.retriever,
            "prompt": self.prompt_template,
            "llm": self.llm,
            "input": self.chat_input,
            "output": self.chat_output
        }
        
        for name, component in components.items():
            self.graph.add_component(name, component)
        
        # Connect components
        self._connect_components()
        
        return self.graph
    
    def _connect_components(self):
        """Define the flow connections between components."""
        
        # Document processing pipeline
        self.graph.connect("loader", "output", "splitter", "documents")
        self.graph.connect("splitter", "chunks", "embeddings", "texts")
        self.graph.connect("embeddings", "embeddings", "vector_store", "embeddings")
        
        # Query processing pipeline
        self.graph.connect("input", "message", "retriever", "query")
        self.graph.connect("vector_store", "store", "retriever", "vector_store")
        self.graph.connect("retriever", "documents", "prompt", "context")
        self.graph.connect("input", "message", "prompt", "question")
        self.graph.connect("prompt", "prompt", "llm", "prompt")
        self.graph.connect("llm", "response", "output", "message")
    
    async def index_documents(self) -> Dict[str, Any]:
        """Index documents into the vector store."""
        
        print(f"Loading documents from {self.document_path}...")
        documents = await self.document_loader.arun()
        
        if not documents:
            raise ValueError(f"No documents found in {self.document_path}")
        
        print(f"Loaded {len(documents)} documents")
        
        print("Splitting documents into chunks...")
        chunks = await self.text_splitter.arun(documents=documents)
        print(f"Created {len(chunks)} chunks")
        
        print("Generating embeddings...")
        embeddings = await self.embeddings.arun(texts=chunks)
        
        print("Storing in vector database...")
        result = await self.vector_store.arun(
            texts=chunks,
            embeddings=embeddings
        )
        
        print("✅ Document indexing complete!")
        return result
    
    async def query(self, question: str) -> str:
        """Query the RAG pipeline with a question."""
        
        if not self.graph:
            self.build_graph()
        
        print(f"🔍 Processing query: {question}")
        
        # Retrieve relevant documents
        print("Retrieving relevant context...")
        retrieved_docs = await self.retriever.arun(query=question)
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate prompt
        prompt = await self.prompt_template.arun(
            context=context,
            question=question
        )
        
        # Generate response
        print("Generating response...")
        response = await self.llm.arun(prompt=prompt)
        
        return response
    
    async def interactive_chat(self):
        """Start an interactive chat session."""
        
        print("\n🤖 RAG Pipeline Chat Interface")
        print("Type 'quit' to exit, 'index' to reindex documents")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n❓ You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'index':
                    await self.index_documents()
                    continue
                
                if not user_input:
                    continue
                
                # Get response
                response = await self.query(user_input)
                print(f"\n🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


class DocumentSetup:
    """Helper class to set up sample documents for testing."""
    
    @staticmethod
    def create_sample_documents(docs_dir: str = "./documents"):
        """Create sample documents for testing the RAG pipeline."""
        
        docs_path = Path(docs_dir)
        docs_path.mkdir(exist_ok=True)
        
        sample_docs = {
            "ai_basics.txt": """
Artificial Intelligence (AI) is a branch of computer science that aims to create
intelligent machines that work and react like humans. AI systems can perform tasks
that typically require human intelligence, such as visual perception, speech
recognition, decision-making, and language translation.

There are three main types of AI:
1. Narrow AI (ANI) - AI that is designed to perform a narrow task
2. General AI (AGI) - AI that has generalized human cognitive abilities
3. Super AI (ASI) - AI that surpasses human intelligence in all areas

Machine Learning is a subset of AI that provides systems the ability to
automatically learn and improve from experience without being explicitly programmed.
            """,
            
            "langflow_info.txt": """
Langflow is a visual AI agent builder that allows users to create complex AI
workflows using a drag-and-drop interface. It provides a bridge between no-code
simplicity and developer flexibility.

Key features of Langflow include:
- Visual flow builder with intuitive interface
- Python customization capabilities
- Support for all major LLMs and vector databases
- Automatic API generation from visual flows
- Model Context Protocol (MCP) integration
- Multi-modal support for text, images, and documents

Langflow is ideal for rapid prototyping, team collaboration, and educational purposes.
It's particularly useful for building RAG applications and multi-agent systems.
            """,
            
            "rag_concepts.txt": """
Retrieval-Augmented Generation (RAG) is a technique that combines information
retrieval with text generation to create more accurate and contextual AI responses.

The RAG process involves:
1. Document Ingestion - Loading and processing documents
2. Chunking - Breaking documents into manageable pieces
3. Embedding - Converting text chunks into vector representations
4. Storage - Storing embeddings in a vector database
5. Retrieval - Finding relevant chunks for a given query
6. Generation - Using retrieved context to generate responses

Benefits of RAG:
- Reduces hallucinations in AI responses
- Provides up-to-date information
- Allows for source attribution
- Enables domain-specific knowledge integration
            """
        }
        
        for filename, content in sample_docs.items():
            file_path = docs_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
        
        print(f"✅ Created {len(sample_docs)} sample documents in {docs_dir}")


async def main():
    """Main function to demonstrate the RAG pipeline."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        return
    
    # Create sample documents
    print("📚 Setting up sample documents...")
    DocumentSetup.create_sample_documents()
    
    # Initialize RAG pipeline
    print("\n🚀 Initializing RAG Pipeline...")
    rag = RAGPipeline(
        document_path="./documents",
        collection_name="demo_knowledge_base",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Build the graph
    graph = rag.build_graph()
    print(f"✅ Built graph with {len(graph.components)} components")
    
    # Index documents
    print("\n📊 Indexing documents...")
    await rag.index_documents()
    
    # Example queries
    example_queries = [
        "What is Artificial Intelligence?",
        "What are the main features of Langflow?",
        "How does RAG work?",
        "What are the benefits of using RAG?"
    ]
    
    print("\n🔄 Running example queries...")
    for query in example_queries:
        print(f"\n{'='*60}")
        response = await rag.query(query)
        print(f"Q: {query}")
        print(f"A: {response}")
    
    # Start interactive chat
    print(f"\n{'='*60}")
    await rag.interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())