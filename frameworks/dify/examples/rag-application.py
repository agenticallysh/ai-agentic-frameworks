#!/usr/bin/env python3
"""
RAG Application Example - Dify
===============================

This example demonstrates how to build a Retrieval-Augmented Generation (RAG)
application using Dify's Knowledge Base and Retrieval APIs. The application
processes documents, creates a knowledge base, and answers questions based
on the retrieved context.

Requirements:
    pip install requests python-dotenv pypdf2 python-docx

Usage:
    python rag-application.py
"""

import os
import json
import time
import uuid
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import requests


@dataclass
class Document:
    """Represents a document in the knowledge base."""
    id: str
    name: str
    content: str
    doc_type: str
    word_count: int
    tokens: int
    status: str = "processing"
    created_at: str = None
    updated_at: str = None


class DifyRAGApplication:
    """
    A comprehensive RAG application using Dify's Knowledge Base APIs.
    
    This class provides functionality for:
    - Creating and managing knowledge bases
    - Uploading and processing documents
    - Performing retrieval-augmented generation
    - Managing document lifecycle
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.dify.ai/v1",
        dataset_id: str = None
    ):
        """
        Initialize the Dify RAG application.
        
        Args:
            api_key: Your Dify API key
            base_url: Dify API base URL
            dataset_id: Existing dataset ID (if None, will create new)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.dataset_id = dataset_id
        self.documents: Dict[str, Document] = {}
        
        # HTTP session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}'
        })
    
    def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        indexing_technique: str = "high_quality",
        embedding_model: str = "text-embedding-ada-002",
        embedding_model_provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Create a new knowledge base (dataset) in Dify.
        
        Args:
            name: Name of the knowledge base
            description: Description of the knowledge base
            indexing_technique: "high_quality" or "economy"
            embedding_model: Embedding model to use
            embedding_model_provider: Provider for embedding model
            
        Returns:
            Dictionary containing the created dataset information
        """
        
        payload = {
            "name": name,
            "description": description,
            "indexing_technique": indexing_technique,
            "embedding_model": embedding_model,
            "embedding_model_provider": embedding_model_provider,
            "retrieval_model": {
                "search_method": "semantic_search",
                "reranking_enable": True,
                "reranking_model": {
                    "reranking_provider_name": "cohere",
                    "reranking_model_name": "rerank-english-v2.0"
                },
                "top_k": 5,
                "score_threshold_enabled": True,
                "score_threshold": 0.5
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/datasets",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            self.dataset_id = result.get('id')
            
            print(f"✅ Created knowledge base: {name}")
            print(f"   Dataset ID: {self.dataset_id}")
            print(f"   Indexing: {indexing_technique}")
            print(f"   Embedding: {embedding_model}")
            
            return result
            
        except requests.RequestException as e:
            print(f"❌ Error creating knowledge base: {e}")
            return {}
    
    def upload_document(
        self,
        file_path: str,
        document_name: str = None,
        indexing_technique: str = "high_quality",
        process_rule: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Upload a document to the knowledge base.
        
        Args:
            file_path: Path to the document file
            document_name: Custom name for the document
            indexing_technique: Processing technique
            process_rule: Custom processing rules
            
        Returns:
            Dictionary containing upload result
        """
        
        if not self.dataset_id:
            raise ValueError("No dataset_id. Create a knowledge base first.")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = 'application/octet-stream'
        
        # Prepare file for upload
        files = {
            'file': (file_path.name, open(file_path, 'rb'), mime_type)
        }
        
        # Prepare form data
        data = {
            'indexing_technique': indexing_technique,
            'duplicate_check': 'true'
        }
        
        if document_name:
            data['name'] = document_name
        
        if process_rule:
            data['process_rule'] = json.dumps(process_rule)
        else:
            # Default processing rules
            data['process_rule'] = json.dumps({
                "rules": {
                    "pre_processing_rules": [
                        {"id": "remove_extra_spaces", "enabled": True},
                        {"id": "remove_urls_emails", "enabled": True}
                    ],
                    "segmentation": {
                        "separator": "\\n",
                        "max_tokens": 1000,
                        "chunk_overlap": 100
                    }
                },
                "mode": "automatic"
            })
        
        try:
            # Remove Content-Type header for multipart upload
            headers = self.session.headers.copy()
            if 'Content-Type' in headers:
                del headers['Content-Type']
            
            response = self.session.post(
                f"{self.base_url}/datasets/{self.dataset_id}/document/create_by_file",
                files=files,
                data=data,
                headers=headers,
                timeout=300  # 5 minutes for large files
            )
            response.raise_for_status()
            
            result = response.json()
            document_id = result.get('document', {}).get('id')
            
            # Track the document
            doc = Document(
                id=document_id,
                name=document_name or file_path.name,
                content="",  # Will be filled later
                doc_type=file_path.suffix,
                word_count=0,
                tokens=0,
                status="processing"
            )
            self.documents[document_id] = doc
            
            print(f"✅ Document uploaded: {doc.name}")
            print(f"   Document ID: {document_id}")
            print(f"   Status: Processing...")
            
            return result
            
        except requests.RequestException as e:
            print(f"❌ Error uploading document: {e}")
            return {}
        finally:
            # Close the file
            for file_obj in files.values():
                if hasattr(file_obj[1], 'close'):
                    file_obj[1].close()
    
    def upload_text_document(
        self,
        content: str,
        name: str,
        indexing_technique: str = "high_quality"
    ) -> Dict[str, Any]:
        """
        Upload text content as a document.
        
        Args:
            content: Text content to upload
            name: Name for the document
            indexing_technique: Processing technique
            
        Returns:
            Dictionary containing upload result
        """
        
        if not self.dataset_id:
            raise ValueError("No dataset_id. Create a knowledge base first.")
        
        payload = {
            "name": name,
            "text": content,
            "indexing_technique": indexing_technique,
            "process_rule": {
                "rules": {
                    "pre_processing_rules": [
                        {"id": "remove_extra_spaces", "enabled": True},
                        {"id": "remove_urls_emails", "enabled": True}
                    ],
                    "segmentation": {
                        "separator": "\\n",
                        "max_tokens": 1000,
                        "chunk_overlap": 100
                    }
                },
                "mode": "automatic"
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/datasets/{self.dataset_id}/document/create_by_text",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            document_id = result.get('document', {}).get('id')
            
            # Track the document
            doc = Document(
                id=document_id,
                name=name,
                content=content[:200] + "..." if len(content) > 200 else content,
                doc_type="text",
                word_count=len(content.split()),
                tokens=0,
                status="processing"
            )
            self.documents[document_id] = doc
            
            print(f"✅ Text document created: {name}")
            print(f"   Document ID: {document_id}")
            print(f"   Word count: {doc.word_count}")
            
            return result
            
        except requests.RequestException as e:
            print(f"❌ Error creating text document: {e}")
            return {}
    
    def check_document_status(self, document_id: str) -> Dict[str, Any]:
        """
        Check the processing status of a document.
        
        Args:
            document_id: ID of the document to check
            
        Returns:
            Dictionary containing document status
        """
        
        try:
            response = self.session.get(
                f"{self.base_url}/datasets/{self.dataset_id}/documents/{document_id}",
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Update tracked document
            if document_id in self.documents:
                doc = self.documents[document_id]
                doc.status = result.get('indexing_status', 'unknown')
                doc.word_count = result.get('word_count', 0)
                doc.tokens = result.get('tokens', 0)
            
            return result
            
        except requests.RequestException as e:
            print(f"❌ Error checking document status: {e}")
            return {}
    
    def wait_for_processing(self, document_ids: List[str], timeout: int = 300) -> bool:
        """
        Wait for documents to finish processing.
        
        Args:
            document_ids: List of document IDs to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all documents processed successfully
        """
        
        start_time = time.time()
        remaining_docs = set(document_ids)
        
        print(f"⏳ Waiting for {len(document_ids)} documents to process...")
        
        while remaining_docs and (time.time() - start_time) < timeout:
            completed_this_round = set()
            
            for doc_id in remaining_docs:
                status_info = self.check_document_status(doc_id)
                status = status_info.get('indexing_status', 'processing')
                
                if status == 'completed':
                    doc_name = self.documents.get(doc_id, {}).name or doc_id
                    print(f"   ✅ {doc_name} - Processing completed")
                    completed_this_round.add(doc_id)
                elif status == 'error':
                    doc_name = self.documents.get(doc_id, {}).name or doc_id
                    print(f"   ❌ {doc_name} - Processing failed")
                    completed_this_round.add(doc_id)
            
            remaining_docs -= completed_this_round
            
            if remaining_docs:
                time.sleep(5)  # Check every 5 seconds
        
        if remaining_docs:
            print(f"⚠️ Timeout: {len(remaining_docs)} documents still processing")
            return False
        
        print("✅ All documents processed successfully!")
        return True
    
    def query_knowledge_base(
        self,
        query: str,
        retrieval_mode: str = "single",
        top_k: int = 5,
        score_threshold: float = 0.5,
        reranking_enable: bool = True
    ) -> Dict[str, Any]:
        """
        Query the knowledge base for relevant information.
        
        Args:
            query: The question or search query
            retrieval_mode: "single" or "multiple"
            top_k: Number of top results to return
            score_threshold: Minimum relevance score
            reranking_enable: Whether to use reranking
            
        Returns:
            Dictionary containing retrieval results
        """
        
        if not self.dataset_id:
            raise ValueError("No dataset_id. Create a knowledge base first.")
        
        payload = {
            "query": query,
            "retrieval_model": {
                "search_method": "semantic_search",
                "reranking_enable": reranking_enable,
                "reranking_model": {
                    "reranking_provider_name": "cohere",
                    "reranking_model_name": "rerank-english-v2.0"
                } if reranking_enable else {},
                "top_k": top_k,
                "score_threshold_enabled": True,
                "score_threshold": score_threshold
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/datasets/{self.dataset_id}/retrieve",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            return result
            
        except requests.RequestException as e:
            print(f"❌ Error querying knowledge base: {e}")
            return {}
    
    def create_rag_app(
        self,
        app_name: str,
        system_prompt: str = None,
        model_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a RAG application that uses the knowledge base.
        
        Args:
            app_name: Name for the RAG application
            system_prompt: Custom system prompt
            model_config: LLM model configuration
            
        Returns:
            Dictionary containing app creation result
        """
        
        if not self.dataset_id:
            raise ValueError("No dataset_id. Create a knowledge base first.")
        
        if not system_prompt:
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from the knowledge base.

Instructions:
1. Use only the information provided in the context to answer questions
2. If the answer is not in the context, say "I don't have enough information to answer that question"
3. Provide accurate, helpful, and detailed responses
4. When possible, cite specific parts of the context
5. If the question is unclear, ask for clarification

Context: {{#context#}}{{content}}{{/context#}}

Question: {{#sys.query#}}

Answer:"""
        
        if not model_config:
            model_config = {
                "provider": "openai",
                "model": "gpt-4",
                "parameters": {
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            }
        
        app_config = {
            "name": app_name,
            "mode": "chat",
            "icon": "📚",
            "description": f"RAG application powered by knowledge base",
            "model_config": model_config,
            "prompt_template": system_prompt,
            "datasets": [{
                "dataset": {
                    "id": self.dataset_id
                },
                "retrieval_model": {
                    "search_method": "semantic_search",
                    "reranking_enable": True,
                    "reranking_model": {
                        "reranking_provider_name": "cohere",
                        "reranking_model_name": "rerank-english-v2.0"
                    },
                    "top_k": 5,
                    "score_threshold_enabled": True,
                    "score_threshold": 0.5
                }
            }],
            "opening_statement": "Hello! I'm your AI assistant powered by the knowledge base. Ask me anything about the documents that have been uploaded.",
            "suggested_questions": [
                "What are the main topics covered in the documents?",
                "Can you summarize the key points?",
                "What specific information is available about [topic]?",
                "Are there any recommendations or best practices mentioned?"
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/apps",
                json=app_config,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            app_id = result.get('id')
            
            print(f"✅ RAG application created: {app_name}")
            print(f"   App ID: {app_id}")
            print(f"   Knowledge base: {self.dataset_id}")
            
            return result
            
        except requests.RequestException as e:
            print(f"❌ Error creating RAG app: {e}")
            return {}
    
    def ask_question(
        self,
        question: str,
        app_id: str = None,
        conversation_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Ask a question using the RAG application.
        
        Args:
            question: The question to ask
            app_id: ID of the RAG application
            conversation_id: Existing conversation ID
            user_id: User identifier
            
        Returns:
            Dictionary containing the response
        """
        
        if not app_id:
            raise ValueError("app_id is required for asking questions")
        
        if not user_id:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        payload = {
            "inputs": {},
            "query": question,
            "response_mode": "blocking",
            "conversation_id": conversation_id,
            "user": user_id,
            "auto_generate_name": True
        }
        
        try:
            # Temporarily update headers for this request
            headers = self.session.headers.copy()
            headers['Content-Type'] = 'application/json'
            
            response = self.session.post(
                f"{self.base_url}/chat-messages",
                json=payload,
                headers=headers,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "answer": result.get('answer', 'No answer received'),
                "conversation_id": result.get('conversation_id'),
                "message_id": result.get('id'),
                "retrieval_info": result.get('metadata', {}).get('retriever_resources', []),
                "usage": result.get('metadata', {}).get('usage', {})
            }
            
        except requests.RequestException as e:
            print(f"❌ Error asking question: {e}")
            return {
                "answer": f"Error: {e}",
                "conversation_id": conversation_id,
                "message_id": None,
                "retrieval_info": [],
                "usage": {}
            }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the knowledge base."""
        
        if not self.dataset_id:
            return []
        
        try:
            response = self.session.get(
                f"{self.base_url}/datasets/{self.dataset_id}/documents",
                params={"limit": 100},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
            
        except requests.RequestException as e:
            print(f"❌ Error listing documents: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the knowledge base."""
        
        try:
            response = self.session.delete(
                f"{self.base_url}/datasets/{self.dataset_id}/documents/{document_id}",
                timeout=30
            )
            response.raise_for_status()
            
            # Remove from tracking
            if document_id in self.documents:
                del self.documents[document_id]
            
            print(f"✅ Document deleted: {document_id}")
            return True
            
        except requests.RequestException as e:
            print(f"❌ Error deleting document: {e}")
            return False
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the knowledge base."""
        
        if not self.dataset_id:
            return {}
        
        try:
            response = self.session.get(
                f"{self.base_url}/datasets/{self.dataset_id}",
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            print(f"❌ Error getting knowledge base info: {e}")
            return {}


def create_sample_documents():
    """Create sample documents for testing the RAG application."""
    
    docs_dir = Path("sample_documents")
    docs_dir.mkdir(exist_ok=True)
    
    sample_docs = {
        "ai_fundamentals.txt": """
# Artificial Intelligence Fundamentals

## Introduction
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The field of AI research was officially born in 1956 at the Dartmouth Conference.

## Types of AI
1. **Narrow AI (ANI)**: AI designed to perform a narrow task (e.g., facial recognition, internet searches, self-driving cars)
2. **General AI (AGI)**: AI that has generalized human cognitive abilities
3. **Super AI (ASI)**: AI that surpasses human intelligence in all areas

## Machine Learning
Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

### Types of Machine Learning:
- **Supervised Learning**: Learning with labeled examples
- **Unsupervised Learning**: Finding patterns in data without labels
- **Reinforcement Learning**: Learning through interaction with environment

## Applications
- Healthcare: Medical diagnosis, drug discovery
- Finance: Fraud detection, algorithmic trading
- Transportation: Autonomous vehicles, route optimization
- Entertainment: Content recommendation, game AI
- Business: Process automation, customer service
        """,
        
        "rag_systems.txt": """
# Retrieval-Augmented Generation (RAG) Systems

## Overview
RAG is a technique that combines information retrieval with text generation to provide more accurate and contextual AI responses. It addresses the limitation of large language models having knowledge cutoffs.

## How RAG Works
1. **Document Ingestion**: Load and process documents into a searchable format
2. **Chunking**: Break documents into smaller, manageable pieces
3. **Embedding**: Convert text chunks into vector representations
4. **Storage**: Store embeddings in a vector database
5. **Retrieval**: Find relevant chunks for a given query
6. **Generation**: Use retrieved context to generate responses

## Benefits
- **Accuracy**: Reduces hallucinations by grounding responses in real data
- **Timeliness**: Provides up-to-date information beyond training data
- **Transparency**: Can show source documents for answers
- **Domain-specific**: Enables specialized knowledge integration

## Implementation Components
- **Vector Databases**: Pinecone, Weaviate, Chroma, Qdrant
- **Embedding Models**: OpenAI Ada, Sentence Transformers, Cohere
- **Retrieval Methods**: Semantic search, keyword search, hybrid
- **Generation Models**: GPT-4, Claude, PaLM, Local models

## Best Practices
1. Choose appropriate chunk sizes (typically 500-1000 tokens)
2. Use overlapping chunks to maintain context
3. Implement reranking for better relevance
4. Monitor and evaluate retrieval quality
5. Handle edge cases (no relevant documents, conflicting information)
        """,
        
        "dify_platform.txt": """
# Dify Platform Guide

## What is Dify?
Dify is an open-source platform for developing and operating generative AI applications. It provides visual orchestration, comprehensive LLMOps capabilities, and production-ready deployment options.

## Key Features
- **Visual Agent Builder**: Drag-and-drop interface for creating AI workflows
- **LLMOps Platform**: Complete lifecycle management for LLM applications
- **RAG & Knowledge Base**: Built-in document processing and retrieval systems
- **Multi-Agent Workflows**: Orchestrate complex agent interactions
- **Observability**: Real-time monitoring, logging, and analytics
- **Cloud & Self-Hosted**: Deploy on Dify Cloud or your own infrastructure

## Use Cases
- **Chatbots**: Customer service, support agents
- **RAG Applications**: Document Q&A, knowledge management
- **Content Generation**: Writing assistants, content creation
- **Data Analysis**: Automated insights, report generation
- **Workflow Automation**: Business process automation

## Getting Started
1. **Dify Cloud**: Quick start with managed service
2. **Self-Hosted**: Deploy using Docker or Kubernetes
3. **API Integration**: Use APIs for custom applications

## Best Practices
- Design workflows with clear objectives
- Use appropriate models for each task
- Implement proper error handling
- Monitor performance and costs
- Test thoroughly before production deployment

## Enterprise Features
- Multi-tenant architecture
- Role-based access control
- Custom branding
- SLA guarantees
- Dedicated support
        """
    }
    
    for filename, content in sample_docs.items():
        file_path = docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    print(f"✅ Created {len(sample_docs)} sample documents in {docs_dir}")
    return docs_dir


def main():
    """Main function demonstrating RAG application usage."""
    
    print("📚 Dify RAG Application Example")
    print("=" * 40)
    
    # Get API key
    api_key = os.getenv('DIFY_API_KEY')
    if not api_key:
        api_key = input("Enter your Dify API key: ").strip()
    
    if not api_key:
        print("❌ API key is required")
        return
    
    # Initialize RAG application
    rag_app = DifyRAGApplication(api_key=api_key)
    
    try:
        # Create knowledge base
        print("\n🏗️ Creating knowledge base...")
        kb_result = rag_app.create_knowledge_base(
            name="AI & RAG Knowledge Base",
            description="A comprehensive knowledge base about AI, RAG systems, and Dify platform"
        )
        
        if not kb_result:
            print("❌ Failed to create knowledge base")
            return
        
        # Create sample documents
        print("\n📄 Creating sample documents...")
        docs_dir = create_sample_documents()
        
        # Upload documents
        print("\n📤 Uploading documents...")
        document_ids = []
        
        for doc_file in docs_dir.glob("*.txt"):
            upload_result = rag_app.upload_document(str(doc_file))
            if upload_result:
                doc_id = upload_result.get('document', {}).get('id')
                if doc_id:
                    document_ids.append(doc_id)
        
        # Wait for processing
        print(f"\n⏳ Processing {len(document_ids)} documents...")
        processing_success = rag_app.wait_for_processing(document_ids, timeout=300)
        
        if not processing_success:
            print("⚠️ Some documents may not have processed completely")
        
        # Create RAG application
        print("\n🤖 Creating RAG application...")
        app_result = rag_app.create_rag_app(
            app_name="AI Knowledge Assistant",
            system_prompt="""You are an AI Knowledge Assistant with expertise in artificial intelligence, RAG systems, and the Dify platform.

Use the provided context to answer questions accurately and comprehensively. If information is not available in the context, clearly state that you don't have that information.

When answering:
1. Be precise and informative
2. Cite relevant parts of the context when possible
3. Provide practical insights and recommendations
4. If asked about implementation, reference best practices from the knowledge base

Context: {{#context#}}{{content}}{{/context#}}

Question: {{#sys.query#}}

Provide a helpful and detailed answer:"""
        )
        
        if not app_result:
            print("❌ Failed to create RAG application")
            return
        
        app_id = app_result.get('id')
        
        # Test queries
        print("\n🔍 Testing RAG application...")
        test_questions = [
            "What is artificial intelligence and what are its main types?",
            "How does a RAG system work and what are its benefits?",
            "What are the key features of the Dify platform?",
            "What are some best practices for implementing RAG systems?",
            "How do you choose the right embedding model for a RAG application?"
        ]
        
        conversation_id = None
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[{i}/{len(test_questions)}] ❓ Question: {question}")
            
            response = rag_app.ask_question(
                question=question,
                app_id=app_id,
                conversation_id=conversation_id
            )
            
            print(f"🤖 Answer: {response['answer']}")
            
            # Show retrieval info
            retrieval_info = response.get('retrieval_info', [])
            if retrieval_info:
                print(f"📚 Sources: {len(retrieval_info)} documents retrieved")
            
            # Update conversation ID for context
            conversation_id = response.get('conversation_id')
            
            time.sleep(1)  # Small delay between questions
        
        # Show knowledge base statistics
        print("\n📊 Knowledge Base Statistics:")
        kb_info = rag_app.get_knowledge_base_info()
        if kb_info:
            print(f"   Documents: {kb_info.get('document_count', 0)}")
            print(f"   Characters: {kb_info.get('character_count', 0)}")
            print(f"   Indexing: {kb_info.get('indexing_technique', 'Unknown')}")
        
        documents = rag_app.list_documents()
        print(f"   Total segments: {sum(doc.get('segment_count', 0) for doc in documents)}")
        
        print(f"\n✅ RAG application demo completed successfully!")
        print(f"   Knowledge Base ID: {rag_app.dataset_id}")
        print(f"   RAG App ID: {app_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()