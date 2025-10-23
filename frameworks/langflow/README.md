# Langflow - Visual AI Agent Builder Guide

[![GitHub Stars](https://img.shields.io/github/stars/langflow-ai/langflow)](https://github.com/langflow-ai/langflow)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[🔍 Compare with other frameworks →](https://www.agentically.sh/ai-agentic-frameworks/compare/langflow/)

Langflow is a visual, low-code platform for building and deploying AI agents and workflows. With its intuitive drag-and-drop interface and deep Python customization capabilities, Langflow bridges the gap between no-code simplicity and developer flexibility.

## Key Features

- 🎨 **Visual Flow Builder**: Drag-and-drop interface for creating AI workflows
- 🐍 **Python Customization**: Full code access for advanced customization
- 🔌 **Rich Integrations**: Support for all major LLMs and vector databases
- 📡 **API Generation**: Auto-generate APIs from visual flows
- 🔄 **Model Context Protocol**: Native MCP support for tool integration
- 📱 **Multi-Modal Support**: Handle text, images, and documents
- 🚀 **Multiple Deployments**: API, MCP server, or embedded modes

## When to Use Langflow

✅ **Best for:**
- Rapid prototyping of AI applications
- Teams with mixed technical backgrounds
- Visual workflow design and documentation
- RAG applications with complex pipelines
- Quick API generation from AI workflows
- Educational and demo environments
- Multi-agent coordination with visual clarity

❌ **Not ideal for:**
- Pure code-first development workflows
- Highly complex algorithmic logic
- Applications requiring custom UI components
- Real-time performance-critical systems

## Quick Start

### Installation

```bash
# Using pip
pip install langflow

# Using uv (recommended)
uv pip install langflow -U

# Using conda
conda install -c conda-forge langflow
```

### Launch Langflow

```bash
# Start the Langflow server
langflow run

# Custom port and host
langflow run --port 7860 --host 0.0.0.0

# Development mode with auto-reload
langflow run --dev
```

Open your browser to `http://localhost:7860` to access the visual interface.

### Your First Flow

```python
# Create a simple chatbot flow programmatically
from langflow.graph import Graph
from langflow.components import OpenAIModel, ChatInput, ChatOutput

# Initialize graph
graph = Graph()

# Add components
chat_input = ChatInput(message="Hello, how can I help you?")
llm = OpenAIModel(model="gpt-4")
chat_output = ChatOutput()

# Connect components
chat_input.message >> llm.input
llm.output >> chat_output.message

# Save flow
graph.save("simple_chatbot.json")
```

## Examples

- [Basic Chatbot](./examples/basic-chatbot.py) - Simple Q&A interface
- [RAG Pipeline](./examples/rag-pipeline.py) - Document-based Q&A system
- [Multi-Agent Flow](./examples/multi-agent-flow.py) - Coordinated agent workflows
- [Custom Component](./examples/custom-component.py) - Building custom nodes

## Benchmarks

[View detailed benchmarks →](./benchmarks.md)

| Metric | Langflow | Industry Average |
|--------|----------|------------------|
| Setup Time | 5 minutes | 30 minutes |
| Visual Clarity | 9/10 | 6/10 |
| Code Flexibility | 8/10 | 9/10 |
| Learning Curve | 8/10 | 6/10 |
| Team Collaboration | 7/10 | 7/10 |

## Core Concepts

### Visual Flow Design

Langflow uses a node-based visual interface where each component represents a step in your AI workflow:

```python
# Example: Building a RAG flow visually translates to this structure
from langflow.components import (
    FileLoader, TextSplitter, VectorStore, 
    Retriever, PromptTemplate, OpenAIModel
)

# Document processing pipeline
loader = FileLoader(file_path="documents/")
splitter = TextSplitter(chunk_size=1000, overlap=200)
vectorstore = VectorStore(collection_name="knowledge_base")

# Query processing pipeline
retriever = Retriever(vectorstore=vectorstore, top_k=5)
prompt = PromptTemplate(
    template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)
llm = OpenAIModel(model="gpt-4")

# Flow connections (done visually in UI)
loader.output >> splitter.input
splitter.output >> vectorstore.input
retriever.input << vectorstore.output
prompt.context << retriever.output
llm.input << prompt.output
```

### Custom Components

Create reusable components for specific use cases:

```python
from langflow.custom import CustomComponent
from typing import Dict, Any

class WebSearchComponent(CustomComponent):
    display_name = "Web Search"
    description = "Search the web using DuckDuckGo"
    
    def build_config(self):
        return {
            "query": {"display_name": "Search Query"},
            "max_results": {"display_name": "Max Results", "value": 5}
        }
    
    def build(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        # Your search implementation
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": result["title"],
                    "snippet": result["body"],
                    "url": result["href"]
                })
        
        return {"results": results, "query": query}
```

### API Generation

Every flow automatically generates REST APIs:

```python
# Your flow is accessible via API
import requests

# Call your Langflow API
response = requests.post(
    "http://localhost:7860/api/v1/run/flow-id",
    json={
        "input_value": "What is machine learning?",
        "output_type": "chat",
        "input_type": "chat"
    }
)

result = response.json()
print(result["outputs"]["answer"])
```

## Use Cases

### Document Q&A System

```python
# Create a sophisticated RAG system visually
# This represents the visual flow structure

components = {
    "document_loader": {
        "type": "DirectoryLoader",
        "path": "documents/",
        "extensions": [".pdf", ".docx", ".txt"]
    },
    "text_splitter": {
        "type": "RecursiveTextSplitter",
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "embeddings": {
        "type": "OpenAIEmbeddings",
        "model": "text-embedding-ada-002"
    },
    "vector_store": {
        "type": "ChromaDB",
        "collection_name": "company_docs"
    },
    "retriever": {
        "type": "VectorStoreRetriever",
        "search_kwargs": {"k": 5}
    },
    "prompt_template": {
        "type": "PromptTemplate",
        "template": """
        You are a helpful assistant that answers questions based on the provided context.
        
        Context: {context}
        
        Question: {question}
        
        Provide a comprehensive answer based only on the context above.
        If the answer is not in the context, say "I don't have enough information to answer that question."
        
        Answer:
        """
    },
    "llm": {
        "type": "OpenAI",
        "model_name": "gpt-4"
    }
}

# Flow connections (done visually in Langflow UI)
# document_loader → text_splitter → embeddings → vector_store
# question_input → retriever ← vector_store
# retriever → prompt_template ← question_input
# prompt_template → llm → answer_output
```

### Multi-Agent Research System

```python
# Visual flow for coordinated research agents
research_flow = {
    "web_searcher": {
        "type": "CustomWebSearchAgent",
        "description": "Searches web for current information"
    },
    "document_analyzer": {
        "type": "DocumentAnalysisAgent", 
        "description": "Analyzes uploaded documents"
    },
    "synthesizer": {
        "type": "SynthesisAgent",
        "description": "Combines insights from multiple sources"
    },
    "fact_checker": {
        "type": "FactCheckingAgent",
        "description": "Verifies claims and provides citations"
    },
    "coordinator": {
        "type": "CoordinatorAgent",
        "description": "Orchestrates the research process"
    }
}

# Agent coordination logic (visual in Langflow)
# research_query → coordinator
# coordinator → [web_searcher, document_analyzer] (parallel)
# [web_searcher, document_analyzer] → synthesizer
# synthesizer → fact_checker → final_report
```

### Chatbot with Memory

```python
# Persistent conversation flow
chatbot_components = {
    "chat_input": {
        "type": "ChatInput",
        "session_id": "user_session"
    },
    "memory": {
        "type": "ConversationBufferMemory",
        "max_token_limit": 2000,
        "return_messages": True
    },
    "prompt": {
        "type": "ChatPromptTemplate",
        "template": """
        You are a helpful AI assistant. Use the conversation history to provide contextual responses.
        
        History: {history}
        Human: {input}
        Assistant:
        """
    },
    "llm": {
        "type": "ChatOpenAI",
        "model_name": "gpt-4",
        "temperature": 0.7
    },
    "chat_output": {
        "type": "ChatOutput"
    }
}
```

## Advanced Features

### Model Context Protocol (MCP) Integration

```python
# Langflow supports MCP for tool integration
from langflow.mcp import MCPComponent

class MCPToolComponent(MCPComponent):
    def build_config(self):
        return {
            "server_name": {"display_name": "MCP Server"},
            "tool_name": {"display_name": "Tool Name"},
            "parameters": {"display_name": "Tool Parameters"}
        }
    
    async def run_mcp_tool(self, server_name: str, tool_name: str, parameters: dict):
        # Connect to MCP server and execute tool
        result = await self.mcp_client.call_tool(
            server=server_name,
            tool=tool_name,
            arguments=parameters
        )
        return result
```

### Conditional Flows

```python
# Create conditional logic in visual flows
conditional_flow = {
    "input_classifier": {
        "type": "TextClassifier",
        "categories": ["question", "request", "complaint"]
    },
    "question_handler": {
        "type": "QAChain",
        "condition": "category == 'question'"
    },
    "request_handler": {
        "type": "TaskExecutor", 
        "condition": "category == 'request'"
    },
    "complaint_handler": {
        "type": "EscalationAgent",
        "condition": "category == 'complaint'"
    },
    "response_merger": {
        "type": "ResponseMerger"
    }
}
```

### Multi-Modal Workflows

```python
# Handle images, text, and documents
multimodal_components = {
    "file_input": {
        "type": "FileUpload",
        "accepted_types": [".jpg", ".png", ".pdf", ".txt"]
    },
    "content_classifier": {
        "type": "ContentTypeClassifier"
    },
    "image_processor": {
        "type": "VisionLLM",
        "model": "gpt-4-vision-preview",
        "condition": "file_type == 'image'"
    },
    "document_processor": {
        "type": "DocumentLoader",
        "condition": "file_type == 'document'"
    },
    "text_processor": {
        "type": "TextLLM",
        "condition": "file_type == 'text'"
    }
}
```

## Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile for Langflow
FROM python:3.11-slim

WORKDIR /app

# Install Langflow
RUN pip install langflow

# Copy your flows
COPY flows/ /app/flows/

# Set environment variables
ENV LANGFLOW_HOST=0.0.0.0
ENV LANGFLOW_PORT=7860

# Expose port
EXPOSE 7860

# Start Langflow
CMD ["langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  langflow:
    build: .
    ports:
      - "7860:7860"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGFLOW_DATABASE_URL=postgresql://user:pass@db:5432/langflow
    depends_on:
      - db
    volumes:
      - ./flows:/app/flows
      - ./data:/app/data

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=langflow
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# langflow-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langflow
  template:
    metadata:
      labels:
        app: langflow
    spec:
      containers:
      - name: langflow
        image: your-registry/langflow:latest
        ports:
        - containerPort: 7860
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-key
        - name: LANGFLOW_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database
              key: connection-string
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

---
apiVersion: v1
kind: Service
metadata:
  name: langflow-service
spec:
  selector:
    app: langflow
  ports:
  - protocol: TCP
    port: 80
    targetPort: 7860
  type: LoadBalancer
```

### Environment Configuration

```bash
# .env file for production
LANGFLOW_HOST=0.0.0.0
LANGFLOW_PORT=7860
LANGFLOW_WORKERS=4

# Database
LANGFLOW_DATABASE_URL=postgresql://user:pass@localhost:5432/langflow

# Authentication
LANGFLOW_SUPERUSER=admin
LANGFLOW_SUPERUSER_PASSWORD=secure_password

# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Monitoring
LANGFLOW_LOG_LEVEL=INFO
LANGFLOW_MONITORING_ENABLED=true
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from langflow.graph import Graph
import asyncio

app = FastAPI(title="Langflow API Service")

# Load your flow
graph = Graph.from_file("flows/chatbot.json")

@app.post("/chat")
async def chat_endpoint(message: str, session_id: str = "default"):
    try:
        # Run the flow
        result = await graph.arun(
            inputs={"message": message, "session_id": session_id}
        )
        return {"response": result["output"], "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/flows")
async def list_flows():
    # Return available flows
    return {"flows": ["chatbot", "rag_system", "research_agent"]}
```

### Streamlit Integration

```python
import streamlit as st
from langflow.graph import Graph
import asyncio

st.title("Langflow Chat Interface")

# Load flow
@st.cache_resource
def load_flow():
    return Graph.from_file("flows/chatbot.json")

flow = load_flow()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What can I help you with?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = asyncio.run(flow.arun(inputs={"message": prompt}))
            response = result["output"]
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
```

## Performance Optimization

### Caching Strategies

```python
# Implement caching for expensive operations
from langflow.components import CacheComponent

class OptimizedRAGComponent(CacheComponent):
    def build_config(self):
        return {
            "cache_embeddings": {"value": True},
            "cache_retrieval": {"value": True},
            "cache_ttl": {"value": 3600}  # 1 hour
        }
    
    def run_with_cache(self, query: str):
        # Check cache first
        cached_result = self.get_cache(query)
        if cached_result:
            return cached_result
        
        # Expensive operation
        result = self.perform_rag(query)
        
        # Cache result
        self.set_cache(query, result, ttl=self.cache_ttl)
        return result
```

### Async Processing

```python
# Handle multiple requests concurrently
import asyncio
from langflow.graph import Graph

class AsyncFlowRunner:
    def __init__(self, flow_path: str):
        self.graph = Graph.from_file(flow_path)
    
    async def process_batch(self, inputs: list):
        """Process multiple inputs concurrently"""
        tasks = [
            self.graph.arun(inputs=input_data) 
            for input_data in inputs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def stream_process(self, input_stream):
        """Process streaming inputs"""
        async for input_data in input_stream:
            result = await self.graph.arun(inputs=input_data)
            yield result
```

## Monitoring and Observability

### LangSmith Integration

```python
# Track flow performance with LangSmith
from langflow.monitoring import LangSmithMonitor

monitor = LangSmithMonitor(
    api_key="your_langsmith_key",
    project_name="langflow_production"
)

# Instrument your flows
instrumented_flow = monitor.instrument_flow(graph)

# Automatic tracing and metrics
result = await instrumented_flow.arun(inputs={"query": "user question"})
```

### Custom Metrics

```python
# Add custom monitoring
from langflow.monitoring import MetricsCollector
import time

class PerformanceMonitor:
    def __init__(self):
        self.metrics = MetricsCollector()
    
    async def run_with_monitoring(self, flow, inputs):
        start_time = time.time()
        
        try:
            result = await flow.arun(inputs=inputs)
            
            # Record success metrics
            self.metrics.record_latency(time.time() - start_time)
            self.metrics.increment_counter("flow_success")
            
            return result
        except Exception as e:
            # Record error metrics
            self.metrics.increment_counter("flow_error")
            self.metrics.record_error(str(e))
            raise
```

## Migration and Compatibility

### From LangChain

```python
# Migrate LangChain chains to Langflow
def migrate_langchain_chain(chain):
    """Convert LangChain chain to Langflow components"""
    
    components = []
    
    # Map LangChain components to Langflow
    mapping = {
        "LLMChain": "LLMComponent",
        "RetrievalQA": "RAGComponent", 
        "ConversationChain": "ChatComponent"
    }
    
    for step in chain.steps:
        component_type = mapping.get(step.__class__.__name__)
        if component_type:
            components.append({
                "type": component_type,
                "config": step.get_config()
            })
    
    return components
```

### Version Control

```bash
# Version control your flows
git init
git add flows/
git commit -m "Add production flows"

# Flow versioning strategy
flows/
├── v1/
│   ├── chatbot.json
│   └── rag_system.json
├── v2/
│   ├── chatbot.json
│   └── rag_system.json
└── current/ -> v2/
```

## Community & Support

- [GitHub Repository](https://github.com/langflow-ai/langflow) - 44.1k+ stars
- [Documentation](https://docs.langflow.org/) - Comprehensive guides
- [Discord Community](https://discord.gg/EqksyE2EX9) - Active support
- [Component Store](https://langflow.org/components) - Community components

## Enterprise Features

### Team Collaboration

```python
# Team flow sharing and collaboration
from langflow.collaboration import TeamWorkspace

workspace = TeamWorkspace(
    organization="your_org",
    members=["dev@company.com", "data@company.com"]
)

# Share flows with team
workspace.share_flow("rag_system.json", permissions=["read", "execute"])

# Version control integration
workspace.enable_git_sync(repo_url="git@github.com:company/flows.git")
```

### Access Control

```python
# Role-based access control
from langflow.auth import RoleManager

roles = RoleManager()

# Define roles
roles.create_role("flow_developer", permissions=[
    "create_flow", "edit_flow", "delete_flow"
])

roles.create_role("flow_executor", permissions=[
    "run_flow", "view_results"
])

# Assign users to roles
roles.assign_user("developer@company.com", "flow_developer")
roles.assign_user("analyst@company.com", "flow_executor")
```

## Best Practices

### 1. Flow Organization

```python
# Good: Modular, reusable components
components = {
    "data_preprocessing": {
        "loader": "DocumentLoader",
        "splitter": "TextSplitter", 
        "embedder": "OpenAIEmbeddings"
    },
    "retrieval": {
        "vectorstore": "ChromaDB",
        "retriever": "VectorRetriever"
    },
    "generation": {
        "prompt": "PromptTemplate",
        "llm": "OpenAI",
        "parser": "OutputParser"
    }
}

# Avoid: Monolithic, single-purpose flows
```

### 2. Error Handling

```python
# Implement robust error handling
from langflow.components import ErrorHandler

class SafeRAGComponent(ErrorHandler):
    def handle_error(self, error: Exception, context: dict):
        if isinstance(error, OpenAIError):
            return {"error": "LLM service unavailable", "fallback": True}
        elif isinstance(error, VectorStoreError):
            return {"error": "Search unavailable", "fallback": True}
        else:
            return {"error": "Unknown error", "fallback": False}
    
    def build(self, query: str):
        try:
            result = self.perform_rag(query)
            return result
        except Exception as e:
            return self.handle_error(e, {"query": query})
```

### 3. Testing Flows

```python
# Test your flows programmatically
import pytest
from langflow.graph import Graph

class TestChatbotFlow:
    def setup_method(self):
        self.flow = Graph.from_file("flows/chatbot.json")
    
    async def test_basic_response(self):
        result = await self.flow.arun(
            inputs={"message": "Hello"}
        )
        assert "output" in result
        assert len(result["output"]) > 0
    
    async def test_rag_retrieval(self):
        result = await self.flow.arun(
            inputs={"message": "What is machine learning?"}
        )
        assert "machine learning" in result["output"].lower()
```

## Further Reading

- [Langflow vs LangChain Comparison](https://www.agentically.sh/ai-agentic-frameworks/compare/langflow-vs-langchain/)
- [Visual AI Development Best Practices](https://www.agentically.sh/ai-agentic-frameworks/langflow/visual-patterns/)
- [Production Deployment Guide](https://www.agentically.sh/ai-agentic-frameworks/langflow/production/)
- [Custom Component Development](https://www.agentically.sh/ai-agentic-frameworks/langflow/custom-components/)

---

[← Back to Framework Comparison](../../) | [Compare Langflow →](https://www.agentically.sh/ai-agentic-frameworks/compare/langflow/)