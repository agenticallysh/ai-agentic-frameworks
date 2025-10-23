# LlamaIndex Agents Framework Guide

[![GitHub Stars](https://img.shields.io/github/stars/run-llama/llama_index)](https://github.com/run-llama/llama_index)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[🔍 Compare with other frameworks →](https://www.agentically.sh/ai-agentic-frameworks/compare/llamaindex/)

LlamaIndex is a data framework for connecting LLMs with external data, featuring powerful agent capabilities for tool calling and workflow orchestration. It's the go-to choice for building sophisticated RAG applications with agent functionality.

## Key Features

- 📚 **Best-in-Class RAG**: Industry-leading retrieval-augmented generation
- 🛠️ **Agent Capabilities**: Tool-calling agents with workflow orchestration
- 🔌 **200+ Data Connectors**: Connect to any data source via LlamaHub
- 🧠 **Advanced Indexing**: Multiple index types for optimal retrieval
- 🎯 **Query Engines**: Sophisticated query processing and routing
- ☁️ **LlamaCloud**: Managed parsing and ingestion service

## When to Use LlamaIndex Agents

✅ **Best for:**
- RAG applications requiring agent functionality
- Document Q&A systems with tool integration
- Knowledge base applications with complex workflows
- Research assistants with data retrieval capabilities
- Applications needing sophisticated data ingestion
- Enterprise knowledge management systems

❌ **Not ideal for:**
- Simple conversational AI without data retrieval
- Pure multi-agent systems (use CrewAI/AutoGen)
- Applications not requiring external data
- Real-time chat applications without knowledge base

## Quick Start

### Installation

```bash
pip install llama-index
```

### Basic Agent Example

```python
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine tool
query_engine = index.as_query_engine()
query_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="company_docs",
        description="Search company documentation and policies"
    )
)

# Create agent
llm = OpenAI(model="gpt-4")
agent = OpenAIAgent.from_tools(
    [query_tool],
    llm=llm,
    verbose=True
)

# Use agent
response = agent.chat("What is our company's vacation policy?")
print(response)
```

## Examples

- [RAG Agent](./examples/rag-agent.py) - Document Q&A with agent capabilities
- [Multi-Tool Agent](./examples/multi-tool-agent.py) - Agent with multiple tools
- [Research Assistant](./examples/research-assistant.py) - Advanced research workflows
- [Data Analysis Agent](./examples/data-analysis-agent.py) - Analytics with natural language

## Benchmarks

[View detailed benchmarks →](./benchmarks.md)

| Metric | LlamaIndex | Industry Average |
|--------|-------------|------------------|
| RAG Quality | 9/10 | 7/10 |
| Data Connectors | 200+ | 50 |
| Agent Maturity | 7/10 | 8/10 |
| Documentation | 9/10 | 7/10 |
| Setup Time | 15 minutes | 25 minutes |

## Core Concepts

### Agents with Tools
```python
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools import FunctionTool

# Define custom tools
def weather_tool(location: str) -> str:
    """Get current weather for a location."""
    # Integration with weather API
    return f"Weather in {location}: 72°F, sunny"

def calculator_tool(expression: str) -> str:
    """Calculate mathematical expressions."""
    return str(eval(expression))

# Create tools
weather_fn = FunctionTool.from_defaults(fn=weather_tool)
calc_fn = FunctionTool.from_defaults(fn=calculator_tool)

# Create agent with multiple tools
agent = OpenAIAgent.from_tools(
    [query_tool, weather_fn, calc_fn],
    verbose=True
)

# Agent can now use any tool as needed
response = agent.chat("What's the weather in NYC and what's 25 * 4?")
```

### Advanced RAG with Agents
```python
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Create sophisticated RAG system
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10
)

# Add post-processing
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[postprocessor]
)

# Wrap in tool for agent
rag_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="knowledge_base",
        description="Search internal knowledge base"
    )
)
```

## Use Cases

### Document Q&A Agent
```python
# Multi-document agent
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.tools import QueryEngineTool

# Load different document sets
hr_docs = SimpleDirectoryReader("hr_policies").load_data()
tech_docs = SimpleDirectoryReader("technical_docs").load_data()

# Create separate indexes
hr_index = VectorStoreIndex.from_documents(hr_docs)
tech_index = VectorStoreIndex.from_documents(tech_docs)

# Create specialized tools
hr_tool = QueryEngineTool(
    query_engine=hr_index.as_query_engine(),
    metadata=ToolMetadata(
        name="hr_policies",
        description="Search HR policies and procedures"
    )
)

tech_tool = QueryEngineTool(
    query_engine=tech_index.as_query_engine(),
    metadata=ToolMetadata(
        name="technical_docs",
        description="Search technical documentation"
    )
)

# Agent can choose appropriate knowledge base
agent = OpenAIAgent.from_tools([hr_tool, tech_tool])
```

### Research Assistant
```python
from llama_index.tools import DuckDuckGoSearchToolSpec

# Web search integration
search_tool = DuckDuckGoSearchToolSpec().to_tool_list()[0]

# Combine web search with knowledge base
research_agent = OpenAIAgent.from_tools([
    query_tool,  # Internal knowledge
    search_tool  # Web search
])

# Agent can research using both sources
response = research_agent.chat(
    "Compare our product with competitors and provide market analysis"
)
```

## Advanced Features

### Multi-Modal Agents
```python
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal import MultiModalVectorStoreIndex

# Multi-modal agent for images and text
mm_llm = OpenAIMultiModal(model="gpt-4-vision-preview")

# Index with images and text
mm_index = MultiModalVectorStoreIndex.from_documents(
    documents,  # Can include images
    llm=mm_llm
)

mm_agent = OpenAIAgent.from_tools([
    QueryEngineTool(
        query_engine=mm_index.as_query_engine(),
        metadata=ToolMetadata(
            name="multimodal_search",
            description="Search text and images"
        )
    )
])
```

### Custom Query Engines
```python
from llama_index.core.query_engine import CustomQueryEngine

class RAGStringQueryEngine(CustomQueryEngine):
    """Custom RAG query engine."""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def custom_query(self, query_str: str):
        # Custom retrieval logic
        nodes = self.retriever.retrieve(query_str)
        
        # Custom synthesis
        context = "\n".join([node.text for node in nodes])
        
        prompt = f"""
        Context: {context}
        
        Question: {query_str}
        
        Answer based on the context:
        """
        
        response = self.llm.complete(prompt)
        return response

# Use custom query engine
custom_engine = RAGStringQueryEngine(retriever, llm)
```

## Production Deployment

### LlamaCloud Integration
```python
import os
from llama_index.core import VectorStoreIndex
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

# Use managed LlamaCloud index
os.environ["LLAMA_CLOUD_API_KEY"] = "your-api-key"

# Create cloud index
cloud_index = LlamaCloudIndex(
    name="company-docs",
    project_name="my-project"
)

# Upload documents
cloud_index.insert_documents(documents)

# Use in agent
cloud_tool = QueryEngineTool(
    query_engine=cloud_index.as_query_engine(),
    metadata=ToolMetadata(
        name="cloud_docs",
        description="Search cloud-managed documents"
    )
)
```

### Async Agents
```python
import asyncio
from llama_index.agent.openai import OpenAIAgent

async def async_agent_example():
    """Async agent for better performance."""
    
    agent = OpenAIAgent.from_tools([query_tool])
    
    # Process multiple queries concurrently
    queries = [
        "What is our return policy?",
        "How do I reset my password?",
        "What are the shipping options?"
    ]
    
    tasks = [agent.achat(query) for query in queries]
    responses = await asyncio.gather(*tasks)
    
    return responses

# Run async
responses = asyncio.run(async_agent_example())
```

## Integration Examples

### Streamlit App
```python
import streamlit as st
from llama_index.agent.openai import OpenAIAgent

@st.cache_resource
def load_agent():
    return OpenAIAgent.from_tools([query_tool])

st.title("LlamaIndex Agent Demo")

if "agent" not in st.session_state:
    st.session_state.agent = load_agent()

user_input = st.text_input("Ask a question:")

if user_input:
    with st.spinner("Thinking..."):
        response = st.session_state.agent.chat(user_input)
        st.write(response.response)
```

### FastAPI Service
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
agent = OpenAIAgent.from_tools([query_tool])

class QueryRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    response = agent.chat(request.message)
    return {"response": response.response}

# Run with: uvicorn app:app --reload
```

## Performance Optimization

### Caching
```python
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer

# Add memory to agent
chat_store = SimpleChatStore()
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user_1"
)

agent = OpenAIAgent.from_tools(
    [query_tool],
    memory=memory,
    verbose=True
)
```

### Efficient Indexing
```python
from llama_index.core import StorageContext, load_index_from_storage

# Persist index to disk
index.storage_context.persist(persist_dir="./storage")

# Load from storage (much faster)
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

## Community & Support

- [GitHub Repository](https://github.com/run-llama/llama_index) - 44.6k+ stars
- [Documentation](https://docs.llamaindex.ai/) - Comprehensive guides
- [Discord Community](https://discord.gg/dGcwcsnxhU) - Active support
- [LlamaHub](https://llamahub.ai/) - Data connectors and tools

## Enterprise Features

### LlamaCloud Platform
- Managed document parsing and ingestion
- Enterprise-grade security and compliance
- Scalable vector storage and retrieval
- Analytics and monitoring dashboards

### LlamaParse
- Advanced document parsing (PDFs, tables, forms)
- Multi-modal parsing for images and text
- API-based parsing service
- Custom parsing configurations

## Best Practices

### 1. Agent Design
```python
# Good: Specific, focused tools
tools = [
    QueryEngineTool(
        query_engine=finance_index.as_query_engine(),
        metadata=ToolMetadata(
            name="financial_data",
            description="Search financial reports and data"
        )
    ),
    QueryEngineTool(
        query_engine=legal_index.as_query_engine(),
        metadata=ToolMetadata(
            name="legal_docs",
            description="Search legal documents and contracts"
        )
    )
]

# Avoid: Generic, overlapping tools
```

### 2. Error Handling
```python
try:
    response = agent.chat(user_query)
    return response.response
except Exception as e:
    logger.error(f"Agent error: {e}")
    return "I'm sorry, I encountered an error processing your request."
```

### 3. Cost Optimization
```python
# Use smaller models for simple queries
from llama_index.llms.openai import OpenAI

cheap_llm = OpenAI(model="gpt-3.5-turbo")
expensive_llm = OpenAI(model="gpt-4")

# Route based on complexity
def get_llm(query_complexity):
    return expensive_llm if query_complexity > 0.7 else cheap_llm
```

## Further Reading

- [LlamaIndex vs LangChain Comparison](https://www.agentically.sh/ai-agentic-frameworks/compare/llamaindex-vs-langchain/)
- [RAG Best Practices](https://www.agentically.sh/ai-agentic-frameworks/llamaindex/rag-patterns/)
- [Agent Development Guide](https://www.agentically.sh/ai-agentic-frameworks/llamaindex/agents/)
- [Production Deployment](https://www.agentically.sh/ai-agentic-frameworks/llamaindex/production/)

---

[← Back to Framework Comparison](../../) | [Compare LlamaIndex →](https://www.agentically.sh/ai-agentic-frameworks/compare/llamaindex/)