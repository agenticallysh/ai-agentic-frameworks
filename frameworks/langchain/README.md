# LangChain Framework Guide

[![GitHub Stars](https://img.shields.io/github/stars/langchain-ai/langchain)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[🔍 Compare with other frameworks →](https://www.agentically.sh/ai-agentic-frameworks/compare/langchain/)

LangChain is a comprehensive framework for developing applications powered by large language models. It simplifies every stage of the LLM application lifecycle through development, productionization, and deployment.

## Key Features

- 🔗 **Component Chaining**: Build complex workflows by chaining LLM calls and tools
- 🔌 **700+ Integrations**: Largest ecosystem of LLM providers, vector stores, and tools
- 📚 **RAG Pipelines**: Built-in support for retrieval-augmented generation
- 🧠 **Memory Management**: Conversation memory and context management
- 🛠️ **Custom Tools**: Easy integration of external APIs and services
- 📊 **LangSmith**: Production observability and debugging

## When to Use LangChain

✅ **Best for:**
- RAG applications and document Q&A systems
- Chatbots and conversational AI
- Flexible LLM applications with multiple integrations
- Prototyping and experimentation
- Applications requiring extensive tool integration

❌ **Not ideal for:**
- Simple single-turn LLM calls
- Complex multi-agent workflows (use LangGraph)
- Performance-critical applications requiring low latency
- Applications needing fine-grained control over execution flow

## Quick Start

### Installation

```bash
pip install langchain langchain-openai langchain-community
```

### Basic Example

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains complex topics simply."),
    ("user", "{input}")
])

# Create output parser
output_parser = StrOutputParser()

# Chain components together
chain = prompt | llm | output_parser

# Execute
result = chain.invoke({"input": "Explain quantum computing in simple terms"})
print(result)
```

## Examples

- [Basic Chain](./examples/basic-chain.py) - Simple prompt-LLM-parser chain
- [RAG System](./examples/rag-system.py) - Document retrieval and Q&A
- [Custom Tools](./examples/custom-tools.py) - Integrating external APIs
- [Memory Chat](./examples/memory-chat.py) - Conversational memory
- [Agent with Tools](./examples/agent-tools.py) - LLM agent with tool access

## Benchmarks

[View detailed benchmarks →](./benchmarks.md)

| Metric | LangChain | Industry Average |
|--------|-----------|------------------|
| Integration Count | 700+ | 150 |
| Setup Time | 10 minutes | 25 minutes |
| Documentation Quality | 9/10 | 6/10 |
| Community Support | 9/10 | 6/10 |
| Memory Usage | 420MB | 380MB |

## Migration Guides

- [From OpenAI SDK to LangChain](../../migration-guides/openai-to-langchain.md)
- [From LangChain to CrewAI](../../migration-guides/langchain-to-crewai.md)
- [From LangChain to LangGraph](../../migration-guides/langchain-to-langgraph.md)

## Core Concepts

### Chains
The fundamental building block of LangChain applications:

```python
from langchain_core.runnables import RunnableLambda

# Simple chain
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Agents
LLM-powered decision makers that can use tools:

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize tools
search = DuckDuckGoSearchRun()
tools = [search]

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute
result = agent_executor.invoke({"input": "What's the weather like in Paris today?"})
```

### Memory
Maintain conversation context:

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi, I'm Alice")
conversation.predict(input="What's my name?")  # Will remember "Alice"
```

## Production Considerations

### Observability with LangSmith
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "your-project-name"

# Your LangChain code will now be traced automatically
```

### Error Handling
```python
from langchain_core.runnables import RunnableConfig

try:
    result = chain.invoke(
        {"input": "your input"},
        config=RunnableConfig(
            max_concurrency=10,
            recursion_limit=5
        )
    )
except Exception as e:
    logger.error(f"Chain execution failed: {e}")
    # Implement fallback logic
```

### Caching
```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Enable caching to reduce API calls
set_llm_cache(InMemoryCache())
```

## Advanced Features

### Custom Retriever
```python
from langchain_core.retrievers import BaseRetriever

class CustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Your custom retrieval logic
        return documents
```

### Custom Output Parser
```python
from langchain_core.output_parsers import BaseOutputParser

class CustomParser(BaseOutputParser):
    def parse(self, text: str):
        # Your custom parsing logic
        return parsed_output
```

### Streaming
```python
# Stream responses for better UX
for chunk in chain.stream({"input": "Tell me a story"}):
    print(chunk, end="", flush=True)
```

## Integration Examples

### Vector Store Integration
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings()
)

# Use as retriever
retriever = vectorstore.as_retriever()
```

### Database Integration
```python
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

db = SQLDatabase.from_uri("sqlite:///example.db")
chain = create_sql_query_chain(llm, db)
```

## Use Cases

### Document Q&A System
Perfect for building systems that answer questions about your documents:
- Knowledge bases
- Customer support bots
- Research assistants
- Legal document analysis

### Chatbots
Build sophisticated conversational interfaces:
- Customer service automation
- Personal assistants
- Educational tutors
- Interactive documentation

### Content Generation
Automated content creation workflows:
- Blog post generation
- Email automation
- Social media content
- Product descriptions

## Performance Tips

### 1. Choose the Right Model
```python
# Use cheaper models for simple tasks
simple_llm = ChatOpenAI(model="gpt-3.5-turbo")
complex_llm = ChatOpenAI(model="gpt-4")

# Route based on complexity
def get_llm(task_complexity):
    return complex_llm if task_complexity > 0.7 else simple_llm
```

### 2. Optimize Prompts
```python
# Use few-shot examples for better performance
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer."),
    ("user", "Document: {example_doc}\nSummary: {example_summary}"),
    ("user", "Document: {document}\nSummary:")
])
```

### 3. Batch Processing
```python
# Process multiple inputs at once
results = chain.batch([
    {"input": "question 1"},
    {"input": "question 2"},
    {"input": "question 3"}
])
```

## Community & Support

- [GitHub Repository](https://github.com/langchain-ai/langchain) - 117k+ stars
- [Documentation](https://python.langchain.com/docs/introduction/) - Comprehensive guides
- [Discord Community](https://discord.gg/langchain) - Active community support
- [LangSmith](https://smith.langchain.com/) - Production monitoring
- [LangChain Hub](https://smith.langchain.com/hub) - Shared prompts and chains

## Enterprise Features

### LangSmith Observability
- Request tracing and debugging
- Performance monitoring
- Cost tracking
- A/B testing support

### LangGraph Integration
- Complex workflow orchestration
- Multi-agent systems
- Human-in-the-loop workflows
- State management

### Security & Compliance
- Data privacy controls
- Audit logging
- Role-based access control
- SOC 2 compliance

## Further Reading

- [LangChain vs LangGraph Comparison](https://www.agentically.sh/ai-agentic-frameworks/compare/langchain-vs-langgraph/)
- [Production Deployment Guide](https://www.agentically.sh/ai-agentic-frameworks/langchain/production/)
- [Cost Optimization Strategies](https://www.agentically.sh/ai-agentic-frameworks/cost-calculator/?framework=langchain)
- [Migration from LangChain](https://www.agentically.sh/ai-agentic-frameworks/migrate/langchain/)

---

[← Back to Framework Comparison](../../) | [Compare LangChain →](https://www.agentically.sh/ai-agentic-frameworks/compare/langchain/)