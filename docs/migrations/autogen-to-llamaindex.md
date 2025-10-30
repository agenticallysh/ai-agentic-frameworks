# Migration Guide: AutoGen to LlamaIndex

## Overview

This guide helps you migrate from AutoGen's conversational multi-agent system to LlamaIndex's data-centric agent framework, focusing on knowledge retrieval and document-based interactions.

## Key Conceptual Differences

| AutoGen | LlamaIndex | Notes |
|---------|-------------|-------|
| **GroupChat** | **Multi-Agent System** | Query orchestration vs conversation |
| **AssistantAgent** | **QueryAgent** | Document-focused vs general purpose |
| **Conversation Flow** | **Query Pipeline** | Linear query processing |
| **Function Calling** | **Tools** | Similar but data-centric tools |
| **Message History** | **Context Management** | Document context vs chat history |

## Migration Steps

### 1. Convert AutoGen Agents to LlamaIndex Agents

**Before (AutoGen):**
```python
import autogen
from autogen import AssistantAgent, UserProxyAgent

research_agent = AssistantAgent(
    name="Research_Specialist",
    system_message="You are an expert researcher.",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10
)
```

**After (LlamaIndex):**
```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# Load and index documents
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine tool
query_engine = index.as_query_engine()
query_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="document_search",
        description="Search through research documents for relevant information"
    )
)

# Create LlamaIndex agent
llm = OpenAI(model="gpt-4")
research_agent = ReActAgent.from_tools(
    [query_tool],
    llm=llm,
    verbose=True
)
```

### 2. Convert GroupChat to Multi-Agent Orchestration

**Before (AutoGen):**
```python
from autogen import GroupChat, GroupChatManager

groupchat = GroupChat(
    agents=[user_proxy, research_agent, analyst_agent],
    messages=[],
    max_round=10
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"model": "gpt-4"}
)
```

**After (LlamaIndex):**
```python
from llama_index.core.agent import AgentRunner
from llama_index.core.workflow import Workflow, StartEvent, StopEvent

class ResearchWorkflow(Workflow):
    def __init__(self, research_agent, analysis_agent):
        super().__init__()
        self.research_agent = research_agent
        self.analysis_agent = analysis_agent
    
    @step
    async def research_step(self, ev: StartEvent) -> str:
        # Use research agent to gather information
        research_result = await self.research_agent.achat(ev.query)
        return research_result.response
    
    @step
    async def analysis_step(self, research_result: str) -> StopEvent:
        # Use analysis agent to process research
        analysis_prompt = f"Analyze this research: {research_result}"
        analysis_result = await self.analysis_agent.achat(analysis_prompt)
        return StopEvent(result=analysis_result.response)

# Create workflow
workflow = ResearchWorkflow(research_agent, analysis_agent)
```

### 3. Tool Migration and Data Integration

**Before (AutoGen):**
```python
def web_search(query: str) -> str:
    return f"Search results for: {query}"

research_agent = AssistantAgent(
    name="Researcher",
    llm_config={
        "model": "gpt-4",
        "functions": [
            {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        ]
    }
)
```

**After (LlamaIndex):**
```python
from llama_index.core.tools import FunctionTool

def web_search(query: str) -> str:
    """Search the web for information"""
    return f"Search results for: {query}"

# Convert to LlamaIndex tool
web_search_tool = FunctionTool.from_defaults(fn=web_search)

# Create agent with tools and document access
tools = [query_tool, web_search_tool]
research_agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True
)
```

### 4. Context and Memory Migration

**Before (AutoGen):**
```python
# AutoGen maintains conversation history automatically
conversation = user_proxy.initiate_chat(
    research_agent,
    message="Research AI trends"
)
```

**After (LlamaIndex):**
```python
from llama_index.core.memory import ChatMemoryBuffer

# Create memory buffer for context
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Create agent with memory
research_agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    memory=memory,
    verbose=True
)

# Chat with context retention
response = research_agent.chat("Research AI trends")
follow_up = research_agent.chat("What are the key findings?")
```

## Complete Migration Example

### AutoGen Implementation

```python
# Original AutoGen implementation
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

def search_documents(query: str) -> str:
    return f"Document search results for: {query}"

def analyze_data(data: str) -> str:
    return f"Analysis of: {data}"

# Agents
researcher = AssistantAgent(
    name="Researcher",
    system_message="You search and gather information from documents.",
    llm_config={
        "model": "gpt-4",
        "functions": [
            {
                "name": "search_documents",
                "description": "Search through documents",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]
    }
)

analyst = AssistantAgent(
    name="Analyst",
    system_message="You analyze research data and provide insights.",
    llm_config={
        "model": "gpt-4",
        "functions": [
            {
                "name": "analyze_data",
                "description": "Analyze research data",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "string"}},
                    "required": ["data"]
                }
            }
        ]
    }
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0
)

# Register functions
user_proxy.register_function(
    function_map={
        "search_documents": search_documents,
        "analyze_data": analyze_data
    }
)

# Group chat
groupchat = GroupChat(
    agents=[user_proxy, researcher, analyst],
    messages=[],
    max_round=8
)

manager = GroupChatManager(groupchat=groupchat, llm_config={"model": "gpt-4"})

# Execution
result = user_proxy.initiate_chat(
    manager,
    message="Research customer feedback trends and analyze them"
)
```

### LlamaIndex Implementation

```python
# Migrated LlamaIndex implementation
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer

# Load and index documents
documents = SimpleDirectoryReader("customer_feedback").load_data()
feedback_index = VectorStoreIndex.from_documents(documents)

# Create query engine
feedback_query_engine = feedback_index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize"
)

# Create tools
document_search_tool = QueryEngineTool(
    query_engine=feedback_query_engine,
    metadata=ToolMetadata(
        name="search_feedback",
        description="Search through customer feedback documents"
    )
)

def analyze_trends(data: str) -> str:
    """Analyze data for trends and patterns"""
    # Analysis logic here
    return f"Trend analysis: {data}"

analysis_tool = FunctionTool.from_defaults(fn=analyze_trends)

# Create agents
llm = OpenAI(model="gpt-4")
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Research agent with document access
research_agent = ReActAgent.from_tools(
    [document_search_tool],
    llm=llm,
    memory=memory,
    verbose=True,
    system_prompt="""You are a research specialist. Use the search_feedback tool 
    to find relevant information in customer feedback documents."""
)

# Analysis agent
analysis_agent = ReActAgent.from_tools(
    [analysis_tool],
    llm=llm,
    memory=memory,
    verbose=True,
    system_prompt="""You are a data analyst. Use the analyze_trends tool 
    to identify patterns and insights in research data."""
)

# Workflow orchestration
class CustomerFeedbackWorkflow(Workflow):
    def __init__(self, research_agent, analysis_agent):
        super().__init__()
        self.research_agent = research_agent
        self.analysis_agent = analysis_agent
    
    @step
    async def research_step(self, ev: StartEvent) -> str:
        research_result = await self.research_agent.achat(
            f"Search for information about: {ev.query}"
        )
        return research_result.response
    
    @step  
    async def analysis_step(self, research_data: str) -> StopEvent:
        analysis_result = await self.analysis_agent.achat(
            f"Analyze this research data for trends: {research_data}"
        )
        return StopEvent(result=analysis_result.response)

# Create and run workflow
workflow = CustomerFeedbackWorkflow(research_agent, analysis_agent)
result = await workflow.run(query="customer feedback trends")
```

## Key Benefits of Migration

### 1. Document-Centric Architecture
```python
# LlamaIndex excels at document processing
documents = SimpleDirectoryReader("research_docs").load_data()

# Multiple index types for different use cases
vector_index = VectorStoreIndex.from_documents(documents)
keyword_index = KeywordTableIndex.from_documents(documents)
tree_index = TreeIndex.from_documents(documents)

# Compose indices for comprehensive search
composable_graph = ComposableGraph.from_indices(
    TreeIndex,
    [vector_index, keyword_index],
    index_summaries=["Vector search", "Keyword search"]
)
```

### 2. Advanced Query Processing
```python
# Sophisticated query engines
query_engine = index.as_query_engine(
    response_mode="tree_summarize",
    similarity_top_k=10,
    streaming=True,
    use_async=True
)

# Custom query transformations
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import CitationQueryEngine

retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
citation_engine = CitationQueryEngine.from_args(
    retriever=retriever,
    response_mode="compact"
)
```

### 3. Multi-Modal Capabilities
```python
# Process different document types
from llama_index.readers.file import PDFReader, ImageReader
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# Multi-modal agent
mm_llm = OpenAIMultiModal(model="gpt-4-vision-preview")
image_documents = ImageReader().load_data("images/")

multi_modal_agent = ReActAgent.from_tools(
    tools=[image_query_tool, text_query_tool],
    llm=mm_llm,
    verbose=True
)
```

## Advanced Migration Patterns

### 1. Sub-Question Query Engine Migration

**AutoGen Multi-Step Reasoning:**
```python
# AutoGen handles complex queries through conversation
def complex_research(topic):
    return user_proxy.initiate_chat(
        manager,
        message=f"Break down and research: {topic}"
    )
```

**LlamaIndex Sub-Question Engine:**
```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# Create tools for different document types
finance_tool = QueryEngineTool.from_defaults(
    query_engine=finance_index.as_query_engine(),
    description="Financial data and reports"
)

market_tool = QueryEngineTool.from_defaults(
    query_engine=market_index.as_query_engine(), 
    description="Market research and trends"
)

# Sub-question engine automatically breaks down complex queries
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[finance_tool, market_tool]
)

response = sub_question_engine.query(
    "What are the financial implications of current market trends?"
)
```

### 2. Routing and Selection Migration

**AutoGen Speaker Selection:**
```python
def speaker_selection(last_speaker, groupchat):
    if "data" in groupchat.messages[-1]["content"]:
        return analyst
    else:
        return researcher
```

**LlamaIndex Router Query Engine:**
```python
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine import RouterQueryEngine

# Define different query engines for different topics
list_query_engine = ListIndex.from_documents(documents).as_query_engine()
vector_query_engine = VectorStoreIndex.from_documents(documents).as_query_engine()

# Router automatically selects appropriate engine
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        QueryEngineTool(
            query_engine=list_query_engine,
            description="For simple factual questions"
        ),
        QueryEngineTool(
            query_engine=vector_query_engine,
            description="For complex analytical questions"
        )
    ]
)
```

## Common Migration Challenges

### 1. Conversation State Management

**Challenge:** AutoGen's natural conversation flow vs LlamaIndex's query-response pattern.

**Solution:**
```python
# Use workflow for conversation-like interactions
class ConversationalWorkflow(Workflow):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.conversation_history = []
    
    @step
    async def chat_step(self, ev: StartEvent) -> StopEvent:
        # Add context from conversation history
        context = "\n".join(self.conversation_history[-3:])  # Last 3 exchanges
        enhanced_query = f"Context: {context}\nCurrent query: {ev.query}"
        
        response = await self.agent.achat(enhanced_query)
        self.conversation_history.append(f"Q: {ev.query}")
        self.conversation_history.append(f"A: {response.response}")
        
        return StopEvent(result=response.response)
```

### 2. Function Calling Complexity

**Challenge:** AutoGen's flexible function calling vs LlamaIndex's tool system.

**Solution:**
```python
# Create tool adapter for complex functions
class AutoGenToolAdapter:
    def __init__(self, autogen_functions):
        self.functions = autogen_functions
    
    def to_llamaindex_tools(self):
        tools = []
        for func_def in self.functions:
            # Convert AutoGen function definition to LlamaIndex tool
            def create_tool(func_name, func_impl):
                return FunctionTool.from_defaults(
                    fn=func_impl,
                    name=func_name,
                    description=func_def.get("description", "")
                )
            
            tools.append(create_tool(func_def["name"], self.functions[func_def["name"]]))
        return tools
```

### 3. Multi-Agent Coordination

**Challenge:** AutoGen's group chat coordination vs LlamaIndex's workflow orchestration.

**Solution:**
```python
# Complex multi-agent workflow
class MultiAgentResearchWorkflow(Workflow):
    def __init__(self, researchers, analysts, synthesizer):
        super().__init__()
        self.researchers = researchers
        self.analysts = analysts  
        self.synthesizer = synthesizer
    
    @step
    async def parallel_research(self, ev: StartEvent) -> list:
        # Run multiple researchers in parallel
        research_tasks = []
        for i, researcher in enumerate(self.researchers):
            task_query = f"{ev.query} - aspect {i+1}"
            research_tasks.append(researcher.achat(task_query))
        
        results = await asyncio.gather(*research_tasks)
        return [r.response for r in results]
    
    @step
    async def analyze_results(self, research_results: list) -> list:
        # Analyze each research result
        analysis_tasks = []
        for result, analyst in zip(research_results, self.analysts):
            analysis_tasks.append(analyst.achat(f"Analyze: {result}"))
        
        analyses = await asyncio.gather(*analysis_tasks)
        return [a.response for a in analyses]
    
    @step
    async def synthesize(self, analyses: list) -> StopEvent:
        # Synthesize all analyses
        combined_analysis = "\n\n".join(analyses)
        synthesis = await self.synthesizer.achat(
            f"Synthesize these analyses: {combined_analysis}"
        )
        return StopEvent(result=synthesis.response)
```

## Testing Your Migration

### 1. Query Quality Testing
```python
def test_query_responses():
    test_queries = [
        "What are the main trends in customer feedback?",
        "How do satisfaction scores correlate with product features?",
        "What are customers saying about our new product?"
    ]
    
    for query in test_queries:
        autogen_result = original_autogen_system(query)
        llamaindex_result = migrated_llamaindex_system(query)
        
        # Compare relevance, accuracy, completeness
        comparison = compare_responses(autogen_result, llamaindex_result)
        assert comparison["similarity"] > 0.8

def test_document_retrieval():
    # Test that relevant documents are being retrieved
    response = research_agent.chat("Find information about pricing feedback")
    
    # Check that response includes document citations
    assert "source" in response.response.lower() or "document" in response.response.lower()
```

### 2. Performance Benchmarking
```python
import time

def benchmark_migration():
    queries = ["test query 1", "test query 2", "test query 3"]
    
    # AutoGen timing
    autogen_times = []
    for query in queries:
        start = time.time()
        autogen_result = autogen_system.query(query)
        autogen_times.append(time.time() - start)
    
    # LlamaIndex timing
    llamaindex_times = []
    for query in queries:
        start = time.time()
        llamaindex_result = llamaindex_system.query(query)
        llamaindex_times.append(time.time() - start)
    
    print(f"AutoGen avg: {sum(autogen_times)/len(autogen_times):.2f}s")
    print(f"LlamaIndex avg: {sum(llamaindex_times)/len(llamaindex_times):.2f}s")
```

## Best Practices for Migration

1. **Focus on Data**: Design around your document/data sources
2. **Leverage Indexing**: Use appropriate index types for your use case
3. **Test Retrieval Quality**: Ensure relevant documents are retrieved
4. **Optimize Query Engines**: Choose the right query engine for your needs
5. **Monitor Performance**: Track query latency and accuracy

## Troubleshooting Common Issues

### Issue: Poor Document Retrieval
```python
# Improve retrieval with better indexing
from llama_index.core.node_parser import SentenceSplitter

# Better text splitting
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# Reindex with better parameters
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[splitter],
    show_progress=True
)
```

### Issue: Slow Query Performance
```python
# Optimize query engine
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Reduce from default 10
    response_mode="compact",  # Faster than tree_summarize
    streaming=True  # Better perceived performance
)
```

### Issue: Agent Not Using Tools
```python
# Ensure tools are properly configured
tools = [document_tool, analysis_tool]
agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True,  # See tool usage
    max_iterations=10,  # Allow multiple tool calls
    allow_parallel_tool_calls=True
)
```

---

**Next Steps:**
- Review [LlamaIndex documentation](../frameworks/llamaindex/README.md)  
- Explore [LlamaIndex examples](../frameworks/llamaindex/examples/)
- Optimize your document indexing strategy