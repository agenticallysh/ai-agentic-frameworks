# LangGraph Framework Guide

[![GitHub Stars](https://img.shields.io/github/stars/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[🔍 Compare with other frameworks →](https://www.agentically.sh/ai-agentic-frameworks/compare/langgraph/)

LangGraph is a framework for building controllable, production-grade AI agents with graph-based workflows. It extends LangChain by providing primitives for orchestrating multi-step, non-linear processes with full control over agent behavior.

## Key Features

- 🔄 **Graph-Based Workflows**: Define complex, non-linear agent workflows as graphs
- 🧠 **State Management**: Built-in state persistence and memory across interactions
- 🎯 **Multi-Agent Orchestration**: Coordinate multiple agents with different roles
- 👥 **Human-in-the-Loop**: First-class support for human intervention and approval
- 🔍 **Observability**: Streaming execution with full visibility into agent reasoning
- ⚡ **Production Ready**: Used by companies like Klarna, Uber, and LinkedIn

## When to Use LangGraph

✅ **Best for:**
- Complex multi-agent systems requiring coordination
- Non-linear workflows with conditional logic
- Applications requiring human oversight and intervention
- Long-running processes with state persistence
- Production systems needing full control over agent behavior

❌ **Not ideal for:**
- Simple linear workflows (use LangChain)
- Quick prototyping and experimentation
- Applications with minimal state requirements
- Single-agent use cases without complex logic

## Quick Start

### Installation

```bash
pip install langgraph langchain langchain-openai
```

### Basic Example

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, List

# Define state
class AgentState(TypedDict):
    messages: List[HumanMessage]
    next_action: str

# Create LLM
llm = ChatOpenAI(model="gpt-4")

# Define nodes
def researcher_node(state: AgentState):
    # Research logic here
    return {"messages": state["messages"] + [HumanMessage(content="Research completed")]}

def writer_node(state: AgentState):
    # Writing logic here
    return {"messages": state["messages"] + [HumanMessage(content="Article written")]}

# Create graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

# Add edges
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

# Set entry point
workflow.set_entry_point("researcher")

# Compile and run
app = workflow.compile()
result = app.invoke({"messages": [HumanMessage(content="Write about AI")]})
```

## Examples

- [Basic Graph](./examples/basic-graph.py) - Simple node and edge workflow
- [Multi-Agent System](./examples/multi-agent-system.py) - Coordinated agent collaboration
- [Human-in-Loop](./examples/human-in-loop.py) - Interactive workflows with human input
- [Conditional Routing](./examples/conditional-routing.py) - Dynamic workflow routing
- [State Management](./examples/state-management.py) - Complex state handling

## Benchmarks

[View detailed benchmarks →](./benchmarks.md)

| Metric | LangGraph | Industry Average |
|--------|-----------|------------------|
| State Management | 9/10 | 6/10 |
| Multi-Agent Coordination | 9/10 | 7/10 |
| Production Readiness | 9/10 | 6/10 |
| Human-in-Loop Support | 9/10 | 4/10 |
| Learning Curve | 6/10 | 8/10 |

## Migration Guides

- [From LangChain to LangGraph](../../migration-guides/langchain-to-langgraph.md)
- [From AutoGen to LangGraph](../../migration-guides/autogen-to-langgraph.md)
- [From CrewAI to LangGraph](../../migration-guides/crewai-to-langgraph.md)

## Core Concepts

### Graph Structure
LangGraph workflows are defined as directed graphs with nodes and edges:

```python
from langgraph.graph import StateGraph, END

# Define workflow graph
workflow = StateGraph(StateType)

# Add nodes (functions that process state)
workflow.add_node("analyze", analyze_function)
workflow.add_node("decide", decision_function)
workflow.add_node("execute", execution_function)

# Add edges (define flow between nodes)
workflow.add_edge("analyze", "decide")
workflow.add_conditional_edges(
    "decide",
    routing_function,  # Function that determines next node
    {
        "continue": "execute",
        "end": END
    }
)
```

### State Management
All data flows through a shared state object:

```python
from typing import TypedDict, List, Optional

class WorkflowState(TypedDict):
    user_input: str
    analysis_results: Optional[dict]
    decisions: List[str]
    final_output: Optional[str]
    iteration_count: int
```

### Conditional Routing
Dynamic workflow paths based on state:

```python
def route_based_on_confidence(state: WorkflowState) -> str:
    confidence = state.get("confidence_score", 0)
    
    if confidence > 0.8:
        return "high_confidence_path"
    elif confidence > 0.5:
        return "medium_confidence_path"
    else:
        return "low_confidence_path"

workflow.add_conditional_edges(
    "analysis",
    route_based_on_confidence,
    {
        "high_confidence_path": "quick_response",
        "medium_confidence_path": "detailed_analysis",
        "low_confidence_path": "human_review"
    }
)
```

## Advanced Features

### Human-in-the-Loop
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Add checkpointing for human intervention
memory = SqliteSaver.from_conn_string(":memory:")

# Compile with checkpointing
app = workflow.compile(checkpointer=memory)

# Add interrupt before human review
workflow.add_node("human_review", human_review_node)
app = workflow.compile(checkpointer=memory, interrupt_before=["human_review"])

# Execute with thread for resumability
config = {"configurable": {"thread_id": "conversation-1"}}
result = app.invoke(initial_state, config)

# Resume after human input
updated_state = {"user_feedback": "Approved"}
final_result = app.invoke(updated_state, config)
```

### Streaming Execution
```python
# Stream workflow execution
for event in app.stream(initial_state):
    print(f"Node: {event.get('node')}")
    print(f"Output: {event.get('output')}")
```

### Error Handling and Retries
```python
def resilient_node(state: WorkflowState):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = api_call(state["input"])
            return {"result": result, "status": "success"}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e), "status": "failed"}
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Production Patterns

### Multi-Agent Coordination
```python
class MultiAgentState(TypedDict):
    task: str
    research_results: Optional[str]
    analysis_results: Optional[str]
    final_report: Optional[str]
    current_agent: str

def create_multi_agent_workflow():
    workflow = StateGraph(MultiAgentState)
    
    # Add agent nodes
    workflow.add_node("researcher", research_agent)
    workflow.add_node("analyst", analysis_agent)
    workflow.add_node("writer", writing_agent)
    workflow.add_node("reviewer", review_agent)
    
    # Define coordination logic
    workflow.add_edge("researcher", "analyst")
    workflow.add_conditional_edges(
        "analyst",
        lambda state: "writer" if state["analysis_results"] else "researcher",
        {"writer": "writer", "researcher": "researcher"}
    )
    workflow.add_edge("writer", "reviewer")
    
    return workflow.compile()
```

### Persistent Workflows
```python
from langgraph.checkpoint.postgres import PostgresSaver

# Production checkpointing with PostgreSQL
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

app = workflow.compile(checkpointer=checkpointer)

# Long-running workflow with persistence
config = {"configurable": {"thread_id": f"workflow-{uuid.uuid4()}"}}
for step in app.stream(initial_state, config):
    # Workflow state is automatically persisted
    process_step(step)
```

### Monitoring and Observability
```python
import logging
from langgraph.graph import StateGraph

# Add logging to nodes
def logged_node(func):
    def wrapper(state):
        logging.info(f"Executing {func.__name__} with state: {state}")
        result = func(state)
        logging.info(f"{func.__name__} completed with result: {result}")
        return result
    return wrapper

@logged_node
def analysis_node(state):
    # Node logic here
    return updated_state
```

## Integration with LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-workflows"

# Your LangGraph workflows will be automatically traced
app = workflow.compile()
result = app.invoke(initial_state)  # Automatically traced
```

## Use Cases

### Customer Support Escalation
Build intelligent support systems that can escalate to humans when needed:
- Automated issue classification
- Conditional routing based on complexity
- Human handoff with full context
- Post-resolution analysis

### Research and Analysis Pipelines
Create sophisticated research workflows:
- Multi-source information gathering
- Cross-validation of findings
- Collaborative analysis by multiple agents
- Human expert review integration

### Content Creation Workflows
Orchestrate complex content creation:
- Research → Outline → Writing → Review → Editing
- Quality control checkpoints
- Stakeholder approval processes
- Version control and iteration

## Performance Optimization

### Efficient State Design
```python
# Good: Minimal state with references
class EfficientState(TypedDict):
    task_id: str
    current_step: str
    results_cache_key: str

# Avoid: Large objects in state
class InEfficientState(TypedDict):
    task_id: str
    large_dataset: List[dict]  # This will slow down state serialization
```

### Parallel Execution
```python
from langgraph.prebuilt import ToolExecutor

# Execute multiple agents in parallel
def parallel_analysis(state):
    # Run multiple analyses concurrently
    executor = ToolExecutor([agent1, agent2, agent3])
    results = executor.batch([
        {"input": state["data"], "agent": "agent1"},
        {"input": state["data"], "agent": "agent2"},
        {"input": state["data"], "agent": "agent3"}
    ])
    return {"parallel_results": results}
```

## Community & Support

- [GitHub Repository](https://github.com/langchain-ai/langgraph) - 19.9k+ stars
- [Documentation](https://langchain-ai.github.io/langgraph/) - Comprehensive guides
- [LangGraph Studio](https://smith.langchain.com/) - Visual workflow designer
- [Discord Community](https://discord.gg/langchain) - Active support community

## Enterprise Features

### LangGraph Platform
- Cloud deployment and scaling
- Visual workflow design interface
- Enterprise security and compliance
- Team collaboration features

### Production Monitoring
- Workflow execution tracking
- Performance analytics
- Error monitoring and alerting
- Cost optimization insights

## Best Practices

### 1. Design for Observability
Always add logging and monitoring to your workflows:
```python
def observable_node(state):
    start_time = time.time()
    try:
        result = process_logic(state)
        duration = time.time() - start_time
        log_success(duration, result)
        return result
    except Exception as e:
        log_error(e, state)
        raise
```

### 2. Handle Edge Cases
Design workflows to gracefully handle failures:
```python
def robust_workflow(state):
    if not validate_input(state):
        return {"error": "Invalid input", "status": "failed"}
    
    try:
        return process_successfully(state)
    except APIError as e:
        return {"error": str(e), "retry": True}
    except Exception as e:
        return {"error": "Unexpected error", "status": "failed"}
```

### 3. Optimize State Size
Keep state minimal and use external storage for large data:
```python
def efficient_state_management(state):
    # Store large results externally
    large_result = expensive_computation()
    result_id = store_in_cache(large_result)
    
    return {"result_id": result_id, "status": "completed"}
```

## Further Reading

- [LangGraph vs CrewAI Comparison](https://www.agentically.sh/ai-agentic-frameworks/compare/langgraph-vs-crewai/)
- [Production Deployment Guide](https://www.agentically.sh/ai-agentic-frameworks/langgraph/production/)
- [Multi-Agent Design Patterns](https://www.agentically.sh/ai-agentic-frameworks/langgraph/patterns/)
- [Cost Optimization Strategies](https://www.agentically.sh/ai-agentic-frameworks/cost-calculator/?framework=langgraph)

---

[← Back to Framework Comparison](../../) | [Compare LangGraph →](https://www.agentically.sh/ai-agentic-frameworks/compare/langgraph/)