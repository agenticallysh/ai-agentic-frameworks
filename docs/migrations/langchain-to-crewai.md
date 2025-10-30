# Migration Guide: LangChain to CrewAI

## Overview

This guide helps you migrate from LangChain to CrewAI, focusing on converting your existing agent workflows to CrewAI's role-based multi-agent system.

## Key Conceptual Differences

| LangChain | CrewAI | Notes |
|-----------|---------|-------|
| **Chains** | **Crews** | Collections of agents working together |
| **Agents** | **Agents with Roles** | More specialized, role-based agents |
| **Tools** | **Tools** | Similar concept, different implementation |
| **Memory** | **Shared Context** | Built-in crew-wide memory |
| **Callbacks** | **Callbacks** | Event-driven execution monitoring |

## Migration Steps

### 1. Convert LangChain Agents to CrewAI Agents

**Before (LangChain):**
```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# LangChain agent setup
llm = ChatOpenAI(model="gpt-4")
tools = [search_tool, calculator_tool]

agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
```

**After (CrewAI):**
```python
from crewai import Agent, Crew, Task
from crewai_tools import SerperDevTool, CalculatorTool

# CrewAI agent setup
research_agent = Agent(
    role="Research Specialist",
    goal="Find accurate information and data",
    backstory="You are an expert researcher with access to web search capabilities.",
    tools=[SerperDevTool(), CalculatorTool()],
    verbose=True,
    allow_delegation=False
)
```

### 2. Convert Chains to Crews and Tasks

**Before (LangChain):**
```python
from langchain.chains import LLMChain, SequentialChain

# LangChain sequential chain
research_chain = LLMChain(
    llm=llm,
    prompt=research_prompt,
    output_key="research_results"
)

analysis_chain = LLMChain(
    llm=llm,
    prompt=analysis_prompt,
    output_key="analysis"
)

overall_chain = SequentialChain(
    chains=[research_chain, analysis_chain],
    input_variables=["topic"],
    output_variables=["analysis"],
    verbose=True
)
```

**After (CrewAI):**
```python
# CrewAI crew and tasks
research_task = Task(
    description="Research the topic: {topic}",
    agent=research_agent,
    expected_output="Comprehensive research findings"
)

analysis_task = Task(
    description="Analyze the research findings and provide insights",
    agent=analysis_agent,
    expected_output="Detailed analysis with recommendations",
    dependencies=[research_task]
)

crew = Crew(
    agents=[research_agent, analysis_agent],
    tasks=[research_task, analysis_task],
    verbose=True
)
```

### 3. Tool Migration

**Before (LangChain):**
```python
from langchain.tools import Tool

def search_function(query: str) -> str:
    # Your search implementation
    return search_results

search_tool = Tool(
    name="web_search",
    description="Search the web for information",
    func=search_function
)
```

**After (CrewAI):**
```python
from crewai_tools import BaseTool

class CustomSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information"
    
    def _run(self, query: str) -> str:
        # Your search implementation
        return search_results

# Or use built-in tools
from crewai_tools import SerperDevTool
search_tool = SerperDevTool()
```

### 4. Memory and Context Migration

**Before (LangChain):**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
```

**After (CrewAI):**
```python
# CrewAI has built-in memory management
crew = Crew(
    agents=[research_agent, analysis_agent],
    tasks=[research_task, analysis_task],
    memory=True,  # Enable crew memory
    verbose=True
)
```

## Complete Migration Example

### LangChain Implementation

```python
# Original LangChain implementation
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

# Setup
llm = ChatOpenAI(model="gpt-4")
memory = ConversationBufferMemory(return_messages=True)

# Tools
def web_search(query: str) -> str:
    return f"Search results for: {query}"

def calculator(expression: str) -> str:
    return str(eval(expression))

tools = [
    Tool(name="search", description="Web search", func=web_search),
    Tool(name="calculator", description="Calculator", func=calculator)
]

# Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Execution
result = agent_executor.invoke({"input": "Research AI trends and calculate market size"})
```

### CrewAI Implementation

```python
# Migrated CrewAI implementation
from crewai import Agent, Crew, Task
from crewai_tools import SerperDevTool, CalculatorTool

# Agents
researcher = Agent(
    role="Research Analyst",
    goal="Conduct thorough research on given topics",
    backstory="You are an expert researcher specializing in technology trends.",
    tools=[SerperDevTool()],
    verbose=True
)

analyst = Agent(
    role="Data Analyst", 
    goal="Analyze data and calculate market metrics",
    backstory="You are a skilled analyst with expertise in market calculations.",
    tools=[CalculatorTool()],
    verbose=True
)

# Tasks
research_task = Task(
    description="Research current AI trends and market data",
    agent=researcher,
    expected_output="Comprehensive research report on AI trends"
)

calculation_task = Task(
    description="Calculate market size based on research findings",
    agent=analyst,
    expected_output="Market size calculations with supporting data",
    dependencies=[research_task]
)

# Crew
crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, calculation_task],
    memory=True,
    verbose=True
)

# Execution
result = crew.kickoff(inputs={"topic": "AI trends and market size"})
```

## Key Benefits of Migration

### 1. Better Role Specialization
```python
# CrewAI agents have clear roles and specializations
marketing_specialist = Agent(
    role="Marketing Specialist",
    goal="Create compelling marketing content",
    backstory="Expert in digital marketing with 10+ years experience"
)

technical_writer = Agent(
    role="Technical Writer", 
    goal="Create clear technical documentation",
    backstory="Skilled technical writer specializing in AI/ML topics"
)
```

### 2. Improved Task Dependencies
```python
# Clear task dependencies and workflow
content_creation = Task(
    description="Create marketing content",
    agent=marketing_specialist,
    expected_output="Marketing copy and headlines"
)

technical_review = Task(
    description="Review content for technical accuracy",
    agent=technical_writer,
    expected_output="Reviewed and approved content",
    dependencies=[content_creation]
)
```

### 3. Built-in Collaboration
```python
# Agents can delegate and collaborate naturally
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.hierarchical,  # Hierarchical workflow
    manager_llm=ChatOpenAI(model="gpt-4")
)
```

## Common Migration Challenges

### 1. Custom Tool Conversion

**Challenge:** Converting complex LangChain tools to CrewAI format.

**Solution:**
```python
# LangChain tool
def complex_search(query: str, filters: dict) -> str:
    # Complex implementation
    pass

langchain_tool = Tool(
    name="complex_search",
    description="Advanced search with filters",
    func=complex_search
)

# CrewAI equivalent
class ComplexSearchTool(BaseTool):
    name: str = "complex_search"
    description: str = "Advanced search with filters"
    
    def _run(self, query: str, filters: str = "{}") -> str:
        import json
        filter_dict = json.loads(filters)
        return complex_search(query, filter_dict)
```

### 2. Memory System Migration

**Challenge:** LangChain's flexible memory systems vs CrewAI's built-in memory.

**Solution:**
```python
# For complex memory needs, use custom memory tools
class CustomMemoryTool(BaseTool):
    name: str = "memory_search"
    description: str = "Search previous conversations and context"
    
    def _run(self, query: str) -> str:
        # Implement your custom memory logic
        return memory_results
```

### 3. Prompt Template Migration

**Challenge:** Converting LangChain prompt templates to CrewAI agent configurations.

**Solution:**
```python
# Instead of complex prompt templates, use agent backstory and goal
agent = Agent(
    role="Customer Service Rep",
    goal="Provide helpful and empathetic customer support",
    backstory="""You are an experienced customer service representative 
    with a track record of resolving complex issues. You always:
    - Listen carefully to customer concerns
    - Provide clear, actionable solutions
    - Follow up to ensure satisfaction""",
    llm=ChatOpenAI(model="gpt-4")
)
```

## Testing Your Migration

### 1. Functional Testing
```python
def test_crew_functionality():
    # Test that crew produces expected outputs
    result = crew.kickoff(inputs={"topic": "test topic"})
    assert result is not None
    assert len(result) > 0

def test_agent_roles():
    # Test that agents fulfill their roles
    for agent in crew.agents:
        assert agent.role is not None
        assert agent.goal is not None
```

### 2. Performance Comparison
```python
import time

# Time LangChain execution
start = time.time()
langchain_result = agent_executor.invoke({"input": "test query"})
langchain_time = time.time() - start

# Time CrewAI execution  
start = time.time()
crewai_result = crew.kickoff(inputs={"query": "test query"})
crewai_time = time.time() - start

print(f"LangChain: {langchain_time}s, CrewAI: {crewai_time}s")
```

## Best Practices for Migration

1. **Start Small**: Migrate one agent/chain at a time
2. **Test Thoroughly**: Compare outputs between old and new implementations
3. **Leverage Roles**: Design agents with clear, specific roles
4. **Use Task Dependencies**: Structure workflows with clear task relationships
5. **Monitor Performance**: Track execution time and quality metrics

## Troubleshooting Common Issues

### Issue: Agent Not Using Tools Properly
```python
# Ensure tools are properly configured
agent = Agent(
    role="Researcher",
    tools=[SerperDevTool()],
    verbose=True,  # Enable to see tool usage
    allow_delegation=False  # Prevent confusion
)
```

### Issue: Tasks Not Executing in Order
```python
# Use explicit dependencies
task2 = Task(
    description="Analyze research results",
    agent=analyst,
    dependencies=[task1]  # Explicit dependency
)
```

### Issue: Memory Not Working
```python
# Enable crew memory
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=True,  # Enable memory
    verbose=True
)
```

---

**Next Steps:**
- Review [CrewAI documentation](../frameworks/crewai/README.md)
- Explore [CrewAI examples](../frameworks/crewai/examples/)
- Test your migrated implementation thoroughly