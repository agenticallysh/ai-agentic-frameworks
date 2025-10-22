# Migration Guide: LangChain to CrewAI

> **Migration Complexity**: Medium  
> **Estimated Time**: 2-4 hours for typical projects  
> **Compatibility**: Most LangChain agent patterns can be migrated

[🔍 Compare LangChain vs CrewAI →](https://www.agentically.sh/ai-agentic-frameworks/compare/langchain-vs-crewai/)

## Why Migrate to CrewAI?

CrewAI offers several advantages over traditional LangChain agents:

- ✅ **Better Multi-Agent Coordination**: Native role-based collaboration
- ✅ **Simplified Agent Management**: Less boilerplate code
- ✅ **Built-in Memory**: Agents remember context across tasks  
- ✅ **Production Ready**: Better error handling and reliability
- ✅ **Cost Efficiency**: 15-20% reduction in token usage

## Before You Start

### Prerequisites
- Python 3.8+
- Existing LangChain agent project
- OpenAI API key or compatible LLM

### Backup Your Code
```bash
git branch langchain-backup
git checkout -b migrate-to-crewai
```

## Step 1: Installation & Setup

### Remove LangChain Dependencies
```bash
pip uninstall langchain langchain-community langchain-openai
```

### Install CrewAI
```bash
pip install crewai crewai-tools
```

### Update Requirements
```txt
# requirements.txt
crewai>=0.12.0
crewai-tools>=0.2.0
openai>=1.0.0
```

## Step 2: Core Concepts Mapping

| LangChain Concept | CrewAI Equivalent | Notes |
|-------------------|-------------------|-------|
| `Agent` | `Agent` | Similar but more role-focused |
| `AgentExecutor` | `Crew` | Manages multiple agents |
| `Tool` | `Tool` (from crewai-tools) | Compatible interface |
| `Memory` | Built-in memory | No separate implementation needed |
| `Chain` | `Task` sequence | More explicit task definition |

## Step 3: Agent Migration

### LangChain Agent (Before)
```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

# LangChain setup
llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [
    Tool(
        name="web_search",
        description="Search the web for information",
        func=web_search_function
    )
]

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=memory,
    verbose=True
)

result = agent_executor.invoke({"input": "Research AI trends"})
```

### CrewAI Agent (After)
```python
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# CrewAI setup
search_tool = SerperDevTool()

researcher = Agent(
    role='Research Specialist',
    goal='Conduct thorough research on given topics',
    backstory="""You are an experienced researcher with expertise in 
    technology trends and market analysis.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    memory=True  # Built-in memory
)

research_task = Task(
    description='Research current AI trends and provide comprehensive analysis',
    agent=researcher,
    expected_output='Detailed research report with key insights'
)

crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=2
)

result = crew.kickoff()
```

## Step 4: Multi-Agent Migration

### LangChain Multi-Agent (Before)
```python
# Complex setup with multiple agents
from langchain.agents import AgentExecutor
from langchain.memory import ConversationSummaryBufferMemory

# Create separate agents
researcher_agent = create_openai_functions_agent(llm, research_tools, research_prompt)
writer_agent = create_openai_functions_agent(llm, writing_tools, writing_prompt)

# Manual coordination required
research_executor = AgentExecutor(agent=researcher_agent, tools=research_tools)
writing_executor = AgentExecutor(agent=writer_agent, tools=writing_tools)

# Sequential execution with manual state passing
research_result = research_executor.invoke({"input": "Research topic"})
writing_result = writing_executor.invoke({
    "input": f"Write article based on: {research_result['output']}"
})
```

### CrewAI Multi-Agent (After)
```python
from crewai import Agent, Task, Crew, Process

# Define agents with roles
researcher = Agent(
    role='Senior Researcher',
    goal='Gather comprehensive information on assigned topics',
    backstory='You are a senior researcher with 10 years of experience...',
    tools=[search_tool],
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Create engaging content based on research',
    backstory='You are a skilled writer specializing in technology content...',
    verbose=True
)

# Define tasks with dependencies
research_task = Task(
    description='Research the latest AI agent frameworks',
    agent=researcher,
    expected_output='Comprehensive research brief'
)

writing_task = Task(
    description='Write an engaging article based on the research',
    agent=writer,
    expected_output='1500-word article',
    context=[research_task]  # Automatic dependency
)

# Create crew with automatic coordination
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=2
)

result = crew.kickoff()
```

## Step 5: Tool Migration

### LangChain Tools
```python
from langchain.tools import Tool

def custom_search(query: str) -> str:
    # Custom implementation
    return search_results

langchain_tool = Tool(
    name="custom_search",
    description="Search for information",
    func=custom_search
)
```

### CrewAI Tools
```python
from crewai_tools import BaseTool

class CustomSearchTool(BaseTool):
    name: str = "custom_search"
    description: str = "Search for information on any topic"
    
    def _run(self, query: str) -> str:
        # Same custom implementation
        return search_results

# Or use built-in tools
from crewai_tools import SerperDevTool, FileReadTool
search_tool = SerperDevTool()
file_tool = FileReadTool()
```

## Step 6: Memory & Context Migration

### LangChain Memory
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Manual memory management required
```

### CrewAI Memory (Built-in)
```python
# Memory is automatically handled
agent = Agent(
    role='Assistant',
    goal='Help with tasks',
    backstory='...',
    memory=True,  # That's it!
    verbose=True
)
```

## Step 7: Configuration Migration

### Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_key_here
SERPER_API_KEY=your_serper_key  # For web search
```

### LangChain Config
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4"
)
```

### CrewAI Config (Automatic)
```python
# CrewAI automatically uses OPENAI_API_KEY
# No explicit LLM configuration needed for basic use

# For custom configuration
import os
os.environ["OPENAI_MODEL_NAME"] = "gpt-4"
os.environ["OPENAI_TEMPERATURE"] = "0.7"
```

## Step 8: Error Handling Migration

### LangChain Error Handling
```python
try:
    result = agent_executor.invoke({"input": query})
except Exception as e:
    # Manual error handling
    print(f"Error: {e}")
```

### CrewAI Error Handling
```python
from crewai.exceptions import CrewException

try:
    result = crew.kickoff()
except CrewException as e:
    print(f"Crew execution failed: {e}")
    # Built-in error context and recovery suggestions
```

## Step 9: Testing Migration

### Create Test Suite
```python
# test_migration.py
import pytest
from crewai import Agent, Task, Crew

def test_agent_creation():
    agent = Agent(
        role='Test Agent',
        goal='Test basic functionality',
        backstory='Test backstory'
    )
    assert agent.role == 'Test Agent'

def test_task_execution():
    agent = Agent(role='Tester', goal='Test', backstory='Test')
    task = Task(
        description='Simple test task',
        agent=agent,
        expected_output='Test output'
    )
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    assert len(result) > 0

if __name__ == "__main__":
    pytest.main([__file__])
```

### Run Tests
```bash
python test_migration.py
```

## Step 10: Performance Optimization

### LangChain vs CrewAI Performance
```python
# Benchmark your migration
import time

def benchmark_execution(crew, iterations=5):
    times = []
    for _ in range(iterations):
        start = time.time()
        result = crew.kickoff()
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average execution time: {avg_time:.2f}s")
    return avg_time

# Compare with your old LangChain implementation
crewai_time = benchmark_execution(crew)
```

## Common Migration Issues

### Issue 1: Tool Compatibility
**Problem**: LangChain tools don't work directly
**Solution**: Wrap in CrewAI tool format
```python
from crewai_tools import BaseTool

class LangChainToolWrapper(BaseTool):
    name: str = "wrapped_tool"
    description: str = "Wrapped LangChain tool"
    
    def __init__(self, langchain_tool):
        super().__init__()
        self.lc_tool = langchain_tool
    
    def _run(self, *args, **kwargs):
        return self.lc_tool.run(*args, **kwargs)
```

### Issue 2: Prompt Templates
**Problem**: LangChain prompt templates need conversion
**Solution**: Use CrewAI's agent backstory and task descriptions
```python
# Instead of complex prompt templates
agent = Agent(
    role='Data Analyst',
    goal='Analyze data and provide insights',
    backstory="""You are an expert data analyst with 5 years of experience.
    You excel at finding patterns and creating actionable insights.
    Always provide specific recommendations based on data."""
)
```

### Issue 3: Streaming Responses
**Problem**: LangChain streaming not directly available
**Solution**: Use verbose mode for real-time feedback
```python
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=2  # Shows real-time progress
)
```

## Feature Comparison

| Feature | LangChain | CrewAI | Migration Notes |
|---------|-----------|--------|-----------------|
| Single Agent | ✅ | ✅ | Direct migration |
| Multi-Agent | ⚠️ Complex | ✅ Simple | Significant improvement |
| Memory | ⚠️ Manual | ✅ Built-in | Easier implementation |
| Tools | ✅ | ✅ | May need wrapper |
| Streaming | ✅ | ⚠️ Limited | Use verbose mode |
| Custom LLMs | ✅ | ⚠️ Limited | Check compatibility |

## Post-Migration Checklist

- [ ] All agents converted to CrewAI format
- [ ] Tools working correctly
- [ ] Memory functioning as expected
- [ ] Error handling implemented
- [ ] Tests passing
- [ ] Performance benchmarked
- [ ] Documentation updated
- [ ] Team trained on new system

## Getting Help

### Community Resources
- [CrewAI Discord](https://discord.gg/crewai)
- [GitHub Issues](https://github.com/joaomdmoura/crewAI/issues)
- [Migration Support Forum](https://www.agentically.sh/ai-agentic-frameworks/migration-support/)

### Professional Support
- [Migration Consulting](https://www.agentically.sh/ai-agentic-frameworks/migration-consulting/)
- [Code Review Service](https://www.agentically.sh/ai-agentic-frameworks/code-review/)

---

**Next Steps**: 
- [Optimize your CrewAI setup](https://www.agentically.sh/ai-agentic-frameworks/crewai/optimization/)
- [Explore advanced CrewAI features](https://www.agentically.sh/ai-agentic-frameworks/crewai/advanced/)
- [Join the CrewAI community](https://discord.gg/crewai)

[← Back to Migration Guides](../) | [Compare More Frameworks →](https://www.agentically.sh/ai-agentic-frameworks/compare/)