# Migration Guide: CrewAI to AutoGen

## Overview

This guide helps you migrate from CrewAI to AutoGen, transitioning from role-based crew management to conversational multi-agent systems with flexible communication patterns.

## Key Conceptual Differences

| CrewAI | AutoGen | Notes |
|--------|---------|-------|
| **Crews** | **GroupChat** | Multi-agent collaboration |
| **Agents with Roles** | **AssistantAgent/UserProxy** | Conversation-focused agents |
| **Tasks** | **Conversations** | Message-based interactions |
| **Task Dependencies** | **Conversation Flow** | Natural conversation ordering |
| **Built-in Memory** | **ConversationHistory** | Message history tracking |

## Migration Steps

### 1. Convert CrewAI Agents to AutoGen Agents

**Before (CrewAI):**
```python
from crewai import Agent
from crewai_tools import SerperDevTool

research_agent = Agent(
    role="Research Specialist",
    goal="Find accurate information and data",
    backstory="You are an expert researcher with access to web search.",
    tools=[SerperDevTool()],
    verbose=True,
    allow_delegation=False
)
```

**After (AutoGen):**
```python
import autogen
from autogen import AssistantAgent, UserProxyAgent

research_agent = AssistantAgent(
    name="Research_Specialist",
    system_message="""You are a Research Specialist. Your goal is to find accurate 
    information and data. You are an expert researcher with access to web search 
    capabilities. When asked to research, provide comprehensive and factual information.""",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "research_output"}
)
```

### 2. Convert Crews and Tasks to GroupChat

**Before (CrewAI):**
```python
from crewai import Crew, Task

research_task = Task(
    description="Research the topic: {topic}",
    agent=research_agent,
    expected_output="Comprehensive research findings"
)

analysis_task = Task(
    description="Analyze the research findings",
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

**After (AutoGen):**
```python
from autogen import GroupChat, GroupChatManager

# Define agents
research_agent = AssistantAgent(
    name="Researcher",
    system_message="You are a research specialist. Research topics thoroughly.",
    llm_config={"model": "gpt-4"}
)

analyst_agent = AssistantAgent(
    name="Analyst", 
    system_message="You analyze research data and provide insights.",
    llm_config={"model": "gpt-4"}
)

# Create group chat
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

### 3. Tool Integration Migration

**Before (CrewAI):**
```python
from crewai_tools import BaseTool

class CustomSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information"
    
    def _run(self, query: str) -> str:
        return search_results
```

**After (AutoGen):**
```python
# AutoGen tool integration via function calling
def web_search(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: The search query
        
    Returns:
        Search results as a string
    """
    return search_results

research_agent = AssistantAgent(
    name="Researcher",
    system_message="You can search the web using the web_search function.",
    llm_config={
        "model": "gpt-4",
        "functions": [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        ]
    }
)

# Register the function
user_proxy.register_function(
    function_map={"web_search": web_search}
)
```

### 4. Conversation Flow Control

**Before (CrewAI):**
```python
# CrewAI task dependencies define execution order
task1 = Task(description="First task", agent=agent1)
task2 = Task(description="Second task", agent=agent2, dependencies=[task1])
```

**After (AutoGen):**
```python
# AutoGen uses conversation flow and speaker selection
def custom_speaker_selection(last_speaker, groupchat):
    """Custom logic for selecting next speaker"""
    if last_speaker.name == "Researcher":
        return analyst_agent  # Researcher passes to analyst
    elif last_speaker.name == "Analyst":
        return user_proxy  # Analyst passes to user proxy
    else:
        return research_agent  # Default to researcher

groupchat = GroupChat(
    agents=[user_proxy, research_agent, analyst_agent],
    messages=[],
    max_round=10,
    speaker_selection_method=custom_speaker_selection
)
```

## Complete Migration Example

### CrewAI Implementation

```python
# Original CrewAI implementation
from crewai import Agent, Crew, Task
from crewai_tools import SerperDevTool, CalculatorTool

# Agents
researcher = Agent(
    role="Research Analyst",
    goal="Conduct thorough research",
    backstory="Expert researcher specializing in technology trends.",
    tools=[SerperDevTool()],
    verbose=True
)

calculator = Agent(
    role="Data Analyst",
    goal="Perform calculations and data analysis",
    backstory="Skilled analyst with expertise in calculations.",
    tools=[CalculatorTool()],
    verbose=True
)

# Tasks
research_task = Task(
    description="Research AI market trends for 2024",
    agent=researcher,
    expected_output="Comprehensive market research report"
)

calculation_task = Task(
    description="Calculate market size and growth projections",
    agent=calculator,
    expected_output="Market calculations with projections",
    dependencies=[research_task]
)

# Crew
crew = Crew(
    agents=[researcher, calculator],
    tasks=[research_task, calculation_task],
    verbose=True
)

result = crew.kickoff()
```

### AutoGen Implementation

```python
# Migrated AutoGen implementation
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Define search and calculation functions
def web_search(query: str) -> str:
    """Search the web for information"""
    # Implementation here
    return f"Search results for: {query}"

def calculate(expression: str) -> float:
    """Perform mathematical calculations"""
    try:
        return eval(expression)
    except:
        return "Calculation error"

# Agents
researcher = AssistantAgent(
    name="Research_Analyst",
    system_message="""You are a Research Analyst. Your goal is to conduct thorough 
    research on technology trends. Use the web_search function to find current 
    information about market trends, company data, and industry reports.""",
    llm_config={
        "model": "gpt-4",
        "functions": [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        ]
    }
)

calculator_agent = AssistantAgent(
    name="Data_Analyst",
    system_message="""You are a Data Analyst specializing in market calculations. 
    Use the calculate function for mathematical operations. Analyze data provided 
    by the researcher and create projections.""",
    llm_config={
        "model": "gpt-4",
        "functions": [
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression"}
                    },
                    "required": ["expression"]
                }
            }
        ]
    }
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config={"work_dir": "analysis_output"}
)

# Register functions
user_proxy.register_function(
    function_map={
        "web_search": web_search,
        "calculate": calculate
    }
)

# Group chat with custom speaker selection
def speaker_selection(last_speaker, groupchat):
    if last_speaker.name == "user_proxy":
        return researcher
    elif last_speaker.name == "Research_Analyst":
        return calculator_agent
    else:
        return user_proxy

groupchat = GroupChat(
    agents=[user_proxy, researcher, calculator_agent],
    messages=[],
    max_round=12,
    speaker_selection_method=speaker_selection
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"model": "gpt-4"}
)

# Execution
result = user_proxy.initiate_chat(
    manager,
    message="Research AI market trends for 2024 and calculate market size projections"
)
```

## Key Benefits of Migration

### 1. Flexible Conversation Flow
```python
# AutoGen allows dynamic conversation patterns
def adaptive_speaker_selection(last_speaker, groupchat):
    messages = groupchat.messages
    
    # If research is mentioned, go to researcher
    if "research" in messages[-1]["content"].lower():
        return researcher
    
    # If calculation needed, go to calculator
    if any(word in messages[-1]["content"].lower() 
           for word in ["calculate", "compute", "math"]):
        return calculator_agent
    
    return user_proxy

groupchat.speaker_selection_method = adaptive_speaker_selection
```

### 2. Code Execution Capabilities
```python
# AutoGen can execute code directly
code_executor = UserProxyAgent(
    name="code_executor",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "code_output",
        "use_docker": True  # Safe code execution
    }
)
```

### 3. Human-in-the-Loop Integration
```python
# Easy human intervention
human_reviewer = UserProxyAgent(
    name="human_reviewer",
    human_input_mode="ALWAYS",  # Always ask for human input
    max_consecutive_auto_reply=0
)
```

## Advanced Migration Patterns

### 1. Hierarchical Agent Structure

**CrewAI Hierarchical Process:**
```python
crew = Crew(
    agents=[manager, worker1, worker2],
    tasks=[task1, task2, task3],
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4")
)
```

**AutoGen Equivalent:**
```python
# Manager agent coordinates work
manager = AssistantAgent(
    name="Manager",
    system_message="""You are a project manager. Coordinate work between 
    specialists and ensure tasks are completed efficiently.""",
    llm_config={"model": "gpt-4"}
)

# Nested group chats for hierarchical structure
sub_team = GroupChat(
    agents=[worker1, worker2],
    messages=[],
    max_round=5
)

main_chat = GroupChat(
    agents=[manager, GroupChatManager(sub_team)],
    messages=[],
    max_round=10
)
```

### 2. Parallel Processing

**CrewAI Parallel Tasks:**
```python
# CrewAI can run independent tasks in parallel
task1 = Task(description="Independent task 1", agent=agent1)
task2 = Task(description="Independent task 2", agent=agent2)
# No dependencies = parallel execution
```

**AutoGen Equivalent:**
```python
import asyncio

async def parallel_conversations():
    # Create separate conversations for parallel work
    conv1 = user_proxy.a_initiate_chat(agent1, message="Task 1")
    conv2 = user_proxy.a_initiate_chat(agent2, message="Task 2")
    
    # Run in parallel
    results = await asyncio.gather(conv1, conv2)
    return results
```

## Common Migration Challenges

### 1. Task Dependency Management

**Challenge:** CrewAI's explicit task dependencies vs AutoGen's conversation flow.

**Solution:**
```python
# Use conversation context to manage dependencies
def context_aware_speaker_selection(last_speaker, groupchat):
    messages = groupchat.messages
    
    # Check if prerequisite work is completed
    research_complete = any("research complete" in msg["content"].lower() 
                           for msg in messages)
    
    if not research_complete:
        return researcher
    else:
        return analyst

groupchat.speaker_selection_method = context_aware_speaker_selection
```

### 2. Tool Integration Complexity

**Challenge:** CrewAI's simple tool system vs AutoGen's function calling.

**Solution:**
```python
# Create tool wrapper for easier migration
class ToolWrapper:
    def __init__(self, crewai_tool):
        self.tool = crewai_tool
    
    def to_autogen_function(self):
        return {
            "name": self.tool.name,
            "description": self.tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Tool input"}
                },
                "required": ["input"]
            }
        }
    
    def execute(self, input_str):
        return self.tool._run(input_str)

# Usage
crewai_tool = SerperDevTool()
wrapper = ToolWrapper(crewai_tool)
autogen_function = wrapper.to_autogen_function()
```

### 3. Memory and Context Management

**Challenge:** CrewAI's built-in crew memory vs AutoGen's conversation history.

**Solution:**
```python
# Implement persistent memory for AutoGen
class PersistentMemory:
    def __init__(self):
        self.memory = {}
    
    def store(self, key, value):
        self.memory[key] = value
    
    def retrieve(self, key):
        return self.memory.get(key, "No information found")

memory = PersistentMemory()

def memory_enhanced_agent():
    return AssistantAgent(
        name="MemoryAgent",
        system_message=f"""You have access to persistent memory. 
        Current memory: {memory.memory}
        Use this information to maintain context across conversations.""",
        llm_config={"model": "gpt-4"}
    )
```

## Testing Your Migration

### 1. Conversation Flow Testing
```python
def test_conversation_flow():
    # Test that agents communicate in expected order
    conversation = user_proxy.initiate_chat(
        manager,
        message="Test conversation flow"
    )
    
    # Verify message sequence
    messages = groupchat.messages
    assert len(messages) > 0
    
    # Check agent participation
    speakers = [msg.get("name") for msg in messages]
    assert "Research_Analyst" in speakers
    assert "Data_Analyst" in speakers

def test_function_calling():
    # Test that tools are called correctly
    result = user_proxy.initiate_chat(
        researcher,
        message="Search for information about AI trends"
    )
    
    # Verify function was called
    assert any("web_search" in str(msg) for msg in groupchat.messages)
```

### 2. Output Quality Comparison
```python
def compare_outputs(crewai_result, autogen_result):
    # Compare output quality metrics
    metrics = {
        "completeness": score_completeness(crewai_result, autogen_result),
        "accuracy": score_accuracy(crewai_result, autogen_result),
        "coherence": score_coherence(crewai_result, autogen_result)
    }
    return metrics
```

## Best Practices for Migration

1. **Preserve Agent Roles**: Maintain role clarity in system messages
2. **Design Conversation Flow**: Plan speaker selection carefully
3. **Test Iteratively**: Migrate one agent/conversation at a time
4. **Monitor Performance**: Track conversation length and quality
5. **Use Code Execution**: Leverage AutoGen's code execution for calculations

## Troubleshooting Common Issues

### Issue: Agents Not Following Conversation Flow
```python
# Add explicit conversation management
def strict_speaker_selection(last_speaker, groupchat):
    # Define strict speaking order
    order = ["user_proxy", "Research_Analyst", "Data_Analyst"]
    current_index = order.index(last_speaker.name)
    next_index = (current_index + 1) % len(order)
    
    agent_map = {
        "user_proxy": user_proxy,
        "Research_Analyst": researcher,
        "Data_Analyst": calculator_agent
    }
    
    return agent_map[order[next_index]]
```

### Issue: Function Calling Not Working
```python
# Ensure proper function registration
user_proxy.register_function(
    function_map={"web_search": web_search},
    auto_exec=True  # Automatically execute functions
)
```

### Issue: Conversation Too Long
```python
# Set appropriate limits
groupchat = GroupChat(
    agents=[user_proxy, researcher, analyst],
    messages=[],
    max_round=6,  # Limit conversation length
    speaker_selection_method="round_robin"
)
```

---

**Next Steps:**
- Review [AutoGen documentation](../frameworks/autogen/README.md)
- Explore [AutoGen examples](../frameworks/autogen/examples/)
- Test conversation flows thoroughly