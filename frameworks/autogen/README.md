# AutoGen Framework Guide

[![GitHub Stars](https://img.shields.io/github/stars/microsoft/autogen)](https://github.com/microsoft/autogen)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[🔍 Compare with other frameworks →](https://www.agentically.sh/ai-agentic-frameworks/compare/autogen/)

AutoGen is a framework for building multi-agent conversational AI systems. It enables agents to converse with each other to solve tasks collaboratively, with support for human-in-the-loop workflows.

## Key Features

- 🗣️ **Conversational AI**: Agents communicate through natural language
- 🔄 **Multi-agent workflows**: Complex task decomposition and collaboration
- 👥 **Human-in-the-loop**: Seamless human intervention and oversight
- 🎛️ **Flexible orchestration**: Various conversation patterns and topologies
- 📊 **Code execution**: Built-in code interpreter and execution capabilities
- 🛡️ **Safety controls**: Content filtering and conversation moderation

## When to Use AutoGen

✅ **Best for:**
- Research and complex problem-solving workflows
- Code generation and debugging tasks
- Educational and tutoring applications
- Multi-step reasoning and analysis
- Human-AI collaborative workflows

❌ **Not ideal for:**
- Simple single-turn interactions
- Real-time chat applications
- High-throughput production systems
- Cost-sensitive applications

## Quick Start

### Installation

```bash
pip install pyautogen
```

### Basic Example

```python
import autogen

# Configure LLM
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-openai-api-key"
    }
]

llm_config = {"config_list": config_list, "temperature": 0}

# Create agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "workspace"},
    llm_config=llm_config,
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate fibonacci numbers and test it."
)
```

## Examples

- [Basic Conversation](./examples/basic-conversation.py) - Simple agent interaction
- [Code Generation](./examples/code-generation.py) - Collaborative coding workflow
- [Research Assistant](./examples/research-assistant.py) - Multi-agent research pipeline
- [Human-in-Loop](./examples/human-in-loop.py) - Interactive problem solving
- [Group Chat](./examples/group-chat.py) - Multiple agents collaboration

## Benchmarks

[View detailed benchmarks →](./benchmarks.md)

| Metric | AutoGen | Industry Average |
|--------|---------|------------------|
| Code Generation Quality | 92% | 78% |
| Problem Solving Accuracy | 89% | 72% |
| Human Collaboration Score | 9.5/10 | 6.2/10 |
| Setup Complexity | Medium | Low |
| Token Efficiency | 78% | 70% |

## Migration Guides

- [From LangChain to AutoGen](../../migration-guides/langchain-to-autogen.md)
- [From CrewAI to AutoGen](../../migration-guides/crewai-to-autogen.md)

## Core Concepts

### Agent Types

1. **AssistantAgent**: AI-powered agent for task execution
2. **UserProxyAgent**: Human proxy with optional auto-reply
3. **GroupChatManager**: Orchestrates multi-agent conversations
4. **Custom Agents**: Specialized agents for specific domains

### Conversation Patterns

```python
# Two-agent conversation
user_proxy.initiate_chat(assistant, message="Task description")

# Group chat with multiple agents
groupchat = autogen.GroupChat(
    agents=[user_proxy, assistant, critic],
    messages=[],
    max_round=10
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
user_proxy.initiate_chat(manager, message="Complex task requiring multiple perspectives")
```

## Production Considerations

### Scaling
- Use conversation caching for repeated patterns
- Implement agent pools for high-throughput scenarios
- Monitor conversation depth and token usage

### Safety & Control
```python
# Content filtering
def content_filter(sender, recipient, message):
    # Implement safety checks
    if "harmful_content" in message["content"]:
        return {"content": "Message filtered for safety"}
    return message

user_proxy.register_reply(assistant, content_filter)
```

### Cost Management
- Set conversation round limits
- Use model hierarchies (cheaper models for simple tasks)
- Implement early termination conditions

## Advanced Features

### Custom Tools Integration

```python
def web_search(query: str) -> str:
    """Custom web search tool."""
    # Implement search logic
    return f"Search results for: {query}"

# Register tool with agent
user_proxy.register_function({
    "name": "web_search",
    "description": "Search the web for information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        }
    }
}, web_search)
```

### Memory and Context

```python
# Persistent conversation history
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful assistant with memory of our previous conversations."
)
```

## Community & Support

- [GitHub Repository](https://github.com/microsoft/autogen)
- [Documentation](https://microsoft.github.io/autogen/)
- [Discord Community](https://discord.gg/autogen)
- [Research Papers](https://arxiv.org/abs/2308.08155)

## Further Reading

- [AutoGen vs CrewAI Comparison](https://www.agentically.sh/ai-agentic-frameworks/compare/autogen-vs-crewai/)
- [Production Deployment Guide](https://www.agentically.sh/ai-agentic-frameworks/autogen/production/)
- [Cost Calculator](https://www.agentically.sh/ai-agentic-frameworks/cost-calculator/?framework=autogen)

---

[← Back to Framework Comparison](../../) | [Compare AutoGen →](https://www.agentically.sh/ai-agentic-frameworks/compare/autogen/)