# Semantic Kernel Framework Guide

[![GitHub Stars](https://img.shields.io/github/stars/microsoft/semantic-kernel)](https://github.com/microsoft/semantic-kernel)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Languages](https://img.shields.io/badge/languages-C%23%20%7C%20Python%20%7C%20Java-blue.svg)](https://learn.microsoft.com/semantic-kernel/)

[🔍 Compare with other frameworks →](https://www.agentically.sh/ai-agentic-frameworks/compare/semantic-kernel/)

Semantic Kernel is Microsoft's open-source SDK that lets developers integrate cutting-edge LLM technology into applications. It supports C#, Python, and Java with enterprise-grade features and seamless integration with Microsoft's ecosystem.

## Key Features

- 🏢 **Enterprise Integration**: Built for enterprise applications with Azure integration
- 🌐 **Multi-Language Support**: Native support for C#, Python, and Java
- 🔌 **Plugin Architecture**: Extensible system with semantic and native functions
- 🧠 **Planning & Memory**: Automatic planning and persistent memory management
- 🔗 **Connector Ecosystem**: Rich set of connectors for Microsoft and third-party services
- 🛡️ **Enterprise Security**: Built-in security features and compliance support

## When to Use Semantic Kernel

✅ **Best for:**
- Enterprise applications requiring Microsoft ecosystem integration
- .NET applications and Azure cloud deployments
- Multi-language development teams (C#, Python, Java)
- Applications requiring robust plugin architecture
- Enterprise-grade security and compliance requirements

❌ **Not ideal for:**
- Pure Python-first development teams
- Quick prototyping and experimentation
- Non-Microsoft cloud environments (though still works)
- Simple single-LLM applications

## Quick Start

### Installation

#### Python
```bash
pip install semantic-kernel
```

#### C#
```bash
dotnet add package Microsoft.SemanticKernel
```

#### Java
```xml
<dependency>
    <groupId>com.microsoft.semantic-kernel</groupId>
    <artifactId>semantic-kernel-api</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Basic Example (Python)

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# Create kernel
kernel = sk.Kernel()

# Add AI service
kernel.add_service(OpenAIChatCompletion(
    ai_model_id="gpt-4",
    api_key="your-openai-key"
))

# Create semantic function
prompt = """
{{$input}}

Please summarize the above text in a clear and concise manner.
"""

summarize_function = kernel.create_function_from_prompt(
    function_name="summarize",
    plugin_name="text_plugin",
    prompt=prompt
)

# Execute function
result = await kernel.invoke(summarize_function, input="Your text here...")
print(result)
```

### Basic Example (C#)

```csharp
using Microsoft.SemanticKernel;

// Create kernel
var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion("gpt-4", "your-openai-key");
var kernel = builder.Build();

// Create semantic function
string prompt = @"
{{$input}}

Please summarize the above text in a clear and concise manner.
";

var summarizeFunction = kernel.CreateFunctionFromPrompt(prompt);

// Execute function
var result = await kernel.InvokeAsync(summarizeFunction, 
    new() { ["input"] = "Your text here..." });

Console.WriteLine(result);
```

## Examples

- [Basic Functions](./examples/basic-functions.py) - Semantic and native functions
- [Plugin System](./examples/plugin-system.py) - Creating and using plugins
- [Memory Integration](./examples/memory-integration.py) - Persistent memory usage
- [Planning Agent](./examples/planning-agent.py) - Automatic task planning
- [Enterprise Integration](./examples/enterprise-integration.cs) - Azure and .NET integration

## Benchmarks

[View detailed benchmarks →](./benchmarks.md)

| Metric | Semantic Kernel | Industry Average |
|--------|-----------------|------------------|
| Enterprise Integration | 9/10 | 6/10 |
| Multi-Language Support | 9/10 | 4/10 |
| Plugin Ecosystem | 8/10 | 6/10 |
| Azure Integration | 10/10 | 5/10 |
| Learning Curve | 7/10 | 8/10 |

## Migration Guides

- [From LangChain to Semantic Kernel](../../migration-guides/langchain-to-semantic-kernel.md)
- [From .NET to Semantic Kernel](../../migration-guides/dotnet-to-semantic-kernel.md)
- [From Azure OpenAI to Semantic Kernel](../../migration-guides/azure-openai-to-semantic-kernel.md)

## Core Concepts

### Kernel
The central orchestrator that manages AI services, plugins, and execution:

```python
import semantic_kernel as sk

kernel = sk.Kernel()

# Add multiple AI services
kernel.add_service(OpenAIChatCompletion(...))
kernel.add_service(AzureOpenAIChatCompletion(...))
```

### Functions
Two types of functions for different use cases:

#### Semantic Functions (Prompt-based)
```python
# Create from prompt template
prompt = """
Analyze the sentiment of this text: {{$input}}
Return: positive, negative, or neutral
"""

sentiment_function = kernel.create_function_from_prompt(
    function_name="analyze_sentiment",
    plugin_name="sentiment_plugin",
    prompt=prompt
)
```

#### Native Functions (Code-based)
```python
from semantic_kernel.functions import kernel_function

class MathPlugin:
    @kernel_function(
        description="Add two numbers",
        name="add"
    )
    def add(self, a: int, b: int) -> int:
        return a + b

# Register plugin
kernel.add_plugin(MathPlugin(), plugin_name="math")
```

### Memory and Embeddings
```python
from semantic_kernel.memory import VolatileMemoryStore
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

# Set up memory
memory_store = VolatileMemoryStore()
embedding_service = OpenAITextEmbedding(
    ai_model_id="text-embedding-ada-002",
    api_key="your-key"
)

kernel.add_service(embedding_service)
kernel.import_plugin_from_object(memory_store, "memory")

# Store and recall information
await kernel.memory.save_information_async(
    collection="facts",
    id="fact1",
    text="The capital of France is Paris"
)

memories = await kernel.memory.search_async(
    collection="facts",
    query="capital of France"
)
```

## Advanced Features

### Planners
Automatic task decomposition and execution:

```python
from semantic_kernel.planners import SequentialPlanner

# Create planner
planner = SequentialPlanner(kernel)

# Generate plan
plan = await planner.create_plan_async("Write a blog post about AI")

# Execute plan
result = await kernel.invoke(plan)
```

### Connectors and Integrations
```python
# Azure integrations
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchMemoryStore

# Database connectors
from semantic_kernel.connectors.memory.postgres import PostgresMemoryStore
from semantic_kernel.connectors.memory.redis import RedisMemoryStore

# Vector database connectors
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore
from semantic_kernel.connectors.memory.pinecone import PineconeMemoryStore
```

### Enterprise Security
```python
from azure.identity import DefaultAzureCredential
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion

# Use Azure managed identity
credential = DefaultAzureCredential()

chat_service = AzureAIInferenceChatCompletion(
    endpoint="your-azure-endpoint",
    credential=credential,  # No API key needed
    ai_model_id="gpt-4"
)

kernel.add_service(chat_service)
```

## Production Patterns

### Plugin Development
```python
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata

class CustomerServicePlugin:
    """Plugin for customer service operations."""
    
    @kernel_function(
        description="Get customer information by ID",
        name="get_customer"
    )
    def get_customer(
        self, 
        customer_id: Annotated[str, "The customer ID to lookup"]
    ) -> str:
        # Integration with customer database
        return self._fetch_customer_data(customer_id)
    
    @kernel_function(
        description="Create support ticket",
        name="create_ticket"
    )
    def create_ticket(
        self,
        customer_id: Annotated[str, "Customer ID"],
        issue: Annotated[str, "Issue description"]
    ) -> str:
        # Create ticket in support system
        return self._create_support_ticket(customer_id, issue)

# Register plugin
kernel.add_plugin(CustomerServicePlugin(), plugin_name="customer_service")
```

### Error Handling and Resilience
```python
from semantic_kernel.exceptions import ServiceResponseException
import asyncio

async def resilient_function_call(kernel, function, **kwargs):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            result = await kernel.invoke(function, **kwargs)
            return result
        except ServiceResponseException as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_delay * (2 ** attempt))
        except Exception as e:
            # Log error and re-raise
            logger.error(f"Unexpected error: {e}")
            raise
```

### Monitoring and Observability
```python
import logging
from semantic_kernel.diagnostics import KernelEvents

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monitor kernel events
def on_function_invoked(sender, event_args):
    logger.info(f"Function invoked: {event_args.function.name}")

def on_function_completed(sender, event_args):
    logger.info(f"Function completed: {event_args.function.name}")

kernel.events.function_invoked += on_function_invoked
kernel.events.function_completed += on_function_completed
```

## Use Cases

### Enterprise Chatbots
Build sophisticated enterprise chatbots with:
- Integration with corporate databases
- Role-based access control
- Audit logging and compliance
- Multi-language support

### Document Processing Workflows
Automate document workflows:
- OCR and text extraction
- Content classification
- Automated summarization
- Compliance checking

### Business Process Automation
Streamline business processes:
- Automated decision making
- Workflow orchestration
- Data validation and processing
- Report generation

## Integration Examples

### Azure Integration
```csharp
using Microsoft.SemanticKernel;
using Azure.Identity;

// Azure integration with managed identity
var builder = Kernel.CreateBuilder();

builder.AddAzureOpenAIChatCompletion(
    "gpt-4",
    "https://your-resource.openai.azure.com/",
    new DefaultAzureCredential()
);

var kernel = builder.Build();
```

### Database Integration
```python
import asyncpg
from semantic_kernel.functions import kernel_function

class DatabasePlugin:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    @kernel_function(
        description="Query customer data from database"
    )
    async def query_customers(self, query: str) -> str:
        conn = await asyncpg.connect(self.connection_string)
        try:
            results = await conn.fetch(query)
            return str(results)
        finally:
            await conn.close()
```

### Microsoft Graph Integration
```python
from semantic_kernel.functions import kernel_function
from microsoft.graph import GraphServiceClient

class GraphPlugin:
    def __init__(self, graph_client: GraphServiceClient):
        self.graph_client = graph_client
    
    @kernel_function(
        description="Get user's calendar events"
    )
    async def get_calendar_events(self, user_id: str) -> str:
        events = await self.graph_client.users[user_id].calendar.events.get()
        return str([event.subject for event in events.value])
```

## Performance Optimization

### Caching Strategies
```python
from functools import lru_cache
import asyncio

class CachedPlugin:
    @lru_cache(maxsize=128)
    def cached_expensive_operation(self, input_data: str) -> str:
        # Expensive computation here
        return processed_result
    
    @kernel_function(description="Cached operation")
    def get_processed_data(self, input_data: str) -> str:
        return self.cached_expensive_operation(input_data)
```

### Batch Processing
```python
async def batch_process_documents(kernel, documents: list, batch_size: int = 10):
    """Process documents in batches for better performance."""
    
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        # Process batch concurrently
        tasks = [
            kernel.invoke(process_function, document=doc)
            for doc in batch
        ]
        
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results
```

## Community & Support

- [GitHub Repository](https://github.com/microsoft/semantic-kernel) - 26.3k+ stars
- [Official Documentation](https://learn.microsoft.com/semantic-kernel/) - Microsoft Learn
- [Discord Community](https://discord.gg/semantic-kernel) - Active community support
- [Microsoft Q&A](https://learn.microsoft.com/answers/tags/semantic-kernel/) - Official support

## Enterprise Features

### Azure AI Studio Integration
- Visual plugin development
- Model management and deployment
- Performance monitoring and analytics
- Cost optimization insights

### Security and Compliance
- Azure Active Directory integration
- Role-based access control (RBAC)
- Data encryption and privacy controls
- Audit logging and compliance reporting

### Scaling and Deployment
- Azure Container Instances
- Azure Kubernetes Service (AKS)
- Azure Functions serverless deployment
- Auto-scaling and load balancing

## Best Practices

### 1. Plugin Design
Structure plugins for reusability and maintainability:
```python
class WellDesignedPlugin:
    """Clear description of plugin purpose."""
    
    def __init__(self, config: dict):
        self.config = config
        self._validate_config()
    
    @kernel_function(
        description="Clear description of what this function does",
        name="descriptive_name"
    )
    def well_documented_function(
        self, 
        param: Annotated[str, "Clear parameter description"]
    ) -> str:
        """Detailed docstring explaining the function."""
        return self._internal_logic(param)
    
    def _validate_config(self):
        """Validate configuration on initialization."""
        required_keys = ["api_key", "endpoint"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
```

### 2. Error Handling
Implement comprehensive error handling:
```python
from semantic_kernel.exceptions import KernelException

async def robust_kernel_operation():
    try:
        result = await kernel.invoke(function, **parameters)
        return result
    except KernelException as e:
        logger.error(f"Kernel error: {e.message}")
        # Implement fallback logic
        return fallback_response
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### 3. Testing Strategies
```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_kernel():
    kernel = Mock()
    kernel.invoke = Mock(return_value="mocked result")
    return kernel

async def test_plugin_function(mock_kernel):
    plugin = MyPlugin()
    result = await plugin.my_function("test input")
    
    assert result == "expected output"
    mock_kernel.invoke.assert_called_once()
```

## Further Reading

- [Semantic Kernel vs LangChain Comparison](https://www.agentically.sh/ai-agentic-frameworks/compare/semantic-kernel-vs-langchain/)
- [Enterprise Deployment Guide](https://www.agentically.sh/ai-agentic-frameworks/semantic-kernel/enterprise/)
- [Azure Integration Patterns](https://www.agentically.sh/ai-agentic-frameworks/semantic-kernel/azure/)
- [Performance Optimization Guide](https://www.agentically.sh/ai-agentic-frameworks/semantic-kernel/optimization/)

---

[← Back to Framework Comparison](../../) | [Compare Semantic Kernel →](https://www.agentically.sh/ai-agentic-frameworks/compare/semantic-kernel/)