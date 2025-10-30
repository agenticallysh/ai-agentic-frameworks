#!/usr/bin/env python3
"""
API Integration Example - Dify
===============================

This example demonstrates comprehensive integration with Dify's APIs,
including apps, workflows, datasets, and various management operations.
Shows how to build applications that leverage Dify's full API ecosystem.

Requirements:
    pip install requests python-dotenv

Usage:
    python api-integration.py
"""

import os
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import requests


@dataclass
class DifyApp:
    """Represents a Dify application."""
    id: str
    name: str
    mode: str  # 'chat', 'workflow', 'agent-chat', 'completion'
    icon: str
    description: str
    status: str = "normal"
    created_at: str = None
    updated_at: str = None


@dataclass
class DifyDataset:
    """Represents a Dify dataset/knowledge base."""
    id: str
    name: str
    description: str
    provider: str
    permission: str = "only_me"
    document_count: int = 0
    word_count: int = 0
    created_at: str = None


class DifyAPIClient:
    """
    Comprehensive Dify API client for managing applications, workflows,
    datasets, and other platform resources.
    
    This client provides a complete interface to Dify's API ecosystem,
    enabling full application lifecycle management.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.dify.ai/v1"
    ):
        """
        Initialize the Dify API client.
        
        Args:
            api_key: Your Dify API key
            base_url: Dify API base URL
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        # Track created resources for cleanup
        self.apps: Dict[str, DifyApp] = {}
        self.datasets: Dict[str, DifyDataset] = {}
    
    # App Management APIs
    def list_apps(self, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        """List all applications."""
        
        try:
            response = self.session.get(
                f"{self.base_url}/apps",
                params={"page": page, "limit": limit},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            apps_data = result.get('data', [])
            
            # Update local app tracking
            for app_data in apps_data:
                app = DifyApp(
                    id=app_data['id'],
                    name=app_data['name'],
                    mode=app_data['mode'],
                    icon=app_data.get('icon', '🤖'),
                    description=app_data.get('description', ''),
                    status=app_data.get('status', 'normal'),
                    created_at=app_data.get('created_at'),
                    updated_at=app_data.get('updated_at')
                )
                self.apps[app.id] = app
            
            return apps_data
            
        except requests.RequestException as e:
            print(f"❌ Error listing apps: {e}")
            return []
    
    def get_app(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific app."""
        
        try:
            response = self.session.get(
                f"{self.base_url}/apps/{app_id}",
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            print(f"❌ Error getting app {app_id}: {e}")
            return None
    
    def create_chat_app(
        self,
        name: str,
        description: str = "",
        model_config: Dict[str, Any] = None,
        prompt_template: str = None
    ) -> Optional[str]:
        """Create a new chat application."""
        
        if not model_config:
            model_config = {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            }
        
        if not prompt_template:
            prompt_template = "You are a helpful AI assistant. Answer questions clearly and helpfully."
        
        app_config = {
            "name": name,
            "description": description,
            "mode": "chat",
            "icon": "🤖",
            "model_config": model_config,
            "prompt_template": prompt_template,
            "opening_statement": f"Hello! I'm {name}. How can I help you today?",
            "suggested_questions": [
                "What can you help me with?",
                "Tell me about your capabilities",
                "How do I get started?"
            ]
        }
        
        return self._create_app(app_config)
    
    def create_workflow_app(
        self,
        name: str,
        description: str = "",
        workflow_config: Dict[str, Any] = None
    ) -> Optional[str]:
        """Create a new workflow application."""
        
        if not workflow_config:
            # Default simple workflow
            workflow_config = {
                "type": "workflow",
                "nodes": [
                    {
                        "id": "start",
                        "type": "start",
                        "data": {
                            "title": "Input",
                            "variables": [
                                {
                                    "variable": "input_text",
                                    "label": "Input Text",
                                    "type": "text-input",
                                    "required": True
                                }
                            ]
                        }
                    },
                    {
                        "id": "llm",
                        "type": "llm",
                        "data": {
                            "title": "Process",
                            "model": {
                                "provider": "openai",
                                "name": "gpt-3.5-turbo",
                                "parameters": {
                                    "temperature": 0.7,
                                    "max_tokens": 500
                                }
                            },
                            "prompt_template": [
                                {
                                    "role": "system",
                                    "text": "Process the user input and provide a helpful response."
                                },
                                {
                                    "role": "user",
                                    "text": "{{#start.input_text#}}"
                                }
                            ]
                        }
                    },
                    {
                        "id": "end",
                        "type": "end",
                        "data": {
                            "title": "Output",
                            "outputs": [
                                {
                                    "variable": "result",
                                    "type": "text",
                                    "value_selector": ["llm", "text"]
                                }
                            ]
                        }
                    }
                ],
                "edges": [
                    {"id": "start-llm", "source": "start", "target": "llm"},
                    {"id": "llm-end", "source": "llm", "target": "end"}
                ]
            }
        
        app_config = {
            "name": name,
            "description": description,
            "mode": "workflow",
            "icon": "⚙️",
            "workflow_config": workflow_config
        }
        
        return self._create_app(app_config)
    
    def _create_app(self, config: Dict[str, Any]) -> Optional[str]:
        """Internal method to create an app."""
        
        try:
            response = self.session.post(
                f"{self.base_url}/apps",
                json=config,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            app_id = result.get('id')
            
            if app_id:
                app = DifyApp(
                    id=app_id,
                    name=config['name'],
                    mode=config['mode'],
                    icon=config.get('icon', '🤖'),
                    description=config.get('description', '')
                )
                self.apps[app_id] = app
                
                print(f"✅ Created app: {config['name']} (ID: {app_id})")
            
            return app_id
            
        except requests.RequestException as e:
            print(f"❌ Error creating app: {e}")
            return None
    
    def delete_app(self, app_id: str) -> bool:
        """Delete an application."""
        
        try:
            response = self.session.delete(
                f"{self.base_url}/apps/{app_id}",
                timeout=30
            )
            response.raise_for_status()
            
            # Remove from local tracking
            if app_id in self.apps:
                app_name = self.apps[app_id].name
                del self.apps[app_id]
                print(f"✅ Deleted app: {app_name}")
            
            return True
            
        except requests.RequestException as e:
            print(f"❌ Error deleting app {app_id}: {e}")
            return False
    
    # Chat and Workflow Execution APIs
    def send_chat_message(
        self,
        app_id: str,
        message: str,
        conversation_id: str = None,
        user_id: str = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Send a message to a chat application."""
        
        if not user_id:
            user_id = f"api_user_{uuid.uuid4().hex[:8]}"
        
        payload = {
            "inputs": {},
            "query": message,
            "response_mode": "streaming" if stream else "blocking",
            "conversation_id": conversation_id,
            "user": user_id,
            "auto_generate_name": True
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat-messages",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                result = response.json()
                return {
                    "answer": result.get('answer', ''),
                    "conversation_id": result.get('conversation_id'),
                    "message_id": result.get('id'),
                    "usage": result.get('metadata', {}).get('usage', {})
                }
            
        except requests.RequestException as e:
            return {"error": str(e), "answer": "", "conversation_id": conversation_id}
    
    def run_workflow(
        self,
        app_id: str,
        inputs: Dict[str, Any],
        user_id: str = None
    ) -> Dict[str, Any]:
        """Execute a workflow application."""
        
        if not user_id:
            user_id = f"workflow_user_{uuid.uuid4().hex[:8]}"
        
        payload = {
            "inputs": inputs,
            "response_mode": "blocking",
            "user": user_id
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/workflows/run",
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                "success": True,
                "outputs": result.get('data', {}).get('outputs', {}),
                "workflow_id": result.get('workflow_run_id'),
                "usage": result.get('data', {}).get('total_tokens', 0)
            }
            
        except requests.RequestException as e:
            return {"success": False, "error": str(e), "outputs": {}}
    
    # Dataset Management APIs
    def list_datasets(self, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        """List all datasets."""
        
        try:
            response = self.session.get(
                f"{self.base_url}/datasets",
                params={"page": page, "limit": limit},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            datasets_data = result.get('data', [])
            
            # Update local dataset tracking
            for dataset_data in datasets_data:
                dataset = DifyDataset(
                    id=dataset_data['id'],
                    name=dataset_data['name'],
                    description=dataset_data.get('description', ''),
                    provider=dataset_data.get('provider', 'vendor'),
                    permission=dataset_data.get('permission', 'only_me'),
                    document_count=dataset_data.get('document_count', 0),
                    word_count=dataset_data.get('word_count', 0),
                    created_at=dataset_data.get('created_at')
                )
                self.datasets[dataset.id] = dataset
            
            return datasets_data
            
        except requests.RequestException as e:
            print(f"❌ Error listing datasets: {e}")
            return []
    
    def create_dataset(
        self,
        name: str,
        description: str = "",
        indexing_technique: str = "high_quality",
        embedding_model: str = "text-embedding-ada-002",
        embedding_model_provider: str = "openai"
    ) -> Optional[str]:
        """Create a new dataset."""
        
        payload = {
            "name": name,
            "description": description,
            "indexing_technique": indexing_technique,
            "embedding_model": embedding_model,
            "embedding_model_provider": embedding_model_provider,
            "retrieval_model": {
                "search_method": "semantic_search",
                "reranking_enable": True,
                "reranking_model": {
                    "reranking_provider_name": "cohere",
                    "reranking_model_name": "rerank-english-v2.0"
                },
                "top_k": 5,
                "score_threshold_enabled": True,
                "score_threshold": 0.5
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/datasets",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            dataset_id = result.get('id')
            
            if dataset_id:
                dataset = DifyDataset(
                    id=dataset_id,
                    name=name,
                    description=description,
                    provider=embedding_model_provider
                )
                self.datasets[dataset_id] = dataset
                
                print(f"✅ Created dataset: {name} (ID: {dataset_id})")
            
            return dataset_id
            
        except requests.RequestException as e:
            print(f"❌ Error creating dataset: {e}")
            return None
    
    def add_document_to_dataset(
        self,
        dataset_id: str,
        content: str,
        name: str,
        indexing_technique: str = "high_quality"
    ) -> Optional[str]:
        """Add a text document to a dataset."""
        
        payload = {
            "name": name,
            "text": content,
            "indexing_technique": indexing_technique,
            "process_rule": {
                "rules": {
                    "pre_processing_rules": [
                        {"id": "remove_extra_spaces", "enabled": True},
                        {"id": "remove_urls_emails", "enabled": True}
                    ],
                    "segmentation": {
                        "separator": "\\n",
                        "max_tokens": 1000,
                        "chunk_overlap": 100
                    }
                },
                "mode": "automatic"
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/datasets/{dataset_id}/document/create_by_text",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            document_id = result.get('document', {}).get('id')
            
            if document_id:
                print(f"✅ Added document: {name} to dataset {dataset_id}")
            
            return document_id
            
        except requests.RequestException as e:
            print(f"❌ Error adding document: {e}")
            return None
    
    def query_dataset(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Query a dataset for relevant documents."""
        
        payload = {
            "query": query,
            "retrieval_model": {
                "search_method": "semantic_search",
                "reranking_enable": True,
                "reranking_model": {
                    "reranking_provider_name": "cohere",
                    "reranking_model_name": "rerank-english-v2.0"
                },
                "top_k": top_k,
                "score_threshold_enabled": True,
                "score_threshold": score_threshold
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/datasets/{dataset_id}/retrieve",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            return {"error": str(e), "query": {"records": []}}
    
    # Message and Conversation Management APIs
    def get_conversations(
        self,
        app_id: str,
        user_id: str = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get conversations for an app."""
        
        params = {"limit": limit}
        if user_id:
            params["user"] = user_id
        
        try:
            response = self.session.get(
                f"{self.base_url}/conversations",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
            
        except requests.RequestException as e:
            print(f"❌ Error getting conversations: {e}")
            return []
    
    def get_conversation_messages(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get messages from a conversation."""
        
        try:
            response = self.session.get(
                f"{self.base_url}/messages",
                params={"conversation_id": conversation_id, "limit": limit},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
            
        except requests.RequestException as e:
            print(f"❌ Error getting messages: {e}")
            return []
    
    # Utility Methods
    def get_app_statistics(self, app_id: str) -> Dict[str, Any]:
        """Get usage statistics for an app."""
        
        try:
            response = self.session.get(
                f"{self.base_url}/apps/{app_id}/statistics",
                params={"period": "30d"},
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            print(f"❌ Error getting app statistics: {e}")
            return {}
    
    def _handle_streaming_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle streaming response from Dify."""
        
        answer_parts = []
        conversation_id = None
        message_id = None
        
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    data_str = line[6:]
                    
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        event = data.get('event')
                        
                        if event == 'message':
                            chunk = data.get('answer', '')
                            if chunk:
                                answer_parts.append(chunk)
                        
                        elif event == 'message_end':
                            conversation_id = data.get('conversation_id')
                            message_id = data.get('id')
                            
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
        
        return {
            "answer": ''.join(answer_parts),
            "conversation_id": conversation_id,
            "message_id": message_id,
            "streaming": True
        }
    
    def cleanup_resources(self):
        """Clean up all created resources."""
        
        print("\n🧹 Cleaning up resources...")
        
        # Delete apps
        for app_id, app in list(self.apps.items()):
            if self.delete_app(app_id):
                print(f"   ✅ Deleted app: {app.name}")
        
        # Note: Datasets are not deleted in cleanup to preserve data
        if self.datasets:
            print(f"   ℹ️ Datasets preserved: {len(self.datasets)} datasets")
            for dataset_id, dataset in self.datasets.items():
                print(f"      • {dataset.name} (ID: {dataset_id})")
    
    def print_summary(self):
        """Print a summary of all resources."""
        
        print("\n📊 API Integration Summary")
        print("=" * 40)
        
        print(f"🤖 Applications: {len(self.apps)}")
        for app_id, app in self.apps.items():
            print(f"   • {app.name} ({app.mode}) - ID: {app_id}")
        
        print(f"\n📚 Datasets: {len(self.datasets)}")
        for dataset_id, dataset in self.datasets.items():
            print(f"   • {dataset.name} ({dataset.document_count} docs) - ID: {dataset_id}")


def comprehensive_api_demo():
    """Run a comprehensive demonstration of Dify API integration."""
    
    print("🤖 Dify Comprehensive API Integration Demo")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('DIFY_API_KEY')
    if not api_key:
        api_key = input("Enter your Dify API key: ").strip()
    
    if not api_key:
        print("❌ API key is required")
        return
    
    # Initialize API client
    client = DifyAPIClient(api_key=api_key)
    
    try:
        # 1. App Management Demo
        print("\n🏗️ Part 1: Application Management")
        print("-" * 40)
        
        # List existing apps
        print("📋 Listing existing applications...")
        existing_apps = client.list_apps()
        print(f"   Found {len(existing_apps)} existing apps")
        
        # Create a chat app
        print("\n🤖 Creating chat application...")
        chat_app_id = client.create_chat_app(
            name="API Demo Chat Bot",
            description="A demonstration chatbot created via API",
            prompt_template="You are a helpful AI assistant created through the Dify API. You can answer questions about API integration and help users understand how to work with Dify programmatically."
        )
        
        # Create a workflow app
        print("\n⚙️ Creating workflow application...")
        workflow_app_id = client.create_workflow_app(
            name="API Demo Workflow",
            description="A demonstration workflow created via API"
        )
        
        # 2. Dataset Management Demo
        print("\n📚 Part 2: Dataset Management")
        print("-" * 40)
        
        # Create a dataset
        print("🗄️ Creating knowledge base...")
        dataset_id = client.create_dataset(
            name="API Integration Knowledge Base",
            description="Knowledge base for API integration examples"
        )
        
        if dataset_id:
            # Add sample documents
            sample_docs = [
                {
                    "name": "API Basics",
                    "content": """
# Dify API Integration Basics

The Dify API provides comprehensive access to the platform's capabilities, allowing developers to:

1. **Application Management**: Create, update, and manage AI applications
2. **Workflow Execution**: Run complex workflows programmatically  
3. **Dataset Operations**: Manage knowledge bases and documents
4. **Conversation Handling**: Manage chat sessions and message history
5. **Monitoring & Analytics**: Track usage and performance metrics

## Authentication
All API requests require a valid API key in the Authorization header:
```
Authorization: Bearer YOUR_API_KEY
```

## Rate Limits
- Free tier: 100 requests per minute
- Pro tier: 1000 requests per minute
- Enterprise: Custom limits

## Best Practices
- Use appropriate timeout values
- Implement retry logic for transient errors
- Cache responses when possible
- Monitor API usage and costs
                    """
                },
                {
                    "name": "Chat API Guide",
                    "content": """
# Chat API Integration Guide

The Chat API enables real-time conversations with AI applications.

## Key Endpoints
- `POST /chat-messages` - Send chat messages
- `GET /conversations` - List conversations
- `GET /messages` - Get conversation history

## Message Flow
1. Send user message with conversation context
2. Receive AI response with updated conversation ID
3. Maintain conversation ID for context continuity

## Streaming Responses
Enable streaming for real-time response delivery:
```python
payload = {
    "response_mode": "streaming",
    "query": "user message"
}
```

## Error Handling
Common errors and solutions:
- 400: Invalid request format
- 401: Invalid API key
- 429: Rate limit exceeded
- 500: Server error - retry with backoff
                    """
                },
                {
                    "name": "Workflow API Guide", 
                    "content": """
# Workflow API Integration Guide

Workflows enable complex multi-step AI processes with conditional logic, loops, and integrations.

## Workflow Components
- **Start Node**: Define input variables
- **LLM Nodes**: AI processing steps
- **Tool Nodes**: External integrations
- **Condition Nodes**: Branching logic
- **End Node**: Define outputs

## Execution Modes
- **Blocking**: Wait for complete execution
- **Streaming**: Real-time step updates

## Input Variables
Define dynamic inputs for flexible workflows:
```json
{
  "inputs": {
    "user_query": "What is machine learning?",
    "processing_mode": "detailed"
  }
}
```

## Output Handling
Workflows return structured outputs from end nodes:
```json
{
  "outputs": {
    "final_answer": "Generated response",
    "confidence_score": 0.95
  }
}
```
                    """
                }
            ]
            
            print("📄 Adding sample documents...")
            for doc in sample_docs:
                client.add_document_to_dataset(
                    dataset_id=dataset_id,
                    content=doc["content"],
                    name=doc["name"]
                )
            
            # Wait for indexing
            print("⏳ Waiting for document indexing...")
            time.sleep(10)
            
            # Test dataset querying
            print("\n🔍 Testing dataset queries...")
            test_queries = [
                "How do I authenticate with the Dify API?",
                "What are the rate limits?",
                "How do workflow executions work?"
            ]
            
            for query in test_queries:
                print(f"\n❓ Query: {query}")
                result = client.query_dataset(dataset_id, query)
                
                records = result.get('query', {}).get('records', [])
                if records:
                    print(f"   📚 Found {len(records)} relevant documents")
                    for i, record in enumerate(records[:2], 1):
                        score = record.get('score', 0)
                        content = record.get('content', '')[:100]
                        print(f"   {i}. Score: {score:.3f} - {content}...")
                else:
                    print("   ❌ No relevant documents found")
        
        # 3. Application Interaction Demo
        print("\n💬 Part 3: Application Interaction")
        print("-" * 40)
        
        if chat_app_id:
            print("🤖 Testing chat application...")
            test_messages = [
                "Hello! What can you help me with?",
                "How do I use the Dify API for chat applications?",
                "What are some best practices for API integration?"
            ]
            
            conversation_id = None
            for i, message in enumerate(test_messages, 1):
                print(f"\n[{i}] 💭 User: {message}")
                
                response = client.send_chat_message(
                    app_id=chat_app_id,
                    message=message,
                    conversation_id=conversation_id
                )
                
                print(f"🤖 Bot: {response.get('answer', 'No response')}")
                conversation_id = response.get('conversation_id')
                
                # Show usage info
                usage = response.get('usage', {})
                if usage.get('total_tokens'):
                    print(f"   📊 Tokens: {usage['total_tokens']}")
        
        if workflow_app_id:
            print("\n⚙️ Testing workflow application...")
            workflow_result = client.run_workflow(
                app_id=workflow_app_id,
                inputs={"input_text": "Explain the benefits of using Dify APIs"}
            )
            
            if workflow_result["success"]:
                outputs = workflow_result["outputs"]
                print(f"✅ Workflow executed successfully")
                print(f"   📤 Output: {outputs.get('result', 'No output')}")
                print(f"   📊 Tokens: {workflow_result['usage']}")
            else:
                print(f"❌ Workflow failed: {workflow_result.get('error')}")
        
        # 4. Analytics and Management Demo
        print("\n📊 Part 4: Analytics & Management")
        print("-" * 40)
        
        # Get app statistics
        if chat_app_id:
            print("📈 Getting app statistics...")
            stats = client.get_app_statistics(chat_app_id)
            if stats:
                print(f"   📊 Usage data available for analysis")
            
            # Get conversations
            print("💬 Getting conversation history...")
            conversations = client.get_conversations(chat_app_id)
            print(f"   📋 Found {len(conversations)} conversations")
        
        # Show summary
        client.print_summary()
        
        print("\n✅ API Integration Demo Completed Successfully!")
        print("\nKey Capabilities Demonstrated:")
        print("• ✅ Application creation and management")
        print("• ✅ Dataset creation and document management")
        print("• ✅ Real-time chat interactions")
        print("• ✅ Workflow execution")
        print("• ✅ Analytics and monitoring")
        print("• ✅ Resource management and cleanup")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    
    finally:
        # Cleanup resources
        try:
            client.cleanup_resources()
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


def main():
    """Main function to run the API integration example."""
    
    print("🤖 Dify API Integration Example")
    print("=" * 40)
    
    # Check for environment setup
    if not os.getenv('DIFY_API_KEY'):
        print("💡 Tip: Set DIFY_API_KEY environment variable")
        print("   export DIFY_API_KEY='your_api_key_here'")
    
    print("\nThis example demonstrates:")
    print("• Complete Dify API integration")
    print("• Application lifecycle management")
    print("• Dataset and document operations")
    print("• Real-time chat and workflow execution")
    print("• Analytics and monitoring capabilities")
    
    try:
        comprehensive_api_demo()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()