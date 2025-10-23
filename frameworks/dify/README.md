# Dify Framework Guide

[![GitHub Stars](https://img.shields.io/github/stars/langgenius/dify)](https://github.com/langgenius/dify)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker](https://img.shields.io/badge/deploy-Docker-blue.svg)](https://docs.dify.ai/getting-started/install-self-hosted/docker-compose)

[🔍 Compare with other frameworks →](https://www.agentically.sh/ai-agentic-frameworks/compare/dify/)

Dify is an open-source platform for developing and operating generative AI applications. It provides visual orchestration, comprehensive LLMOps capabilities, and production-ready deployment options for building sophisticated AI agents and workflows.

## Key Features

- 🎨 **Visual Agent Builder**: Drag-and-drop interface for creating AI workflows
- 📊 **LLMOps Platform**: Complete lifecycle management for LLM applications  
- 🧠 **RAG & Knowledge Base**: Built-in document processing and retrieval systems
- 🔄 **Multi-Agent Workflows**: Orchestrate complex agent interactions
- 📈 **Observability**: Real-time monitoring, logging, and analytics
- ☁️ **Cloud & Self-Hosted**: Deploy on Dify Cloud or your own infrastructure

## When to Use Dify

✅ **Best for:**
- Teams wanting visual workflow design with code flexibility
- Production LLM applications requiring observability
- RAG applications with document management
- Multi-modal AI applications (text, image, audio)
- Organizations needing LLMOps and governance
- API-first AI service development

❌ **Not ideal for:**
- Simple single-LLM integrations
- Lightweight prototyping and experimentation
- Applications requiring extensive custom code
- Resource-constrained environments

## Quick Start

### Option 1: Dify Cloud (Fastest)

1. Visit [Dify Cloud](https://cloud.dify.ai)
2. Sign up for free sandbox account
3. Create your first AI application
4. Use visual builder to design workflows

### Option 2: Docker Deployment

```bash
# Clone repository
git clone https://github.com/langgenius/dify.git
cd dify/docker

# Start with docker-compose
cp .env.example .env
docker-compose up -d

# Access at http://localhost/install
```

### Option 3: Source Installation

```bash
# Backend setup
git clone https://github.com/langgenius/dify.git
cd dify/api

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize database
flask db upgrade

# Start backend
python app.py

# Frontend setup (new terminal)
cd ../web
npm install
npm run dev

# Access at http://localhost:3000
```

## Examples

- [Basic Chatbot](./examples/basic-chatbot.py) - Create simple conversational AI
- [RAG Application](./examples/rag-application.py) - Document Q&A system
- [Multi-Agent Workflow](./examples/multi-agent-workflow.py) - Coordinated agent tasks
- [API Integration](./examples/api-integration.py) - Using Dify APIs
- [Custom Tool](./examples/custom-tool.py) - Adding custom functionality

## Benchmarks

[View detailed benchmarks →](./benchmarks.md)

| Metric | Dify | Industry Average |
|--------|------|------------------|
| Setup Time | 15 minutes | 45 minutes |
| Visual Development | 9/10 | 6/10 |
| Production Readiness | 9/10 | 6/10 |
| Observability | 9/10 | 5/10 |
| Multi-Modal Support | 8/10 | 4/10 |

## Migration Guides

- [From LangChain to Dify](../../migration-guides/langchain-to-dify.md)
- [From Flowise to Dify](../../migration-guides/flowise-to-dify.md)
- [From Custom Solution to Dify](../../migration-guides/custom-to-dify.md)

## Core Concepts

### Applications
Dify organizes AI functionality into applications:

- **Chatbot**: Conversational AI applications
- **Text Generator**: Content generation apps
- **Agent**: Tool-using AI agents
- **Workflow**: Complex multi-step processes

### Workflow Builder
Visual interface for creating AI workflows:

```yaml
# Example workflow configuration
workflow:
  name: "Customer Support Agent"
  nodes:
    - type: "llm"
      name: "intent_classifier"
      model: "gpt-4"
      prompt: "Classify the customer intent: {{user_input}}"
    
    - type: "condition"
      name: "route_intent"
      conditions:
        - if: "intent == 'billing'"
          then: "billing_agent"
        - if: "intent == 'technical'"
          then: "tech_agent"
    
    - type: "agent"
      name: "billing_agent"
      tools: ["billing_lookup", "payment_processor"]
```

### Knowledge Base
Built-in document management and RAG:

```python
# Upload documents via API
import requests

files = {'file': open('document.pdf', 'rb')}
data = {
    'knowledge_base_id': 'kb_12345',
    'indexing_technique': 'high_quality'
}

response = requests.post(
    'http://localhost/v1/datasets/documents',
    files=files,
    data=data,
    headers={'Authorization': 'Bearer your-api-key'}
)
```

## Advanced Features

### Multi-Agent Orchestration
```yaml
# Multi-agent workflow example
agents:
  - name: "researcher"
    role: "Research and gather information"
    tools: ["web_search", "document_retrieval"]
    
  - name: "analyst"
    role: "Analyze gathered data"
    tools: ["data_analysis", "chart_generator"]
    
  - name: "writer"
    role: "Create final report"
    tools: ["document_writer", "formatter"]

workflow:
  - researcher → analyst → writer
  - parallel: [researcher, analyst] → writer
```

### Custom Tools Integration
```python
# Custom tool for Dify
from dify import Tool

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="weather_lookup",
            description="Get current weather for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                }
            }
        )
    
    def run(self, location: str) -> dict:
        # Your weather API integration
        weather_data = get_weather(location)
        return {
            "temperature": weather_data.temp,
            "conditions": weather_data.conditions,
            "humidity": weather_data.humidity
        }

# Register tool
dify.register_tool(WeatherTool())
```

### API Development
```python
# Using Dify APIs programmatically
import requests

class DifyClient:
    def __init__(self, api_key, base_url="https://api.dify.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, app_id, query, user_id=None):
        """Send chat completion request."""
        url = f"{self.base_url}/chat-messages"
        data = {
            "inputs": {},
            "query": query,
            "response_mode": "blocking",
            "conversation_id": "",
            "user": user_id or "default_user"
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()
    
    def run_workflow(self, app_id, inputs):
        """Execute workflow with inputs."""
        url = f"{self.base_url}/workflows/run"
        data = {
            "inputs": inputs,
            "response_mode": "blocking",
            "user": "workflow_user"
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()

# Usage
client = DifyClient("your-api-key")
result = client.chat_completion("app-123", "Hello, how can you help me?")
```

## Production Deployment

### Docker Compose (Recommended)
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  api:
    image: langgenius/dify-api:latest
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - CELERY_BROKER_URL=${CELERY_BROKER_URL}
    depends_on:
      - postgres
      - redis
    
  web:
    image: langgenius/dify-web:latest
    environment:
      - API_URL=http://api:5001
    ports:
      - "80:3000"
    
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: dify
      POSTGRES_USER: dify
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment
```yaml
# dify-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dify-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dify-api
  template:
    metadata:
      labels:
        app: dify-api
    spec:
      containers:
      - name: api
        image: langgenius/dify-api:latest
        ports:
        - containerPort: 5001
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: dify-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: dify-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Environment Configuration
```bash
# .env.production
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:pass@host:5432/dify
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1

# LLM Provider Configuration
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
AZURE_OPENAI_API_KEY=your-azure-key

# Security
CORS_ALLOW_ORIGINS=https://yourdomain.com
SESSION_COOKIE_SECURE=true
CSRF_COOKIE_SECURE=true

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
```

## Monitoring & Observability

### Built-in Analytics
```python
# Access application metrics
def get_app_metrics(app_id, start_date, end_date):
    url = f"{base_url}/apps/{app_id}/statistics"
    params = {
        "start": start_date,
        "end": end_date
    }
    response = requests.get(url, params=params, headers=headers)
    return response.json()

metrics = get_app_metrics("app-123", "2025-10-01", "2025-10-22")
print(f"Total conversations: {metrics['total_conversations']}")
print(f"Average response time: {metrics['avg_response_time']}ms")
```

### Custom Monitoring
```python
# Add custom monitoring hooks
import logging
from dify.core.monitoring import monitor

@monitor.track_execution
def custom_workflow_step(inputs):
    """Custom step with monitoring."""
    start_time = time.time()
    
    try:
        result = process_data(inputs)
        monitor.record_success("custom_step", time.time() - start_time)
        return result
    except Exception as e:
        monitor.record_error("custom_step", str(e))
        raise
```

## Use Cases

### Customer Support Automation
Build intelligent support systems:
- Intent classification and routing
- Knowledge base integration
- Escalation workflows
- Multi-language support

### Content Generation Platform
Create content generation services:
- Blog post generation
- Social media content
- Product descriptions
- Marketing copy

### Document Processing Pipeline
Automate document workflows:
- PDF extraction and analysis
- Content summarization
- Data extraction
- Report generation

## Integration Examples

### Slack Integration
```python
# Slack bot integration
from slack_bolt import App
from dify_client import DifyClient

app = App(token="your-slack-token")
dify = DifyClient("your-dify-api-key")

@app.message("hello")
def message_hello(message, say):
    user_query = message['text']
    
    # Send to Dify
    response = dify.chat_completion(
        app_id="slack-bot-app",
        query=user_query,
        user_id=message['user']
    )
    
    say(response['answer'])

app.start(port=3000)
```

### WhatsApp Integration
```python
# WhatsApp business integration
from twilio.rest import Client
from flask import Flask, request

app = Flask(__name__)
twilio_client = Client("account_sid", "auth_token")
dify = DifyClient("your-dify-api-key")

@app.route("/webhook", methods=['POST'])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '')
    from_number = request.values.get('From', '')
    
    # Process with Dify
    response = dify.chat_completion(
        app_id="whatsapp-bot",
        query=incoming_msg,
        user_id=from_number
    )
    
    # Send response via WhatsApp
    twilio_client.messages.create(
        body=response['answer'],
        from_='whatsapp:+14155238886',
        to=from_number
    )
    
    return "OK"
```

## Performance Optimization

### Caching Strategies
```python
# Implement caching for frequent queries
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_dify_request(query, app_id, cache_ttl=3600):
    """Cache Dify responses for repeated queries."""
    cache_key = f"dify:{app_id}:{hash(query)}"
    
    # Check cache first
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    # Make Dify request
    result = dify.chat_completion(app_id, query)
    
    # Cache result
    redis_client.setex(
        cache_key, 
        cache_ttl, 
        json.dumps(result)
    )
    
    return result
```

### Load Balancing
```nginx
# nginx.conf for load balancing
upstream dify_backend {
    server dify-api-1:5001;
    server dify-api-2:5001;
    server dify-api-3:5001;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location /api/ {
        proxy_pass http://dify_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location / {
        proxy_pass http://dify-web:3000;
    }
}
```

## Community & Support

- [GitHub Repository](https://github.com/langgenius/dify) - 100k+ stars
- [Documentation](https://docs.dify.ai/) - Comprehensive guides
- [Discord Community](https://discord.gg/AhzKf7dNgk) - Active community support
- [Dify Cloud](https://cloud.dify.ai/) - Managed hosting platform

## Enterprise Features

### Dify Cloud Platform
- Multi-tenant isolation
- Enterprise SSO integration
- Advanced analytics and reporting
- Priority support and SLA

### Security & Compliance
- SOC 2 Type II certification
- GDPR compliance
- Data encryption at rest and in transit
- Audit logging and monitoring

### Team Collaboration
- Role-based access control
- Team workspaces
- Version control for applications
- Collaborative editing

## Best Practices

### 1. Application Design
Structure applications for maintainability:
```yaml
# Good application structure
application:
  name: "Customer Service Bot"
  description: "Handles customer inquiries with escalation"
  
  workflows:
    intent_classification:
      description: "Classify user intent"
      timeout: 30s
      
    response_generation:
      description: "Generate contextual response"
      timeout: 60s
      
  knowledge_bases:
    - name: "FAQ"
      description: "Frequently asked questions"
    - name: "Product Docs"
      description: "Product documentation"
```

### 2. Error Handling
Implement robust error handling:
```python
def robust_dify_request(query, max_retries=3):
    """Make Dify request with retry logic."""
    for attempt in range(max_retries):
        try:
            response = dify.chat_completion("app-id", query)
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
```

### 3. Cost Optimization
Monitor and optimize costs:
```python
# Implement usage tracking
def track_usage(app_id, user_id, tokens_used, cost):
    """Track API usage for cost optimization."""
    usage_data = {
        "app_id": app_id,
        "user_id": user_id,
        "tokens_used": tokens_used,
        "cost": cost,
        "timestamp": datetime.utcnow()
    }
    
    # Store in database for analysis
    db.usage_logs.insert(usage_data)
    
    # Alert if usage exceeds threshold
    if cost > DAILY_COST_THRESHOLD:
        send_alert(f"High usage detected: ${cost}")
```

## Further Reading

- [Dify vs Flowise Comparison](https://www.agentically.sh/ai-agentic-frameworks/compare/dify-vs-flowise/)
- [Production Deployment Guide](https://www.agentically.sh/ai-agentic-frameworks/dify/production/)
- [LLMOps Best Practices](https://www.agentically.sh/ai-agentic-frameworks/dify/llmops/)
- [Cost Optimization Strategies](https://www.agentically.sh/ai-agentic-frameworks/cost-calculator/?framework=dify)

---

[← Back to Framework Comparison](../../) | [Compare Dify →](https://www.agentically.sh/ai-agentic-frameworks/compare/dify/)