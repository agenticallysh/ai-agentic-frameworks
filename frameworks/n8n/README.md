# n8n Framework Guide

[![GitHub Stars](https://img.shields.io/github/stars/n8n-io/n8n)](https://github.com/n8n-io/n8n)
[![License](https://img.shields.io/badge/license-Fair--Code-blue.svg)](https://docs.n8n.io/license/)
[![Node.js](https://img.shields.io/badge/node.js-16+-green.svg)](https://nodejs.org/)

[🔍 Compare with other frameworks →](https://www.agentically.sh/ai-agentic-frameworks/compare/n8n/)

n8n is a fair-code workflow automation platform that combines visual building with custom code capabilities. It offers 400+ integrations and native AI capabilities for building sophisticated agent workflows and business process automation.

## Key Features

- 🎨 **Visual Workflow Builder**: Drag-and-drop interface for creating complex workflows
- 🔌 **400+ Integrations**: Pre-built connectors for popular services and APIs
- 🤖 **Native AI Capabilities**: Built-in AI nodes for LLM integration
- 💻 **Custom Code Support**: Write JavaScript/TypeScript when needed
- 🏢 **Enterprise Ready**: Self-hosted or cloud deployment options
- 📊 **Real-time Monitoring**: Workflow execution tracking and debugging

## When to Use n8n

✅ **Best for:**
- Business process automation requiring visual design
- Teams with mixed technical skill levels
- Applications needing extensive third-party integrations
- AI-powered workflows with traditional business logic
- Organizations wanting self-hosted workflow automation
- Prototyping and rapid development of automation workflows

❌ **Not ideal for:**
- Pure code-first development approaches
- Simple single-step automations
- Applications requiring complex custom algorithms
- High-frequency, low-latency operations

## Quick Start

### Installation Options

#### Option 1: Quick Start (Cloud)
```bash
# Try n8n instantly at https://n8n.cloud
```

#### Option 2: NPX (Local)
```bash
npx n8n
# Access at http://localhost:5678
```

#### Option 3: Docker
```bash
docker run -it --rm --name n8n -p 5678:5678 n8nio/n8n
```

#### Option 4: NPM Global Install
```bash
npm install n8n -g
n8n start
```

### Basic Workflow Example

1. **Create a Simple AI Chat Workflow**:
   - Start node → OpenAI node → Response node
   - Configure OpenAI with your API key
   - Set up input/output handling

2. **Test the Workflow**:
   - Use the manual trigger
   - Send test data through the workflow
   - View execution results

## Examples

- [Basic AI Chatbot](./examples/basic-chatbot.json) - Simple conversational workflow
- [Email Automation](./examples/email-automation.json) - AI-powered email responses  
- [Data Processing](./examples/data-processing.json) - Transform and analyze data
- [Multi-Step Agent](./examples/multi-step-agent.json) - Complex agent workflow
- [API Integration](./examples/api-integration.json) - Connect multiple services

## Benchmarks

[View detailed benchmarks →](./benchmarks.md)

| Metric | n8n | Industry Average |
|--------|-----|------------------|
| Setup Time | 5 minutes | 30 minutes |
| Integration Count | 400+ | 100 |
| Visual Development | 9/10 | 6/10 |
| Learning Curve | 8/10 | 6/10 |
| Enterprise Features | 8/10 | 7/10 |

## Migration Guides

- [From Zapier to n8n](../../migration-guides/zapier-to-n8n.md)
- [From Microsoft Power Automate to n8n](../../migration-guides/power-automate-to-n8n.md)
- [From Custom Scripts to n8n](../../migration-guides/scripts-to-n8n.md)

## Core Concepts

### Workflows
n8n workflows consist of connected nodes that process data:

```json
{
  "nodes": [
    {
      "name": "Start",
      "type": "n8n-nodes-base.start",
      "position": [240, 300]
    },
    {
      "name": "OpenAI Chat Model",
      "type": "n8n-nodes-base.openAi",
      "position": [460, 300],
      "parameters": {
        "operation": "chat",
        "model": "gpt-4",
        "messages": {
          "messageValues": [
            {
              "role": "user",
              "content": "={{$json.query}}"
            }
          ]
        }
      }
    }
  ],
  "connections": {
    "Start": {
      "main": [
        [
          {
            "node": "OpenAI Chat Model",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### Node Types
Different node categories for various use cases:

#### Trigger Nodes
- **Webhook**: HTTP endpoint triggers
- **Schedule**: Time-based triggers  
- **Email**: Email-based triggers
- **File Watch**: File system monitoring

#### AI Nodes
- **OpenAI**: GPT models and DALL-E
- **Anthropic**: Claude models
- **Hugging Face**: Open-source models
- **AI Agent**: Multi-step AI reasoning

#### Action Nodes
- **HTTP Request**: API calls
- **Database**: SQL operations
- **Email**: Send emails
- **Slack**: Team notifications

### Data Flow
Data flows between nodes as JSON objects:

```javascript
// Example data transformation
{
  "query": "What's the weather like?",
  "timestamp": "2025-10-22T10:00:00Z",
  "user_id": "user_123"
}

// After OpenAI processing
{
  "query": "What's the weather like?",
  "response": "I'll help you check the weather...",
  "model": "gpt-4",
  "tokens_used": 150
}
```

## Advanced Features

### Custom Code Nodes
Write JavaScript/TypeScript for complex logic:

```javascript
// Code node example
const items = $input.all();
const processedItems = [];

for (const item of items) {
  const data = item.json;
  
  // Custom processing logic
  const processed = {
    ...data,
    processed_at: new Date().toISOString(),
    sentiment: analyzeSentiment(data.text),
    category: classifyText(data.text)
  };
  
  processedItems.push({ json: processed });
}

return processedItems;

// Helper functions
function analyzeSentiment(text) {
  // Custom sentiment analysis logic
  if (text.includes('great') || text.includes('excellent')) {
    return 'positive';
  }
  if (text.includes('bad') || text.includes('terrible')) {
    return 'negative';
  }
  return 'neutral';
}

function classifyText(text) {
  // Custom classification logic
  if (text.includes('order') || text.includes('purchase')) {
    return 'sales';
  }
  if (text.includes('support') || text.includes('help')) {
    return 'support';
  }
  return 'general';
}
```

### AI Agent Workflows
Build multi-step AI agents:

```json
{
  "name": "Customer Support Agent",
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "customer-support",
        "httpMethod": "POST"
      }
    },
    {
      "name": "Intent Classification",
      "type": "n8n-nodes-base.openAi",
      "parameters": {
        "operation": "chat",
        "model": "gpt-4",
        "messages": {
          "messageValues": [
            {
              "role": "system",
              "content": "Classify customer intent: billing, technical, general"
            },
            {
              "role": "user", 
              "content": "={{$json.message}}"
            }
          ]
        }
      }
    },
    {
      "name": "Route by Intent",
      "type": "n8n-nodes-base.switch",
      "parameters": {
        "conditions": [
          {
            "field": "{{$json.choices[0].message.content}}",
            "operation": "contains",
            "value": "billing"
          },
          {
            "field": "{{$json.choices[0].message.content}}",
            "operation": "contains", 
            "value": "technical"
          }
        ]
      }
    },
    {
      "name": "Billing Handler",
      "type": "n8n-nodes-base.openAi",
      "parameters": {
        "operation": "chat",
        "model": "gpt-4",
        "messages": {
          "messageValues": [
            {
              "role": "system",
              "content": "You are a billing specialist. Help with billing inquiries."
            },
            {
              "role": "user",
              "content": "={{$('Webhook Trigger').item.json.message}}"
            }
          ]
        }
      }
    }
  ]
}
```

### Environment Configuration
```bash
# .env file for n8n
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=your-password

# Database
DB_TYPE=postgresdb
DB_POSTGRESDB_HOST=localhost
DB_POSTGRESDB_PORT=5432
DB_POSTGRESDB_DATABASE=n8n
DB_POSTGRESDB_USER=n8n
DB_POSTGRESDB_PASSWORD=n8n

# Encryption
N8N_ENCRYPTION_KEY=your-encryption-key

# AI Service Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

## Production Deployment

### Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  n8n:
    image: n8nio/n8n:latest
    restart: always
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_HOST=${DOMAIN_NAME}
      - N8N_PORT=5678
      - N8N_PROTOCOL=https
      - NODE_ENV=production
      - WEBHOOK_URL=https://${DOMAIN_NAME}/
      - GENERIC_TIMEZONE=${TIMEZONE}
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=${POSTGRES_DB}
      - DB_POSTGRESDB_USER=${POSTGRES_USER}
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD}
    links:
      - postgres
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    image: postgres:15
    restart: always
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  n8n_data:
  postgres_data:
```

### Kubernetes Deployment
```yaml
# n8n-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: n8n
spec:
  replicas: 2
  selector:
    matchLabels:
      app: n8n
  template:
    metadata:
      labels:
        app: n8n
    spec:
      containers:
      - name: n8n
        image: n8nio/n8n:latest
        ports:
        - containerPort: 5678
        env:
        - name: N8N_BASIC_AUTH_ACTIVE
          value: "true"
        - name: N8N_BASIC_AUTH_USER
          valueFrom:
            secretKeyRef:
              name: n8n-secrets
              key: username
        - name: N8N_BASIC_AUTH_PASSWORD
          valueFrom:
            secretKeyRef:
              name: n8n-secrets
              key: password
        - name: DB_TYPE
          value: "postgresdb"
        - name: DB_POSTGRESDB_HOST
          value: "postgres-service"
        volumeMounts:
        - name: n8n-data
          mountPath: /home/node/.n8n
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: n8n-data
        persistentVolumeClaim:
          claimName: n8n-pvc
```

## Integration Examples

### Slack Bot Integration
```json
{
  "name": "Slack AI Bot",
  "nodes": [
    {
      "name": "Slack Event",
      "type": "n8n-nodes-base.slackTrigger",
      "parameters": {
        "events": ["message"]
      }
    },
    {
      "name": "Filter Bot Messages",
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": [
          {
            "field": "{{$json.event.bot_id}}",
            "operation": "isEmpty"
          }
        ]
      }
    },
    {
      "name": "Generate Response",
      "type": "n8n-nodes-base.openAi",
      "parameters": {
        "operation": "chat",
        "model": "gpt-4",
        "messages": {
          "messageValues": [
            {
              "role": "system",
              "content": "You are a helpful Slack assistant."
            },
            {
              "role": "user",
              "content": "={{$json.event.text}}"
            }
          ]
        }
      }
    },
    {
      "name": "Send to Slack",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "operation": "postMessage",
        "channel": "={{$('Slack Event').item.json.event.channel}}",
        "text": "={{$json.choices[0].message.content}}"
      }
    }
  ]
}
```

### Email Processing Workflow
```json
{
  "name": "Email AI Assistant", 
  "nodes": [
    {
      "name": "Email Trigger",
      "type": "n8n-nodes-base.emailReadImap",
      "parameters": {
        "host": "imap.gmail.com",
        "port": 993,
        "secure": true,
        "format": "simple"
      }
    },
    {
      "name": "Extract Content",
      "type": "n8n-nodes-base.code",
      "parameters": {
        "jsCode": "const email = $input.first().json;\nreturn [{\n  json: {\n    from: email.from,\n    subject: email.subject,\n    body: email.textPlain || email.html,\n    timestamp: new Date().toISOString()\n  }\n}];"
      }
    },
    {
      "name": "Classify Urgency",
      "type": "n8n-nodes-base.openAi",
      "parameters": {
        "operation": "chat",
        "model": "gpt-4",
        "messages": {
          "messageValues": [
            {
              "role": "system",
              "content": "Classify email urgency: high, medium, low"
            },
            {
              "role": "user",
              "content": "Subject: {{$json.subject}}\nBody: {{$json.body}}"
            }
          ]
        }
      }
    },
    {
      "name": "Generate Reply",
      "type": "n8n-nodes-base.openAi",
      "parameters": {
        "operation": "chat",
        "model": "gpt-4",
        "messages": {
          "messageValues": [
            {
              "role": "system",
              "content": "Generate a professional email reply."
            },
            {
              "role": "user",
              "content": "Original: {{$('Extract Content').item.json.body}}"
            }
          ]
        }
      }
    }
  ]
}
```

## Use Cases

### Business Process Automation
- Lead qualification and routing
- Invoice processing and approval
- Customer onboarding workflows
- Employee offboarding checklists

### Data Pipeline Orchestration
- ETL processes with validation
- Data synchronization between systems
- Report generation and distribution
- Real-time data monitoring

### Customer Experience Automation
- Support ticket routing and escalation
- Personalized marketing campaigns
- Customer feedback processing
- Order fulfillment automation

## Performance Optimization

### Workflow Design Best Practices
```javascript
// Efficient data processing
const items = $input.all();
const batchSize = 10;
const results = [];

// Process in batches to avoid memory issues
for (let i = 0; i < items.length; i += batchSize) {
  const batch = items.slice(i, i + batchSize);
  const batchResults = await processBatch(batch);
  results.push(...batchResults);
}

return results.map(item => ({ json: item }));
```

### Caching Strategies
```javascript
// Cache API responses
const cacheKey = `weather_${$json.location}`;
let cachedResult = await $cache.get(cacheKey);

if (!cachedResult) {
  const apiResponse = await $http.request({
    method: 'GET',
    url: `https://api.weather.com/v1/current?location=${$json.location}`
  });
  
  cachedResult = apiResponse.data;
  await $cache.set(cacheKey, cachedResult, 3600); // Cache for 1 hour
}

return [{ json: cachedResult }];
```

### Resource Management
```yaml
# Resource limits for production
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

# Environment variables for optimization
environment:
  - name: N8N_WORKERS_COUNT
    value: "4"
  - name: N8N_MAX_EXECUTION_TIMEOUT
    value: "3600"
  - name: N8N_EXECUTION_DATA_MAX_AGE
    value: "168" # 7 days
```

## Community & Support

- [GitHub Repository](https://github.com/n8n-io/n8n) - 49.5k+ stars
- [Documentation](https://docs.n8n.io/) - Comprehensive guides and API reference
- [Community Forum](https://community.n8n.io/) - Active community discussions
- [Discord Server](https://discord.gg/n8n) - Real-time community support

## Enterprise Features

### n8n Cloud Platform
- Managed hosting with automatic updates
- Enhanced security and compliance
- Priority support and SLA
- Advanced monitoring and analytics

### Self-Hosted Enterprise
- SSO integration (SAML, OAuth)
- Advanced user management
- Audit logging and compliance
- Custom integrations and support

### Fair-Code License
- Source code available for inspection
- Self-hosting allowed for internal use
- Commercial restrictions for competing services
- Enterprise license for commercial offerings

## Best Practices

### 1. Workflow Organization
Structure workflows for maintainability:
```json
{
  "name": "Customer Onboarding - Main",
  "description": "Primary workflow for new customer onboarding",
  "tags": ["customer", "onboarding", "production"],
  "settings": {
    "executionOrder": "v1",
    "saveManualExecutions": true,
    "callerPolicy": "workflowsFromSameOwner"
  }
}
```

### 2. Error Handling
Implement comprehensive error handling:
```javascript
// Error handling in Code node
try {
  const result = await processData($json.input);
  return [{ json: { success: true, result } }];
} catch (error) {
  // Log error for debugging
  console.error('Processing failed:', error);
  
  // Return error for workflow handling
  return [{
    json: {
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    }
  }];
}
```

### 3. Security Considerations
```javascript
// Sanitize user inputs
function sanitizeInput(input) {
  if (typeof input !== 'string') return '';
  
  return input
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/[<>]/g, '')
    .trim();
}

const sanitizedInput = sanitizeInput($json.userInput);
```

### 4. Monitoring and Alerting
```json
{
  "name": "Error Notification",
  "type": "n8n-nodes-base.slack",
  "parameters": {
    "operation": "postMessage",
    "channel": "#alerts",
    "text": "🚨 Workflow Error: {{$json.error}} in {{$workflow.name}}"
  }
}
```

## Further Reading

- [n8n vs Zapier Comparison](https://www.agentically.sh/ai-agentic-frameworks/compare/n8n-vs-zapier/)
- [Enterprise Deployment Guide](https://www.agentically.sh/ai-agentic-frameworks/n8n/enterprise/)
- [AI Workflow Patterns](https://www.agentically.sh/ai-agentic-frameworks/n8n/ai-patterns/)
- [Performance Optimization](https://www.agentically.sh/ai-agentic-frameworks/n8n/optimization/)

---

[← Back to Framework Comparison](../../) | [Compare n8n →](https://www.agentically.sh/ai-agentic-frameworks/compare/n8n/)