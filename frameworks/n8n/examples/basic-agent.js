#!/usr/bin/env node
/**
 * Basic AI Agent Example - n8n
 * =============================
 * 
 * This example demonstrates how to create a basic AI agent workflow using n8n's API.
 * The agent can perform web searches and answer questions using OpenAI.
 * 
 * Prerequisites:
 * - Node.js 18+ installed
 * - n8n server running: npx n8n
 * - OpenAI API key set in environment: OPENAI_API_KEY
 * 
 * Usage:
 *   node basic-agent.js
 */

const fetch = require('node-fetch');
const readline = require('readline');

class N8nAgent {
    constructor(baseUrl = 'http://localhost:5678') {
        this.baseUrl = baseUrl;
        this.workflowId = null;
        
        // Verify API key is set
        if (!process.env.OPENAI_API_KEY) {
            throw new Error('Please set OPENAI_API_KEY environment variable');
        }
    }
    
    /**
     * Create an AI agent workflow configuration
     */
    createWorkflowConfig() {
        return {
            name: "AI Research Agent",
            active: true,
            nodes: [
                {
                    id: "webhook-trigger",
                    name: "Webhook Trigger",
                    type: "n8n-nodes-base.webhook",
                    position: [240, 300],
                    parameters: {
                        httpMethod: "POST",
                        path: "agent",
                        options: {}
                    }
                },
                {
                    id: "analyze-query",
                    name: "Analyze Query",
                    type: "n8n-nodes-base.openAi",
                    position: [460, 300],
                    parameters: {
                        resource: "chat",
                        operation: "message",
                        model: "gpt-3.5-turbo",
                        messages: {
                            values: [
                                {
                                    role: "system",
                                    content: `You are an AI assistant that determines if a query requires web search.
                                    
                                    Respond with JSON only:
                                    {"needsSearch": true/false, "searchQuery": "optimized search terms", "reasoning": "why search is/isn't needed"}
                                    
                                    User Query: {{$json.query}}`
                                }
                            ]
                        },
                        options: {
                            temperature: 0.1,
                            maxTokens: 200
                        }
                    },
                    credentials: {
                        openAiApi: {
                            id: "openai-credentials",
                            name: "OpenAI API"
                        }
                    }
                },
                {
                    id: "conditional-search",
                    name: "Conditional Web Search",
                    type: "n8n-nodes-base.if",
                    position: [680, 300],
                    parameters: {
                        conditions: {
                            string: [
                                {
                                    value1: "={{JSON.parse($json.choices[0].message.content).needsSearch}}",
                                    operation: "equal",
                                    value2: "true"
                                }
                            ]
                        }
                    }
                },
                {
                    id: "web-search",
                    name: "Web Search",
                    type: "n8n-nodes-base.httpRequest",
                    position: [900, 200],
                    parameters: {
                        method: "GET",
                        url: "https://api.duckduckgo.com/",
                        qs: {
                            q: "={{JSON.parse($('Analyze Query').item.json.choices[0].message.content).searchQuery}}",
                            format: "json",
                            no_html: "1",
                            skip_disambig: "1"
                        },
                        options: {
                            timeout: 10000
                        }
                    }
                },
                {
                    id: "generate-response",
                    name: "Generate Response",
                    type: "n8n-nodes-base.openAi",
                    position: [1120, 300],
                    parameters: {
                        resource: "chat",
                        operation: "message",
                        model: "gpt-4",
                        messages: {
                            values: [
                                {
                                    role: "system",
                                    content: `You are a helpful AI assistant. Use the provided context to answer the user's question.
                                    
                                    Context: {{$json.searchResults || 'No additional context available'}}
                                    
                                    User Question: {{$('Webhook Trigger').item.json.body.query}}
                                    
                                    Provide a comprehensive, accurate answer based on the context and your knowledge.`
                                }
                            ]
                        },
                        options: {
                            temperature: 0.7,
                            maxTokens: 1000
                        }
                    },
                    credentials: {
                        openAiApi: {
                            id: "openai-credentials",
                            name: "OpenAI API"
                        }
                    }
                },
                {
                    id: "direct-response",
                    name: "Direct Response",
                    type: "n8n-nodes-base.openAi",
                    position: [900, 400],
                    parameters: {
                        resource: "chat",
                        operation: "message",
                        model: "gpt-4",
                        messages: {
                            values: [
                                {
                                    role: "system",
                                    content: `You are a helpful AI assistant. Answer the user's question directly.
                                    
                                    User Question: {{$('Webhook Trigger').item.json.body.query}}
                                    
                                    Provide a comprehensive, accurate answer.`
                                }
                            ]
                        },
                        options: {
                            temperature: 0.7,
                            maxTokens: 1000
                        }
                    },
                    credentials: {
                        openAiApi: {
                            id: "openai-credentials", 
                            name: "OpenAI API"
                        }
                    }
                },
                {
                    id: "merge-responses",
                    name: "Merge Responses",
                    type: "n8n-nodes-base.merge",
                    position: [1340, 300],
                    parameters: {
                        mode: "append",
                        options: {}
                    }
                },
                {
                    id: "format-output",
                    name: "Format Output",
                    type: "n8n-nodes-base.set",
                    position: [1560, 300],
                    parameters: {
                        values: {
                            string: [
                                {
                                    name: "response",
                                    value: "={{$json.choices[0].message.content}}"
                                },
                                {
                                    name: "timestamp",
                                    value: "={{new Date().toISOString()}}"
                                },
                                {
                                    name: "query",
                                    value: "={{$('Webhook Trigger').item.json.body.query}}"
                                }
                            ]
                        },
                        options: {}
                    }
                }
            ],
            connections: {
                "Webhook Trigger": {
                    "main": [
                        [
                            {
                                "node": "Analyze Query",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "Analyze Query": {
                    "main": [
                        [
                            {
                                "node": "Conditional Web Search",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "Conditional Web Search": {
                    "main": [
                        [
                            {
                                "node": "Web Search",
                                "type": "main",
                                "index": 0
                            }
                        ],
                        [
                            {
                                "node": "Direct Response",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "Web Search": {
                    "main": [
                        [
                            {
                                "node": "Generate Response",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "Generate Response": {
                    "main": [
                        [
                            {
                                "node": "Merge Responses",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "Direct Response": {
                    "main": [
                        [
                            {
                                "node": "Merge Responses",
                                "type": "main",
                                "index": 1
                            }
                        ]
                    ]
                },
                "Merge Responses": {
                    "main": [
                        [
                            {
                                "node": "Format Output",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                }
            },
            settings: {
                saveExecutionProgress: true,
                saveManualExecutions: true,
                callerPolicy: "workflowsFromSameOwner"
            }
        };
    }
    
    /**
     * Create OpenAI credentials in n8n
     */
    async setupCredentials() {
        try {
            const credentialData = {
                name: "OpenAI API",
                type: "openAiApi",
                data: {
                    apiKey: process.env.OPENAI_API_KEY
                }
            };
            
            const response = await fetch(`${this.baseUrl}/rest/credentials`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(credentialData)
            });
            
            if (!response.ok) {
                // Credentials might already exist, continue
                console.log('⚠️  Credentials might already exist, continuing...');
            } else {
                console.log('✅ OpenAI credentials created');
            }
            
        } catch (error) {
            console.log('⚠️  Error setting up credentials (might already exist):', error.message);
        }
    }
    
    /**
     * Create the agent workflow in n8n
     */
    async createWorkflow() {
        try {
            // Setup credentials first
            await this.setupCredentials();
            
            const config = this.createWorkflowConfig();
            
            const response = await fetch(`${this.baseUrl}/rest/workflows`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                throw new Error(`Failed to create workflow: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.workflowId = result.id;
            
            console.log(`✅ Agent workflow created with ID: ${this.workflowId}`);
            return this.workflowId;
            
        } catch (error) {
            console.error('❌ Error creating workflow:', error.message);
            throw error;
        }
    }
    
    /**
     * Send a query to the agent and get response
     */
    async queryAgent(query) {
        try {
            if (!this.workflowId) {
                throw new Error('No workflow created. Call createWorkflow() first.');
            }
            
            const response = await fetch(`${this.baseUrl}/webhook/agent`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query
                })
            });
            
            if (!response.ok) {
                throw new Error(`Webhook request failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            return result.response || 'No response received';
            
        } catch (error) {
            console.error('❌ Error querying agent:', error.message);
            return `Error: ${error.message}`;
        }
    }
    
    /**
     * Check if n8n server is running
     */
    async checkServerStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/rest/active-workflows`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }
    
    /**
     * Clean up - delete the workflow
     */
    async cleanup() {
        if (this.workflowId) {
            try {
                await fetch(`${this.baseUrl}/rest/workflows/${this.workflowId}`, {
                    method: 'DELETE'
                });
                console.log('🧹 Workflow cleaned up');
            } catch (error) {
                console.error('Warning: Failed to cleanup workflow:', error.message);
            }
        }
    }
}

/**
 * Main function
 */
async function main() {
    console.log('🤖 n8n AI Research Agent');
    console.log('=' .repeat(50));
    
    const agent = new N8nAgent();
    
    // Check if server is running
    console.log('🔍 Checking n8n server...');
    const serverRunning = await agent.checkServerStatus();
    
    if (!serverRunning) {
        console.error('❌ n8n server is not running!');
        console.log('💡 Start it with: npx n8n');
        process.exit(1);
    }
    
    console.log('✅ n8n server is running');
    
    try {
        // Create workflow
        console.log('🏗️  Creating agent workflow...');
        await agent.createWorkflow();
        
        // Wait a moment for workflow to be ready
        console.log('⏳ Waiting for workflow to be ready...');
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // Setup readline interface
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
            prompt: '\n🔍 Ask me anything: '
        });
        
        console.log('\n🎉 AI Agent is ready! Type "quit" to exit.\n');
        console.log('💡 The agent can search the web for current information or answer from knowledge.\n');
        rl.prompt();
        
        rl.on('line', async (input) => {
            const query = input.trim();
            
            if (query.toLowerCase() === 'quit') {
                console.log('\n👋 Goodbye!');
                await agent.cleanup();
                rl.close();
                return;
            }
            
            if (query) {
                console.log('🤖 Agent is researching...');
                const response = await agent.queryAgent(query);
                console.log(`\n🤖 Agent: ${response}\n`);
            }
            
            rl.prompt();
        });
        
        rl.on('SIGINT', async () => {
            console.log('\n\n👋 Goodbye!');
            await agent.cleanup();
            process.exit(0);
        });
        
    } catch (error) {
        console.error('❌ Failed to start agent:', error.message);
        process.exit(1);
    }
}

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Start the application
if (require.main === module) {
    main().catch(console.error);
}

module.exports = N8nAgent;