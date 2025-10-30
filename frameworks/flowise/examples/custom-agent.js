#!/usr/bin/env node
/**
 * Custom Agent Example - Flowise
 * ===============================
 * 
 * This example demonstrates how to create a custom agent with tools and memory
 * in Flowise. The agent can perform web searches, calculations, and maintain
 * conversation history while providing personalized assistance.
 * 
 * Prerequisites:
 * - Node.js 18+ installed
 * - Flowise server running: npx flowise start
 * - OpenAI API key set in environment: OPENAI_API_KEY
 * 
 * Usage:
 *   node custom-agent.js
 */

const readline = require('readline');
const fetch = require('node-fetch');

class FlowiseCustomAgent {
    constructor(baseUrl = 'http://localhost:3000') {
        this.baseUrl = baseUrl;
        this.chatflowId = null;
        this.sessionId = `session_${Date.now()}`;
        
        // Verify API key is set
        if (!process.env.OPENAI_API_KEY) {
            throw new Error('Please set OPENAI_API_KEY environment variable');
        }
    }
    
    /**
     * Create a custom agent flow configuration with tools and memory
     */
    createCustomAgentFlowConfig() {
        return {
            name: "Custom AI Agent with Tools & Memory",
            flowData: JSON.stringify({
                nodes: [
                    // Memory component for conversation history
                    {
                        id: "bufferMemory_0",
                        position: { x: 100, y: 100 },
                        type: "customNode",
                        data: {
                            id: "bufferMemory_0",
                            label: "Buffer Memory",
                            name: "bufferMemory",
                            type: "BufferMemory",
                            baseClasses: ["BufferMemory", "BaseMemory"],
                            inputs: {
                                memoryKey: "chat_history",
                                inputKey: "human_input",
                                outputKey: "output",
                                returnMessages: true,
                                humanPrefix: "Human",
                                aiPrefix: "AI"
                            }
                        }
                    },
                    
                    // Calculator Tool
                    {
                        id: "calculator_0",
                        position: { x: 100, y: 300 },
                        type: "customNode",
                        data: {
                            id: "calculator_0",
                            label: "Calculator",
                            name: "calculator",
                            type: "Calculator",
                            baseClasses: ["Calculator", "Tool"],
                            inputs: {
                                description: "Useful for mathematical calculations and arithmetic operations. Input should be a valid mathematical expression."
                            }
                        }
                    },
                    
                    // Web Browser Tool (simulated)
                    {
                        id: "webBrowser_0",
                        position: { x: 100, y: 500 },
                        type: "customNode",
                        data: {
                            id: "webBrowser_0",
                            label: "Web Search Tool",
                            name: "customTool",
                            type: "CustomTool",
                            baseClasses: ["CustomTool", "Tool"],
                            inputs: {
                                name: "web_search",
                                description: "Search the web for current information. Input should be a search query string.",
                                func: `
async function webSearch(query) {
    // Simulate web search - in production, integrate with real search API
    const mockResults = [
        "Recent information about: " + query,
        "Latest developments and trends related to the search query",
        "Expert insights and analysis on the topic"
    ];
    
    return "Web search results for '" + query + "':\\n" + 
           mockResults.map((result, i) => (i + 1) + ". " + result).join("\\n");
}

return await webSearch(input);
                `
                            }
                        }
                    },
                    
                    // Custom Knowledge Tool
                    {
                        id: "knowledgeTool_0",
                        position: { x: 100, y: 700 },
                        type: "customNode",
                        data: {
                            id: "knowledgeTool_0",
                            label: "Knowledge Base Tool",
                            name: "customTool",
                            type: "CustomTool",
                            baseClasses: ["CustomTool", "Tool"],
                            inputs: {
                                name: "knowledge_lookup",
                                description: "Look up information from the agent's knowledge base about AI, technology, and programming topics.",
                                func: `
async function knowledgeLookup(query) {
    const knowledge = {
        "ai": "Artificial Intelligence is the simulation of human intelligence in machines programmed to think and learn.",
        "machine learning": "A subset of AI that enables computers to learn without being explicitly programmed.",
        "neural networks": "Computing systems inspired by biological neural networks that constitute animal brains.",
        "deep learning": "A subset of machine learning using neural networks with multiple layers.",
        "nlp": "Natural Language Processing helps computers understand, interpret and generate human language.",
        "rag": "Retrieval-Augmented Generation combines information retrieval with text generation for better AI responses.",
        "langchain": "A framework for developing applications powered by language models.",
        "flowise": "An open-source low-code tool for building customized LLM applications.",
        "vector database": "Specialized databases designed to store and query high-dimensional vector embeddings."
    };
    
    const lowerQuery = query.toLowerCase();
    for (const [key, value] of Object.entries(knowledge)) {
        if (lowerQuery.includes(key)) {
            return "Knowledge Base: " + value;
        }
    }
    
    return "No specific information found in knowledge base for: " + query;
}

return await knowledgeLookup(input);
                `
                            }
                        }
                    },
                    
                    // Agent Executor
                    {
                        id: "agentExecutor_0",
                        position: { x: 600, y: 400 },
                        type: "customNode",
                        data: {
                            id: "agentExecutor_0",
                            label: "Custom Agent Executor",
                            name: "agentExecutor",
                            type: "AgentExecutor",
                            baseClasses: ["AgentExecutor", "BaseChain"],
                            inputs: {
                                agentType: "zero-shot-react-description",
                                verbose: true,
                                maxIterations: 5,
                                returnIntermediateSteps: true,
                                handleParsingErrors: true
                            }
                        }
                    },
                    
                    // LLM for the agent
                    {
                        id: "agentLLM_0",
                        position: { x: 400, y: 200 },
                        type: "customNode",
                        data: {
                            id: "agentLLM_0",
                            label: "Agent LLM",
                            name: "chatOpenAI",
                            type: "ChatOpenAI",
                            baseClasses: ["ChatOpenAI", "BaseChatModel"],
                            inputs: {
                                modelName: "gpt-4",
                                temperature: 0.7,
                                maxTokens: 1500,
                                topP: 1,
                                frequencyPenalty: 0,
                                presencePenalty: 0,
                                timeout: 60000,
                                openAIApiKey: process.env.OPENAI_API_KEY
                            }
                        }
                    },
                    
                    // Custom system prompt
                    {
                        id: "systemPrompt_0",
                        position: { x: 400, y: 50 },
                        type: "customNode",
                        data: {
                            id: "systemPrompt_0",
                            label: "System Prompt",
                            name: "promptTemplate",
                            type: "PromptTemplate",
                            baseClasses: ["PromptTemplate", "BasePromptTemplate"],
                            inputs: {
                                template: `You are an intelligent AI assistant with access to several tools and a memory of our conversation.

Your capabilities include:
- Performing mathematical calculations
- Searching the web for current information
- Looking up knowledge from your specialized knowledge base
- Maintaining conversation context and memory

Guidelines:
1. Always be helpful, accurate, and engaging
2. Use tools when appropriate to provide better answers
3. Reference previous conversation when relevant
4. Explain your reasoning when using tools
5. Be proactive in suggesting useful tools for user queries
6. Maintain a friendly but professional tone

Remember: You can use multiple tools in sequence if needed to fully answer a question.

Current conversation context: {chat_history}
Human: {human_input}
Assistant:`,
                                inputVariables: ["chat_history", "human_input"]
                            }
                        }
                    }
                ],
                edges: [
                    // Memory connection to agent
                    {
                        id: "bufferMemory_0-agentExecutor_0",
                        source: "bufferMemory_0",
                        target: "agentExecutor_0",
                        sourceHandle: "bufferMemory_0-output-bufferMemory-BufferMemory|BaseMemory",
                        targetHandle: "agentExecutor_0-input-memory-BaseMemory"
                    },
                    
                    // Tools connections to agent
                    {
                        id: "calculator_0-agentExecutor_0",
                        source: "calculator_0",
                        target: "agentExecutor_0",
                        sourceHandle: "calculator_0-output-calculator-Calculator|Tool",
                        targetHandle: "agentExecutor_0-input-tools-Tool"
                    },
                    {
                        id: "webBrowser_0-agentExecutor_0",
                        source: "webBrowser_0",
                        target: "agentExecutor_0",
                        sourceHandle: "webBrowser_0-output-customTool-CustomTool|Tool",
                        targetHandle: "agentExecutor_0-input-tools-Tool"
                    },
                    {
                        id: "knowledgeTool_0-agentExecutor_0",
                        source: "knowledgeTool_0",
                        target: "agentExecutor_0",
                        sourceHandle: "knowledgeTool_0-output-customTool-CustomTool|Tool",
                        targetHandle: "agentExecutor_0-input-tools-Tool"
                    },
                    
                    // LLM connection to agent
                    {
                        id: "agentLLM_0-agentExecutor_0",
                        source: "agentLLM_0",
                        target: "agentExecutor_0",
                        sourceHandle: "agentLLM_0-output-chatOpenAI-ChatOpenAI|BaseChatModel",
                        targetHandle: "agentExecutor_0-input-model-BaseChatModel"
                    },
                    
                    // System prompt connection
                    {
                        id: "systemPrompt_0-agentExecutor_0",
                        source: "systemPrompt_0",
                        target: "agentExecutor_0",
                        sourceHandle: "systemPrompt_0-output-promptTemplate-PromptTemplate|BasePromptTemplate",
                        targetHandle: "agentExecutor_0-input-agentPrompt-BasePromptTemplate"
                    }
                ]
            })
        };
    }
    
    /**
     * Create a new custom agent chatflow in Flowise
     */
    async createCustomAgentChatflow() {
        try {
            const config = this.createCustomAgentFlowConfig();
            
            const response = await fetch(`${this.baseUrl}/api/v1/chatflows`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to create custom agent chatflow: ${response.statusText} - ${errorText}`);
            }
            
            const result = await response.json();
            this.chatflowId = result.id;
            
            console.log(`✅ Custom agent chatflow created with ID: ${this.chatflowId}`);
            
            // Wait a moment for the flow to initialize
            console.log('⏳ Initializing agent tools and memory...');
            await this.sleep(3000);
            
            return this.chatflowId;
            
        } catch (error) {
            console.error('❌ Error creating custom agent chatflow:', error.message);
            throw error;
        }
    }
    
    /**
     * Interact with the custom agent
     */
    async chatWithAgent(message, chatHistory = []) {
        try {
            if (!this.chatflowId) {
                throw new Error('No custom agent chatflow created. Call createCustomAgentChatflow() first.');
            }
            
            const response = await fetch(`${this.baseUrl}/api/v1/prediction/${this.chatflowId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: message,
                    history: chatHistory,
                    sessionId: this.sessionId,
                    overrideConfig: {}
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Agent interaction failed: ${response.statusText} - ${errorText}`);
            }
            
            const result = await response.json();
            
            return {
                response: result.text || result.answer || 'No response received',
                intermediateSteps: result.intermediateSteps || [],
                toolsUsed: this.extractToolsUsed(result.intermediateSteps || []),
                chatHistory: result.chatHistory || chatHistory
            };
            
        } catch (error) {
            console.error('❌ Error chatting with agent:', error.message);
            return {
                response: `Error: ${error.message}`,
                intermediateSteps: [],
                toolsUsed: [],
                chatHistory: chatHistory
            };
        }
    }
    
    /**
     * Extract tools used from intermediate steps
     */
    extractToolsUsed(intermediateSteps) {
        const tools = [];
        
        for (const step of intermediateSteps) {
            if (step.action && step.action.tool) {
                tools.push({
                    tool: step.action.tool,
                    input: step.action.toolInput,
                    output: step.observation
                });
            }
        }
        
        return tools;
    }
    
    /**
     * Demonstrate the custom agent capabilities
     */
    async demonstrateAgentCapabilities() {
        console.log('\n🤖 Custom Agent Capabilities Demonstration');
        console.log('='.repeat(60));
        
        const demonstrations = [
            {
                category: "Mathematical Calculation",
                message: "What's 15% of 2,340 plus the square root of 144?",
                expectedTools: ["calculator"]
            },
            {
                category: "Knowledge Base Lookup", 
                message: "Tell me about RAG and how it works in AI systems.",
                expectedTools: ["knowledge_lookup"]
            },
            {
                category: "Web Search",
                message: "What are the latest developments in large language models this year?",
                expectedTools: ["web_search"]
            },
            {
                category: "Multi-Tool Usage",
                message: "If I have a dataset with 50,000 records and I want to process 15% of them using a machine learning model, how many records is that? Also, what should I know about machine learning?",
                expectedTools: ["calculator", "knowledge_lookup"]
            },
            {
                category: "Conversational Memory",
                message: "Based on what we discussed earlier about machine learning, what would be a good next step for someone learning AI?",
                expectedTools: []
            }
        ];
        
        let chatHistory = [];
        
        for (const demo of demonstrations) {
            console.log(`\n${'='.repeat(60)}`);
            console.log(`📋 Demonstration: ${demo.category}`);
            console.log(`💬 Message: "${demo.message}"`);
            console.log(`🔧 Expected Tools: ${demo.expectedTools.join(', ') || 'None'}`);
            console.log(`${'='.repeat(60)}`);
            
            const startTime = Date.now();
            const result = await this.chatWithAgent(demo.message, chatHistory);
            const responseTime = Date.now() - startTime;
            
            console.log(`\n🤖 Agent Response:`);
            console.log(`${result.response}`);
            
            console.log(`\n🔧 Tools Used: ${result.toolsUsed.length}`);
            result.toolsUsed.forEach((tool, index) => {
                console.log(`   ${index + 1}. ${tool.tool}`);
                console.log(`      Input: ${tool.input}`);
                console.log(`      Output: ${tool.output?.substring(0, 100)}...`);
            });
            
            console.log(`\n📊 Performance:`);
            console.log(`   ⏱️  Response time: ${responseTime}ms`);
            console.log(`   🔄 Steps taken: ${result.intermediateSteps.length || 0}`);
            
            // Update chat history for conversation continuity
            chatHistory = result.chatHistory;
            
            // Small delay between demonstrations
            await this.sleep(2000);
        }
    }
    
    /**
     * Interactive chat session with the custom agent
     */
    async runInteractiveChat() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
            prompt: '\n💬 You: '
        });
        
        console.log('\n🤖 Custom AI Agent Chat');
        console.log('Your agent has access to:');
        console.log('  🧮 Calculator - for mathematical operations');
        console.log('  🌐 Web Search - for current information (simulated)');
        console.log('  📚 Knowledge Base - for AI/tech information');
        console.log('  💭 Memory - remembers our conversation');
        console.log('\nCommands: "quit" to exit, "tools" to show tool usage, "memory" to show conversation summary');
        console.log('-'.repeat(60));
        
        let chatHistory = [];
        let showTools = true;
        
        rl.prompt();
        
        rl.on('line', async (input) => {
            const message = input.trim();
            
            if (message.toLowerCase() === 'quit') {
                console.log('\n👋 Goodbye! Thanks for chatting with the custom agent!');
                await this.cleanup();
                rl.close();
                return;
            }
            
            if (message.toLowerCase() === 'tools') {
                showTools = !showTools;
                console.log(`🔧 Tool usage display: ${showTools ? 'ON' : 'OFF'}`);
                rl.prompt();
                return;
            }
            
            if (message.toLowerCase() === 'memory') {
                console.log(`💭 Conversation History: ${chatHistory.length} exchanges`);
                if (chatHistory.length > 0) {
                    console.log('Recent context maintained by agent for personalized responses.');
                }
                rl.prompt();
                return;
            }
            
            if (message) {
                console.log('🤖 Agent is thinking and may use tools...');
                
                const result = await this.chatWithAgent(message, chatHistory);
                
                console.log(`\n🤖 Agent: ${result.response}`);
                
                if (showTools && result.toolsUsed.length > 0) {
                    console.log(`\n🔧 Tools Used:`);
                    result.toolsUsed.forEach((tool, index) => {
                        console.log(`   ${index + 1}. ${tool.tool}: ${tool.input}`);
                    });
                }
                
                // Update chat history
                chatHistory = result.chatHistory;
            }
            
            rl.prompt();
        });
        
        rl.on('SIGINT', async () => {
            console.log('\n\n👋 Goodbye!');
            await this.cleanup();
            process.exit(0);
        });
    }
    
    /**
     * Check if Flowise server is running
     */
    async checkServerStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/ping`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }
    
    /**
     * Clean up - delete the chatflow
     */
    async cleanup() {
        if (this.chatflowId) {
            try {
                await fetch(`${this.baseUrl}/api/v1/chatflows/${this.chatflowId}`, {
                    method: 'DELETE'
                });
                console.log('🧹 Custom agent chatflow cleaned up');
            } catch (error) {
                console.error('Warning: Failed to cleanup chatflow:', error.message);
            }
        }
    }
    
    /**
     * Sleep helper function
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Main function
 */
async function main() {
    console.log('🤖 Flowise Custom Agent with Tools & Memory');
    console.log('='.repeat(60));
    
    const agent = new FlowiseCustomAgent();
    
    try {
        // Check if server is running
        console.log('🔍 Checking Flowise server...');
        const serverRunning = await agent.checkServerStatus();
        
        if (!serverRunning) {
            console.error('❌ Flowise server is not running!');
            console.log('💡 Start it with: npx flowise start');
            process.exit(1);
        }
        
        console.log('✅ Flowise server is running');
        
        // Create custom agent chatflow
        console.log('\n🏗️  Creating custom agent with tools and memory...');
        await agent.createCustomAgentChatflow();
        
        // Ask user what they want to do
        console.log('\n🎯 Choose an option:');
        console.log('1. Run capability demonstrations');
        console.log('2. Start interactive chat');
        console.log('3. Both (demonstrations first, then chat)');
        
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        rl.question('\nEnter choice (1-3): ', async (choice) => {
            rl.close();
            
            try {
                switch (choice.trim()) {
                    case '1':
                        await agent.demonstrateAgentCapabilities();
                        await agent.cleanup();
                        break;
                    case '2':
                        await agent.runInteractiveChat();
                        break;
                    case '3':
                        await agent.demonstrateAgentCapabilities();
                        await agent.runInteractiveChat();
                        break;
                    default:
                        console.log('Invalid choice. Starting interactive chat...');
                        await agent.runInteractiveChat();
                }
            } catch (error) {
                console.error('❌ Custom agent failed:', error.message);
                await agent.cleanup();
            }
        });
        
    } catch (error) {
        console.error('❌ Failed to start custom agent:', error.message);
        await agent.cleanup();
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

module.exports = FlowiseCustomAgent;