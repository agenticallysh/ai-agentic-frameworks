#!/usr/bin/env node
/**
 * Simple Chatbot Example - Flowise
 * =================================
 * 
 * This example demonstrates how to create and interact with a simple chatbot 
 * using Flowise's REST API. The chatbot uses OpenAI's GPT model.
 * 
 * Prerequisites:
 * - Node.js 18+ installed
 * - Flowise server running: npx flowise start
 * - OpenAI API key set in environment: OPENAI_API_KEY
 * 
 * Usage:
 *   node simple-chatbot.js
 */

const readline = require('readline');
const fetch = require('node-fetch');

class FlowiseChatbot {
    constructor(baseUrl = 'http://localhost:3000') {
        this.baseUrl = baseUrl;
        this.chatflowId = null;
        
        // Verify API key is set
        if (!process.env.OPENAI_API_KEY) {
            throw new Error('Please set OPENAI_API_KEY environment variable');
        }
    }
    
    /**
     * Create a simple chatbot flow configuration
     */
    createChatflowConfig() {
        return {
            name: "Simple Chatbot",
            flowData: JSON.stringify({
                nodes: [
                    {
                        id: "chatPromptTemplate_0",
                        position: { x: 100, y: 200 },
                        type: "customNode",
                        data: {
                            id: "chatPromptTemplate_0",
                            label: "ChatPromptTemplate",
                            name: "chatPromptTemplate",
                            type: "ChatPromptTemplate",
                            baseClasses: ["ChatPromptTemplate", "BasePromptTemplate"],
                            inputs: {
                                template: "You are a helpful AI assistant. Answer the user's question: {question}",
                                humanMessageVariableName: "question"
                            }
                        }
                    },
                    {
                        id: "chatOpenAI_0", 
                        position: { x: 400, y: 200 },
                        type: "customNode",
                        data: {
                            id: "chatOpenAI_0",
                            label: "ChatOpenAI",
                            name: "chatOpenAI",
                            type: "ChatOpenAI",
                            baseClasses: ["ChatOpenAI", "BaseChatModel"],
                            inputs: {
                                modelName: "gpt-3.5-turbo",
                                temperature: 0.7,
                                maxTokens: 1000,
                                topP: 1,
                                frequencyPenalty: 0,
                                presencePenalty: 0,
                                timeout: 60000,
                                openAIApiKey: process.env.OPENAI_API_KEY
                            }
                        }
                    }
                ],
                edges: [
                    {
                        id: "chatPromptTemplate_0-chatOpenAI_0",
                        source: "chatPromptTemplate_0",
                        target: "chatOpenAI_0",
                        sourceHandle: "chatPromptTemplate_0-output-chatPromptTemplate-ChatPromptTemplate|BasePromptTemplate",
                        targetHandle: "chatOpenAI_0-input-prompt-BasePromptTemplate"
                    }
                ]
            })
        };
    }
    
    /**
     * Create a new chatflow in Flowise
     */
    async createChatflow() {
        try {
            const config = this.createChatflowConfig();
            
            const response = await fetch(`${this.baseUrl}/api/v1/chatflows`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                throw new Error(`Failed to create chatflow: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.chatflowId = result.id;
            
            console.log(`✅ Chatflow created with ID: ${this.chatflowId}`);
            return this.chatflowId;
            
        } catch (error) {
            console.error('❌ Error creating chatflow:', error.message);
            throw error;
        }
    }
    
    /**
     * Send a message to the chatbot and get response
     */
    async sendMessage(question) {
        try {
            if (!this.chatflowId) {
                throw new Error('No chatflow created. Call createChatflow() first.');
            }
            
            const response = await fetch(`${this.baseUrl}/api/v1/prediction/${this.chatflowId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    overrideConfig: {}
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            return result.text || result.answer || 'No response received';
            
        } catch (error) {
            console.error('❌ Error sending message:', error.message);
            return `Error: ${error.message}`;
        }
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
                console.log('🧹 Chatflow cleaned up');
            } catch (error) {
                console.error('Warning: Failed to cleanup chatflow:', error.message);
            }
        }
    }
}

/**
 * Main chat loop
 */
async function main() {
    console.log('🤖 Flowise Simple Chatbot');
    console.log('=' .repeat(50));
    
    const chatbot = new FlowiseChatbot();
    
    // Check if server is running
    console.log('🔍 Checking Flowise server...');
    const serverRunning = await chatbot.checkServerStatus();
    
    if (!serverRunning) {
        console.error('❌ Flowise server is not running!');
        console.log('💡 Start it with: npx flowise start');
        process.exit(1);
    }
    
    console.log('✅ Flowise server is running');
    
    try {
        // Create chatflow
        console.log('🏗️  Creating chatflow...');
        await chatbot.createChatflow();
        
        // Setup readline interface
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
            prompt: '\n💬 You: '
        });
        
        console.log('\n🎉 Chatbot is ready! Type "quit" to exit.\n');
        rl.prompt();
        
        rl.on('line', async (input) => {
            const message = input.trim();
            
            if (message.toLowerCase() === 'quit') {
                console.log('\n👋 Goodbye!');
                await chatbot.cleanup();
                rl.close();
                return;
            }
            
            if (message) {
                console.log('🤖 Thinking...');
                const response = await chatbot.sendMessage(message);
                console.log(`🤖 Bot: ${response}`);
            }
            
            rl.prompt();
        });
        
        rl.on('SIGINT', async () => {
            console.log('\n\n👋 Goodbye!');
            await chatbot.cleanup();
            process.exit(0);
        });
        
    } catch (error) {
        console.error('❌ Failed to start chatbot:', error.message);
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

module.exports = FlowiseChatbot;