#!/usr/bin/env node
/**
 * Multi-Chain Flow Example - Flowise
 * ===================================
 * 
 * This example demonstrates how to create complex workflow orchestration using
 * multiple chains in Flowise. The flow processes user input through multiple
 * specialized chains for content analysis, sentiment analysis, and response generation.
 * 
 * Prerequisites:
 * - Node.js 18+ installed
 * - Flowise server running: npx flowise start
 * - OpenAI API key set in environment: OPENAI_API_KEY
 * 
 * Usage:
 *   node multi-chain-flow.js
 */

const readline = require('readline');
const fetch = require('node-fetch');

class FlowiseMultiChainOrchestrator {
    constructor(baseUrl = 'http://localhost:3000') {
        this.baseUrl = baseUrl;
        this.chatflowId = null;
        
        // Verify API key is set
        if (!process.env.OPENAI_API_KEY) {
            throw new Error('Please set OPENAI_API_KEY environment variable');
        }
    }
    
    /**
     * Create a multi-chain orchestration flow configuration
     */
    createMultiChainFlowConfig() {
        return {
            name: "Multi-Chain Content Processor",
            flowData: JSON.stringify({
                nodes: [
                    // Input Processing Chain
                    {
                        id: "inputProcessor_0",
                        position: { x: 100, y: 100 },
                        type: "customNode",
                        data: {
                            id: "inputProcessor_0",
                            label: "Input Processor",
                            name: "llmChain",
                            type: "LLMChain",
                            baseClasses: ["LLMChain", "BaseChain"],
                            inputs: {
                                chainName: "Input Processing"
                            }
                        }
                    },
                    
                    // Content Analysis Chain
                    {
                        id: "contentAnalyzer_0",
                        position: { x: 100, y: 300 },
                        type: "customNode",
                        data: {
                            id: "contentAnalyzer_0",
                            label: "Content Analyzer",
                            name: "llmChain",
                            type: "LLMChain",
                            baseClasses: ["LLMChain", "BaseChain"],
                            inputs: {
                                chainName: "Content Analysis"
                            }
                        }
                    },
                    
                    // Sentiment Analysis Chain
                    {
                        id: "sentimentAnalyzer_0",
                        position: { x: 100, y: 500 },
                        type: "customNode",
                        data: {
                            id: "sentimentAnalyzer_0",
                            label: "Sentiment Analyzer",
                            name: "llmChain",
                            type: "LLMChain",
                            baseClasses: ["LLMChain", "BaseChain"],
                            inputs: {
                                chainName: "Sentiment Analysis"
                            }
                        }
                    },
                    
                    // Response Generator Chain
                    {
                        id: "responseGenerator_0",
                        position: { x: 600, y: 300 },
                        type: "customNode",
                        data: {
                            id: "responseGenerator_0",
                            label: "Response Generator",
                            name: "llmChain",
                            type: "LLMChain",
                            baseClasses: ["LLMChain", "BaseChain"],
                            inputs: {
                                chainName: "Response Generation"
                            }
                        }
                    },
                    
                    // Input Processing Prompt
                    {
                        id: "inputPrompt_0",
                        position: { x: 400, y: 100 },
                        type: "customNode",
                        data: {
                            id: "inputPrompt_0",
                            label: "Input Processing Prompt",
                            name: "promptTemplate",
                            type: "PromptTemplate",
                            baseClasses: ["PromptTemplate", "BasePromptTemplate"],
                            inputs: {
                                template: `You are an input processing specialist. Analyze the user input and extract key information.

User Input: {userInput}

Extract and format the following:
1. Main topic/subject
2. Intent (question, request, complaint, compliment, etc.)
3. Key entities (people, places, things)
4. Urgency level (low, medium, high)
5. Required processing type (simple response, detailed analysis, action needed)

Format your response as a structured analysis:`,
                                inputVariables: ["userInput"]
                            }
                        }
                    },
                    
                    // Content Analysis Prompt
                    {
                        id: "contentPrompt_0",
                        position: { x: 400, y: 300 },
                        type: "customNode",
                        data: {
                            id: "contentPrompt_0",
                            label: "Content Analysis Prompt",
                            name: "promptTemplate",
                            type: "PromptTemplate",
                            baseClasses: ["PromptTemplate", "BasePromptTemplate"],
                            inputs: {
                                template: `You are a content analysis expert. Analyze the following input for content characteristics.

Input Analysis: {inputAnalysis}
Original Input: {userInput}

Provide detailed content analysis including:
1. Content type (factual, opinion, creative, technical, etc.)
2. Complexity level (basic, intermediate, advanced)
3. Domain/field (technology, business, personal, academic, etc.)
4. Required expertise level
5. Potential challenges in responding
6. Recommended response approach

Format as structured analysis:`,
                                inputVariables: ["inputAnalysis", "userInput"]
                            }
                        }
                    },
                    
                    // Sentiment Analysis Prompt
                    {
                        id: "sentimentPrompt_0",
                        position: { x: 400, y: 500 },
                        type: "customNode",
                        data: {
                            id: "sentimentPrompt_0",
                            label: "Sentiment Analysis Prompt",
                            name: "promptTemplate",
                            type: "PromptTemplate",
                            baseClasses: ["PromptTemplate", "BasePromptTemplate"],
                            inputs: {
                                template: `You are a sentiment analysis specialist. Analyze the emotional tone and sentiment of the user input.

Input Analysis: {inputAnalysis}
Original Input: {userInput}

Provide comprehensive sentiment analysis:
1. Overall sentiment (positive, negative, neutral, mixed)
2. Emotion indicators (joy, anger, frustration, excitement, etc.)
3. Confidence level (how certain you are about the sentiment)
4. Emotional intensity (low, medium, high)
5. Context considerations
6. Recommended response tone

Format as structured sentiment report:`,
                                inputVariables: ["inputAnalysis", "userInput"]
                            }
                        }
                    },
                    
                    // Response Generation Prompt
                    {
                        id: "responsePrompt_0",
                        position: { x: 900, y: 300 },
                        type: "customNode",
                        data: {
                            id: "responsePrompt_0",
                            label: "Response Generation Prompt",
                            name: "promptTemplate",
                            type: "PromptTemplate",
                            baseClasses: ["PromptTemplate", "BasePromptTemplate"],
                            inputs: {
                                template: `You are a response generation expert. Using all the analysis provided, create an appropriate response to the user.

Original Input: {userInput}
Input Analysis: {inputAnalysis}
Content Analysis: {contentAnalysis}
Sentiment Analysis: {sentimentAnalysis}

Create a response that:
1. Addresses the user's needs based on input analysis
2. Matches the appropriate complexity and domain from content analysis
3. Uses the right tone based on sentiment analysis
4. Is helpful, accurate, and engaging
5. Shows understanding of the user's emotional state

Generate a thoughtful, well-crafted response:`,
                                inputVariables: ["userInput", "inputAnalysis", "contentAnalysis", "sentimentAnalysis"]
                            }
                        }
                    },
                    
                    // LLM Models for each chain
                    {
                        id: "inputLLM_0",
                        position: { x: 700, y: 100 },
                        type: "customNode",
                        data: {
                            id: "inputLLM_0",
                            label: "Input Processing LLM",
                            name: "chatOpenAI",
                            type: "ChatOpenAI",
                            baseClasses: ["ChatOpenAI", "BaseChatModel"],
                            inputs: {
                                modelName: "gpt-4",
                                temperature: 0.3,
                                maxTokens: 500,
                                openAIApiKey: process.env.OPENAI_API_KEY
                            }
                        }
                    },
                    
                    {
                        id: "contentLLM_0",
                        position: { x: 700, y: 300 },
                        type: "customNode",
                        data: {
                            id: "contentLLM_0",
                            label: "Content Analysis LLM",
                            name: "chatOpenAI",
                            type: "ChatOpenAI",
                            baseClasses: ["ChatOpenAI", "BaseChatModel"],
                            inputs: {
                                modelName: "gpt-4",
                                temperature: 0.2,
                                maxTokens: 600,
                                openAIApiKey: process.env.OPENAI_API_KEY
                            }
                        }
                    },
                    
                    {
                        id: "sentimentLLM_0",
                        position: { x: 700, y: 500 },
                        type: "customNode",
                        data: {
                            id: "sentimentLLM_0",
                            label: "Sentiment Analysis LLM",
                            name: "chatOpenAI",
                            type: "ChatOpenAI",
                            baseClasses: ["ChatOpenAI", "BaseChatModel"],
                            inputs: {
                                modelName: "gpt-3.5-turbo",
                                temperature: 0.1,
                                maxTokens: 400,
                                openAIApiKey: process.env.OPENAI_API_KEY
                            }
                        }
                    },
                    
                    {
                        id: "responseLLM_0",
                        position: { x: 1200, y: 300 },
                        type: "customNode",
                        data: {
                            id: "responseLLM_0",
                            label: "Response Generation LLM",
                            name: "chatOpenAI",
                            type: "ChatOpenAI",
                            baseClasses: ["ChatOpenAI", "BaseChatModel"],
                            inputs: {
                                modelName: "gpt-4",
                                temperature: 0.7,
                                maxTokens: 1000,
                                openAIApiKey: process.env.OPENAI_API_KEY
                            }
                        }
                    },
                    
                    // Sequential Agent for orchestration
                    {
                        id: "sequentialAgent_0",
                        position: { x: 1000, y: 100 },
                        type: "customNode",
                        data: {
                            id: "sequentialAgent_0",
                            label: "Sequential Processing Agent",
                            name: "sequentialAgent",
                            type: "SequentialAgent",
                            baseClasses: ["SequentialAgent", "BaseAgent"],
                            inputs: {
                                agentName: "Multi-Chain Orchestrator",
                                systemMessage: "You orchestrate multiple chains for comprehensive input processing and response generation.",
                                verbose: true
                            }
                        }
                    }
                ],
                edges: [
                    // Input Processing Chain connections
                    {
                        id: "inputPrompt_0-inputProcessor_0",
                        source: "inputPrompt_0",
                        target: "inputProcessor_0",
                        sourceHandle: "inputPrompt_0-output-promptTemplate-PromptTemplate|BasePromptTemplate",
                        targetHandle: "inputProcessor_0-input-prompt-BasePromptTemplate"
                    },
                    {
                        id: "inputLLM_0-inputProcessor_0",
                        source: "inputLLM_0",
                        target: "inputProcessor_0",
                        sourceHandle: "inputLLM_0-output-chatOpenAI-ChatOpenAI|BaseChatModel",
                        targetHandle: "inputProcessor_0-input-model-BaseLanguageModel"
                    },
                    
                    // Content Analysis Chain connections
                    {
                        id: "contentPrompt_0-contentAnalyzer_0",
                        source: "contentPrompt_0",
                        target: "contentAnalyzer_0",
                        sourceHandle: "contentPrompt_0-output-promptTemplate-PromptTemplate|BasePromptTemplate",
                        targetHandle: "contentAnalyzer_0-input-prompt-BasePromptTemplate"
                    },
                    {
                        id: "contentLLM_0-contentAnalyzer_0",
                        source: "contentLLM_0",
                        target: "contentAnalyzer_0",
                        sourceHandle: "contentLLM_0-output-chatOpenAI-ChatOpenAI|BaseChatModel",
                        targetHandle: "contentAnalyzer_0-input-model-BaseLanguageModel"
                    },
                    
                    // Sentiment Analysis Chain connections
                    {
                        id: "sentimentPrompt_0-sentimentAnalyzer_0",
                        source: "sentimentPrompt_0",
                        target: "sentimentAnalyzer_0",
                        sourceHandle: "sentimentPrompt_0-output-promptTemplate-PromptTemplate|BasePromptTemplate",
                        targetHandle: "sentimentAnalyzer_0-input-prompt-BasePromptTemplate"
                    },
                    {
                        id: "sentimentLLM_0-sentimentAnalyzer_0",
                        source: "sentimentLLM_0",
                        target: "sentimentAnalyzer_0",
                        sourceHandle: "sentimentLLM_0-output-chatOpenAI-ChatOpenAI|BaseChatModel",
                        targetHandle: "sentimentAnalyzer_0-input-model-BaseLanguageModel"
                    },
                    
                    // Response Generation Chain connections
                    {
                        id: "responsePrompt_0-responseGenerator_0",
                        source: "responsePrompt_0",
                        target: "responseGenerator_0",
                        sourceHandle: "responsePrompt_0-output-promptTemplate-PromptTemplate|BasePromptTemplate",
                        targetHandle: "responseGenerator_0-input-prompt-BasePromptTemplate"
                    },
                    {
                        id: "responseLLM_0-responseGenerator_0",
                        source: "responseLLM_0",
                        target: "responseGenerator_0",
                        sourceHandle: "responseLLM_0-output-chatOpenAI-ChatOpenAI|BaseChatModel",
                        targetHandle: "responseGenerator_0-input-model-BaseLanguageModel"
                    },
                    
                    // Sequential Agent connections
                    {
                        id: "inputProcessor_0-sequentialAgent_0",
                        source: "inputProcessor_0",
                        target: "sequentialAgent_0",
                        sourceHandle: "inputProcessor_0-output-llmChain-LLMChain|BaseChain",
                        targetHandle: "sequentialAgent_0-input-tools-Tool"
                    },
                    {
                        id: "contentAnalyzer_0-sequentialAgent_0",
                        source: "contentAnalyzer_0",
                        target: "sequentialAgent_0",
                        sourceHandle: "contentAnalyzer_0-output-llmChain-LLMChain|BaseChain",
                        targetHandle: "sequentialAgent_0-input-tools-Tool"
                    },
                    {
                        id: "sentimentAnalyzer_0-sequentialAgent_0",
                        source: "sentimentAnalyzer_0",
                        target: "sequentialAgent_0",
                        sourceHandle: "sentimentAnalyzer_0-output-llmChain-LLMChain|BaseChain",
                        targetHandle: "sequentialAgent_0-input-tools-Tool"
                    },
                    {
                        id: "responseGenerator_0-sequentialAgent_0",
                        source: "responseGenerator_0",
                        target: "sequentialAgent_0",
                        sourceHandle: "responseGenerator_0-output-llmChain-LLMChain|BaseChain",
                        targetHandle: "sequentialAgent_0-input-tools-Tool"
                    }
                ]
            })
        };
    }
    
    /**
     * Create a new multi-chain chatflow in Flowise
     */
    async createMultiChainChatflow() {
        try {
            const config = this.createMultiChainFlowConfig();
            
            const response = await fetch(`${this.baseUrl}/api/v1/chatflows`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to create multi-chain chatflow: ${response.statusText} - ${errorText}`);
            }
            
            const result = await response.json();
            this.chatflowId = result.id;
            
            console.log(`✅ Multi-chain chatflow created with ID: ${this.chatflowId}`);
            
            // Wait a moment for the flow to initialize
            console.log('⏳ Initializing multi-chain processors...');
            await this.sleep(3000);
            
            return this.chatflowId;
            
        } catch (error) {
            console.error('❌ Error creating multi-chain chatflow:', error.message);
            throw error;
        }
    }
    
    /**
     * Process input through the multi-chain workflow
     */
    async processInput(userInput) {
        try {
            if (!this.chatflowId) {
                throw new Error('No multi-chain chatflow created. Call createMultiChainChatflow() first.');
            }
            
            console.log('🔄 Processing through multiple chains...');
            
            const response = await fetch(`${this.baseUrl}/api/v1/prediction/${this.chatflowId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: userInput,
                    overrideConfig: {}
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Multi-chain processing failed: ${response.statusText} - ${errorText}`);
            }
            
            const result = await response.json();
            
            return {
                finalResponse: result.text || result.answer || 'No response received',
                processingSteps: result.processingSteps || [],
                metadata: result.metadata || {}
            };
            
        } catch (error) {
            console.error('❌ Error processing input:', error.message);
            return {
                finalResponse: `Error: ${error.message}`,
                processingSteps: [],
                metadata: {}
            };
        }
    }
    
    /**
     * Demonstrate multi-chain processing with various inputs
     */
    async demonstrateMultiChain() {
        console.log('\n🔄 Multi-Chain Processing Demonstration');
        console.log('='.repeat(50));
        
        const testInputs = [
            {
                category: "Technical Question",
                input: "I'm having trouble setting up a RAG system with vector embeddings. Can you help me understand the best practices?"
            },
            {
                category: "Complaint",
                input: "I'm frustrated with this AI system! It keeps giving me wrong answers and wasting my time."
            },
            {
                category: "Creative Request",
                input: "Could you help me write a short story about an AI that discovers emotions? Make it inspiring and thoughtful."
            },
            {
                category: "Business Query",
                input: "What are the key considerations for implementing AI in a customer service department of a mid-sized company?"
            },
            {
                category: "Personal Advice",
                input: "I'm feeling overwhelmed with learning about AI and machine learning. Where should I start as a complete beginner?"
            }
        ];
        
        for (const testCase of testInputs) {
            console.log(`\n${'='.repeat(60)}`);
            console.log(`📋 Category: ${testCase.category}`);
            console.log(`📝 Input: "${testCase.input}"`);
            console.log(`${'='.repeat(60)}`);
            
            const startTime = Date.now();
            const result = await this.processInput(testCase.input);
            const processingTime = Date.now() - startTime;
            
            console.log(`\n🤖 Final Response:`);
            console.log(`${result.finalResponse}`);
            
            console.log(`\n📊 Processing Info:`);
            console.log(`   ⏱️  Processing time: ${processingTime}ms`);
            console.log(`   🔗 Steps completed: ${result.processingSteps.length || 'N/A'}`);
            
            if (result.metadata && Object.keys(result.metadata).length > 0) {
                console.log(`   📋 Metadata: ${JSON.stringify(result.metadata, null, 2)}`);
            }
            
            // Small delay between demonstrations
            await this.sleep(2000);
        }
    }
    
    /**
     * Interactive multi-chain processing session
     */
    async runInteractiveSession() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
            prompt: '\n💭 Enter your input: '
        });
        
        console.log('\n🤖 Multi-Chain Processing System');
        console.log('Your input will be processed through multiple specialized chains:');
        console.log('  1. 🔍 Input Analysis - Understanding your request');
        console.log('  2. 📄 Content Analysis - Analyzing content characteristics');
        console.log('  3. 💭 Sentiment Analysis - Understanding emotional context');
        console.log('  4. ✨ Response Generation - Creating appropriate response');
        console.log('\nType "quit" to exit, "demo" for demonstrations');
        console.log('-'.repeat(60));
        
        rl.prompt();
        
        rl.on('line', async (input) => {
            const userInput = input.trim();
            
            if (userInput.toLowerCase() === 'quit') {
                console.log('\n👋 Goodbye!');
                await this.cleanup();
                rl.close();
                return;
            }
            
            if (userInput.toLowerCase() === 'demo') {
                rl.pause();
                await this.demonstrateMultiChain();
                rl.resume();
                rl.prompt();
                return;
            }
            
            if (userInput) {
                console.log('\n🚀 Processing through multi-chain workflow...');
                console.log('   Step 1/4: Analyzing input structure and intent...');
                
                const result = await this.processInput(userInput);
                
                console.log('   Step 2/4: Analyzing content characteristics...');
                console.log('   Step 3/4: Performing sentiment analysis...');
                console.log('   Step 4/4: Generating tailored response...');
                console.log('\n✅ Processing complete!');
                
                console.log(`\n🎯 Final Response:`);
                console.log(`${result.finalResponse}`);
                
                if (result.processingSteps && result.processingSteps.length > 0) {
                    console.log(`\n🔄 Processing completed in ${result.processingSteps.length} steps`);
                }
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
                console.log('🧹 Multi-chain chatflow cleaned up');
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
    console.log('🤖 Flowise Multi-Chain Workflow Orchestration');
    console.log('='.repeat(60));
    
    const orchestrator = new FlowiseMultiChainOrchestrator();
    
    try {
        // Check if server is running
        console.log('🔍 Checking Flowise server...');
        const serverRunning = await orchestrator.checkServerStatus();
        
        if (!serverRunning) {
            console.error('❌ Flowise server is not running!');
            console.log('💡 Start it with: npx flowise start');
            process.exit(1);
        }
        
        console.log('✅ Flowise server is running');
        
        // Create multi-chain chatflow
        console.log('\n🏗️  Creating multi-chain workflow...');
        await orchestrator.createMultiChainChatflow();
        
        // Ask user what they want to do
        console.log('\n🎯 Choose an option:');
        console.log('1. Run interactive session');
        console.log('2. Run demonstrations');
        console.log('3. Both');
        
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        rl.question('\nEnter choice (1-3): ', async (choice) => {
            rl.close();
            
            try {
                switch (choice.trim()) {
                    case '1':
                        await orchestrator.runInteractiveSession();
                        break;
                    case '2':
                        await orchestrator.demonstrateMultiChain();
                        await orchestrator.cleanup();
                        break;
                    case '3':
                        await orchestrator.demonstrateMultiChain();
                        await orchestrator.runInteractiveSession();
                        break;
                    default:
                        console.log('Invalid choice. Running interactive session...');
                        await orchestrator.runInteractiveSession();
                }
            } catch (error) {
                console.error('❌ Multi-chain workflow failed:', error.message);
                await orchestrator.cleanup();
            }
        });
        
    } catch (error) {
        console.error('❌ Failed to start multi-chain orchestrator:', error.message);
        await orchestrator.cleanup();
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

module.exports = FlowiseMultiChainOrchestrator;