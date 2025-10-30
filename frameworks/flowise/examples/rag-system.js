#!/usr/bin/env node
/**
 * RAG System Example - Flowise
 * =============================
 * 
 * This example demonstrates how to create a Retrieval-Augmented Generation (RAG) system
 * using Flowise. The system loads documents, creates embeddings, stores them in a vector
 * database, and answers questions based on the retrieved context.
 * 
 * Prerequisites:
 * - Node.js 18+ installed
 * - Flowise server running: npx flowise start
 * - OpenAI API key set in environment: OPENAI_API_KEY
 * - Sample documents in ./documents/ directory
 * 
 * Usage:
 *   node rag-system.js
 */

const fs = require('fs').promises;
const path = require('path');
const readline = require('readline');
const fetch = require('node-fetch');

class FlowiseRAGSystem {
    constructor(baseUrl = 'http://localhost:3000') {
        this.baseUrl = baseUrl;
        this.chatflowId = null;
        this.documentsPath = './documents';
        
        // Verify API key is set
        if (!process.env.OPENAI_API_KEY) {
            throw new Error('Please set OPENAI_API_KEY environment variable');
        }
    }
    
    /**
     * Create a RAG system flow configuration
     */
    createRAGFlowConfig() {
        return {
            name: "RAG Document Q&A System",
            flowData: JSON.stringify({
                nodes: [
                    // Document Loader Node
                    {
                        id: "folderFiles_0",
                        position: { x: 100, y: 100 },
                        type: "customNode",
                        data: {
                            id: "folderFiles_0",
                            label: "Folder Files",
                            name: "folderFiles",
                            type: "FolderFiles",
                            baseClasses: ["Document"],
                            inputs: {
                                folderPath: this.documentsPath,
                                recursive: true,
                                ignoredFiles: ".DS_Store"
                            }
                        }
                    },
                    
                    // Text Splitter Node
                    {
                        id: "recursiveCharacterTextSplitter_0",
                        position: { x: 400, y: 100 },
                        type: "customNode",
                        data: {
                            id: "recursiveCharacterTextSplitter_0",
                            label: "Recursive Character Text Splitter",
                            name: "recursiveCharacterTextSplitter",
                            type: "RecursiveCharacterTextSplitter",
                            baseClasses: ["RecursiveCharacterTextSplitter", "TextSplitter"],
                            inputs: {
                                chunkSize: 1000,
                                chunkOverlap: 200,
                                separators: "\\n\\n,\\n, ,."
                            }
                        }
                    },
                    
                    // OpenAI Embeddings Node
                    {
                        id: "openAIEmbeddings_0",
                        position: { x: 700, y: 100 },
                        type: "customNode",
                        data: {
                            id: "openAIEmbeddings_0",
                            label: "OpenAI Embeddings",
                            name: "openAIEmbeddings",
                            type: "OpenAIEmbeddings",
                            baseClasses: ["OpenAIEmbeddings", "Embeddings"],
                            inputs: {
                                modelName: "text-embedding-ada-002",
                                stripNewLines: true,
                                batchSize: 512,
                                timeout: 60000,
                                openAIApiKey: process.env.OPENAI_API_KEY
                            }
                        }
                    },
                    
                    // In-Memory Vector Store
                    {
                        id: "memoryVectorStore_0",
                        position: { x: 1000, y: 100 },
                        type: "customNode",
                        data: {
                            id: "memoryVectorStore_0",
                            label: "In-Memory Vector Store",
                            name: "memoryVectorStore",
                            type: "MemoryVectorStore",
                            baseClasses: ["MemoryVectorStore", "VectorStore"],
                            inputs: {}
                        }
                    },
                    
                    // Conversational Retrieval QA Chain
                    {
                        id: "conversationalRetrievalQAChain_0",
                        position: { x: 700, y: 400 },
                        type: "customNode",
                        data: {
                            id: "conversationalRetrievalQAChain_0",
                            label: "Conversational Retrieval QA Chain",
                            name: "conversationalRetrievalQAChain",
                            type: "ConversationalRetrievalQAChain",
                            baseClasses: ["ConversationalRetrievalQAChain", "BaseChain"],
                            inputs: {
                                returnSourceDocuments: true,
                                rephrasedQuestion: true,
                                responseTopK: 5,
                                systemMessagePrompt: `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always provide the source of your information when possible.

{context}

Question: {question}
Helpful Answer:`
                            }
                        }
                    },
                    
                    // ChatOpenAI Model
                    {
                        id: "chatOpenAI_0",
                        position: { x: 400, y: 400 },
                        type: "customNode",
                        data: {
                            id: "chatOpenAI_0",
                            label: "ChatOpenAI",
                            name: "chatOpenAI",
                            type: "ChatOpenAI",
                            baseClasses: ["ChatOpenAI", "BaseChatModel"],
                            inputs: {
                                modelName: "gpt-4",
                                temperature: 0.1,
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
                    // Document processing pipeline
                    {
                        id: "folderFiles_0-recursiveCharacterTextSplitter_0",
                        source: "folderFiles_0",
                        target: "recursiveCharacterTextSplitter_0",
                        sourceHandle: "folderFiles_0-output-folderFiles-Document",
                        targetHandle: "recursiveCharacterTextSplitter_0-input-documents-Document"
                    },
                    {
                        id: "recursiveCharacterTextSplitter_0-memoryVectorStore_0",
                        source: "recursiveCharacterTextSplitter_0",
                        target: "memoryVectorStore_0",
                        sourceHandle: "recursiveCharacterTextSplitter_0-output-recursiveCharacterTextSplitter-Document",
                        targetHandle: "memoryVectorStore_0-input-documents-Document"
                    },
                    {
                        id: "openAIEmbeddings_0-memoryVectorStore_0",
                        source: "openAIEmbeddings_0",
                        target: "memoryVectorStore_0",
                        sourceHandle: "openAIEmbeddings_0-output-openAIEmbeddings-OpenAIEmbeddings|Embeddings",
                        targetHandle: "memoryVectorStore_0-input-embeddings-Embeddings"
                    },
                    
                    // QA Chain connections
                    {
                        id: "memoryVectorStore_0-conversationalRetrievalQAChain_0",
                        source: "memoryVectorStore_0",
                        target: "conversationalRetrievalQAChain_0",
                        sourceHandle: "memoryVectorStore_0-output-retriever-VectorStoreRetriever",
                        targetHandle: "conversationalRetrievalQAChain_0-input-vectorStoreRetriever-VectorStoreRetriever"
                    },
                    {
                        id: "chatOpenAI_0-conversationalRetrievalQAChain_0",
                        source: "chatOpenAI_0",
                        target: "conversationalRetrievalQAChain_0",
                        sourceHandle: "chatOpenAI_0-output-chatOpenAI-ChatOpenAI|BaseChatModel",
                        targetHandle: "conversationalRetrievalQAChain_0-input-model-BaseChatModel"
                    }
                ]
            })
        };
    }
    
    /**
     * Create sample documents for testing
     */
    async createSampleDocuments() {
        try {
            await fs.mkdir(this.documentsPath, { recursive: true });
            
            const sampleDocs = {
                'ai_basics.txt': `
Artificial Intelligence (AI) is a branch of computer science that aims to create
intelligent machines that work and react like humans. AI systems can perform tasks
that typically require human intelligence, such as visual perception, speech
recognition, decision-making, and language translation.

There are three main types of AI:
1. Narrow AI (ANI) - AI that is designed to perform a narrow task
2. General AI (AGI) - AI that has generalized human cognitive abilities  
3. Super AI (ASI) - AI that surpasses human intelligence in all areas

Machine Learning is a subset of AI that provides systems the ability to
automatically learn and improve from experience without being explicitly programmed.

Deep Learning is a subset of machine learning that uses neural networks with
multiple layers to model and understand complex patterns in data.
                `,
                
                'flowise_guide.txt': `
Flowise is an open-source low-code tool for developers to build customized LLM
orchestration flows and AI agents. It provides a visual interface to create
complex AI workflows without extensive coding.

Key Features:
- Drag & drop interface for building AI flows
- Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- Vector database integration for RAG applications
- Memory management for conversational agents
- Custom function calling capabilities
- API generation for created flows

Flowise is built on top of LangChain and provides a visual abstraction layer
that makes it easier to create production-ready AI applications.

Use Cases:
- Document Q&A systems
- Conversational AI chatbots
- Content generation workflows
- Data analysis and insights
- Customer support automation
                `,
                
                'rag_concepts.txt': `
Retrieval-Augmented Generation (RAG) is a technique that combines information
retrieval with text generation to create more accurate and contextual AI responses.

The RAG process involves several steps:
1. Document Ingestion - Loading and processing documents
2. Chunking - Breaking documents into manageable pieces
3. Embedding - Converting text chunks into vector representations
4. Storage - Storing embeddings in a vector database
5. Retrieval - Finding relevant chunks for a given query
6. Generation - Using retrieved context to generate responses

Benefits of RAG:
- Reduces hallucinations in AI responses
- Provides up-to-date information beyond training data
- Allows for source attribution and transparency
- Enables domain-specific knowledge integration
- Maintains context relevance for user queries

Vector databases commonly used with RAG:
- Pinecone (cloud-based)
- Chroma (open-source)
- Weaviate (open-source)
- Qdrant (open-source)
- FAISS (Facebook AI)
                `
            };
            
            for (const [filename, content] of Object.entries(sampleDocs)) {
                const filePath = path.join(this.documentsPath, filename);
                await fs.writeFile(filePath, content.trim());
            }
            
            console.log(`✅ Created ${Object.keys(sampleDocs).length} sample documents in ${this.documentsPath}`);
            
        } catch (error) {
            console.error('❌ Error creating sample documents:', error.message);
            throw error;
        }
    }
    
    /**
     * Create a new RAG chatflow in Flowise
     */
    async createRAGChatflow() {
        try {
            const config = this.createRAGFlowConfig();
            
            const response = await fetch(`${this.baseUrl}/api/v1/chatflows`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to create RAG chatflow: ${response.statusText} - ${errorText}`);
            }
            
            const result = await response.json();
            this.chatflowId = result.id;
            
            console.log(`✅ RAG chatflow created with ID: ${this.chatflowId}`);
            
            // Wait a moment for the flow to initialize
            console.log('⏳ Initializing document processing...');
            await this.sleep(3000);
            
            return this.chatflowId;
            
        } catch (error) {
            console.error('❌ Error creating RAG chatflow:', error.message);
            throw error;
        }
    }
    
    /**
     * Query the RAG system with a question
     */
    async queryRAG(question, chatHistory = []) {
        try {
            if (!this.chatflowId) {
                throw new Error('No RAG chatflow created. Call createRAGChatflow() first.');
            }
            
            const response = await fetch(`${this.baseUrl}/api/v1/prediction/${this.chatflowId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    history: chatHistory,
                    overrideConfig: {}
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`RAG query failed: ${response.statusText} - ${errorText}`);
            }
            
            const result = await response.json();
            
            return {
                answer: result.text || result.answer || 'No response received',
                sourceDocuments: result.sourceDocuments || [],
                chatHistory: result.chatHistory || chatHistory
            };
            
        } catch (error) {
            console.error('❌ Error querying RAG system:', error.message);
            return {
                answer: `Error: ${error.message}`,
                sourceDocuments: [],
                chatHistory: chatHistory
            };
        }
    }
    
    /**
     * Check document indexing status
     */
    async checkIndexingStatus() {
        try {
            // Try a simple query to see if documents are indexed
            const testResult = await this.queryRAG("What is AI?");
            
            if (testResult.sourceDocuments && testResult.sourceDocuments.length > 0) {
                console.log(`✅ Documents indexed successfully! Found ${testResult.sourceDocuments.length} relevant sources.`);
                return true;
            } else {
                console.log('⏳ Documents may still be indexing...');
                return false;
            }
            
        } catch (error) {
            console.log('⏳ Indexing in progress or not ready yet...');
            return false;
        }
    }
    
    /**
     * Display source information
     */
    displaySources(sourceDocuments) {
        if (!sourceDocuments || sourceDocuments.length === 0) {
            console.log('📚 No source documents found');
            return;
        }
        
        console.log(`\n📚 Sources (${sourceDocuments.length}):`);
        sourceDocuments.forEach((doc, index) => {
            const source = doc.metadata?.source || 'Unknown source';
            const content = doc.pageContent?.substring(0, 100) || 'No content';
            console.log(`  ${index + 1}. ${source}: "${content}..."`);
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
                console.log('🧹 RAG chatflow cleaned up');
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
 * Interactive RAG system demo
 */
async function runInteractiveRAG() {
    const rag = new FlowiseRAGSystem();
    let chatHistory = [];
    
    // Setup readline interface
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: '\n❓ Question: '
    });
    
    console.log('\n🤖 RAG System - Ask questions about the documents!');
    console.log('Commands: "quit" to exit, "sources" to toggle source display');
    console.log('-'.repeat(60));
    
    let showSources = true;
    
    rl.prompt();
    
    rl.on('line', async (input) => {
        const question = input.trim();
        
        if (question.toLowerCase() === 'quit') {
            console.log('\n👋 Goodbye!');
            await rag.cleanup();
            rl.close();
            return;
        }
        
        if (question.toLowerCase() === 'sources') {
            showSources = !showSources;
            console.log(`📚 Source display: ${showSources ? 'ON' : 'OFF'}`);
            rl.prompt();
            return;
        }
        
        if (question) {
            console.log('🔍 Searching documents and generating answer...');
            
            const result = await rag.queryRAG(question, chatHistory);
            console.log(`\n🤖 Answer: ${result.answer}`);
            
            if (showSources) {
                rag.displaySources(result.sourceDocuments);
            }
            
            // Update chat history
            chatHistory = result.chatHistory;
        }
        
        rl.prompt();
    });
    
    rl.on('SIGINT', async () => {
        console.log('\n\n👋 Goodbye!');
        await rag.cleanup();
        process.exit(0);
    });
    
    return rag;
}

/**
 * Demonstration of various RAG queries
 */
async function demonstrateRAGQueries(rag) {
    console.log('\n🔄 Running demonstration queries...');
    console.log('='.repeat(50));
    
    const testQueries = [
        "What is Artificial Intelligence?",
        "What are the key features of Flowise?", 
        "How does RAG work?",
        "What are the benefits of using vector databases?",
        "What types of AI are there?"
    ];
    
    let chatHistory = [];
    
    for (const query of testQueries) {
        console.log(`\n❓ Question: ${query}`);
        console.log('🔍 Processing...');
        
        const result = await rag.queryRAG(query, chatHistory);
        
        console.log(`🤖 Answer: ${result.answer}`);
        
        if (result.sourceDocuments && result.sourceDocuments.length > 0) {
            console.log(`📚 Sources: ${result.sourceDocuments.length} documents found`);
        }
        
        chatHistory = result.chatHistory;
        
        // Small delay between queries
        await rag.sleep(1000);
    }
}

/**
 * Main function
 */
async function main() {
    console.log('🤖 Flowise RAG System Demo');
    console.log('='.repeat(50));
    
    const rag = new FlowiseRAGSystem();
    
    try {
        // Check if server is running
        console.log('🔍 Checking Flowise server...');
        const serverRunning = await rag.checkServerStatus();
        
        if (!serverRunning) {
            console.error('❌ Flowise server is not running!');
            console.log('💡 Start it with: npx flowise start');
            process.exit(1);
        }
        
        console.log('✅ Flowise server is running');
        
        // Create sample documents
        console.log('\n📚 Setting up sample documents...');
        await rag.createSampleDocuments();
        
        // Create RAG chatflow
        console.log('\n🏗️  Creating RAG chatflow...');
        await rag.createRAGChatflow();
        
        // Wait for document indexing
        console.log('\n⏳ Waiting for document indexing...');
        let indexed = false;
        let attempts = 0;
        const maxAttempts = 10;
        
        while (!indexed && attempts < maxAttempts) {
            await rag.sleep(2000);
            indexed = await rag.checkIndexingStatus();
            attempts++;
        }
        
        if (!indexed) {
            console.log('⚠️  Documents may still be indexing. Proceeding anyway...');
        }
        
        // Ask user what they want to do
        console.log('\n🎯 Choose an option:');
        console.log('1. Run interactive RAG session');
        console.log('2. Run demonstration queries');
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
                        await runInteractiveRAG();
                        break;
                    case '2':
                        await demonstrateRAGQueries(rag);
                        await rag.cleanup();
                        break;
                    case '3':
                        await demonstrateRAGQueries(rag);
                        await runInteractiveRAG();
                        break;
                    default:
                        console.log('Invalid choice. Running interactive session...');
                        await runInteractiveRAG();
                }
            } catch (error) {
                console.error('❌ Demo failed:', error.message);
                await rag.cleanup();
            }
        });
        
    } catch (error) {
        console.error('❌ Failed to start RAG system:', error.message);
        await rag.cleanup();
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

module.exports = FlowiseRAGSystem;