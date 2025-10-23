"""
LangChain RAG (Retrieval-Augmented Generation) System Example
Demonstrates how to build a document Q&A system with retrieval.
"""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set up environment
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

def format_docs(docs):
    """Format retrieved documents for the prompt."""
    return "\n\n".join([d.page_content for d in docs])

def create_sample_documents():
    """Create sample documents for demonstration."""
    
    documents = [
        {
            "content": """
            LangChain is a framework for developing applications powered by language models. 
            It enables applications that are context-aware and can reason. The framework 
            consists of several parts: LangChain Libraries, LangChain Templates, LangServe, 
            and LangSmith. It supports Python and JavaScript.
            """,
            "metadata": {"source": "langchain_intro", "topic": "framework"}
        },
        {
            "content": """
            CrewAI is designed for orchestrating role-playing, autonomous AI agents. 
            By fostering collaborative intelligence, CrewAI empowers agents to work 
            together seamlessly, tackling complex tasks. It's built in Python and 
            focuses on multi-agent coordination.
            """,
            "metadata": {"source": "crewai_intro", "topic": "multi-agent"}
        },
        {
            "content": """
            AutoGen is a framework that enables the development of LLM applications using 
            multiple agents that can converse with each other to solve tasks. It's developed 
            by Microsoft and supports various conversation patterns including two-agent, 
            group chat, and hierarchical structures.
            """,
            "metadata": {"source": "autogen_intro", "topic": "conversation"}
        },
        {
            "content": """
            Retrieval-Augmented Generation (RAG) is a technique that combines the power of 
            large language models with external knowledge sources. It retrieves relevant 
            information from a knowledge base and uses it to generate more accurate and 
            contextual responses.
            """,
            "metadata": {"source": "rag_explanation", "topic": "technique"}
        },
        {
            "content": """
            Vector databases are specialized databases designed to store and query 
            high-dimensional vectors efficiently. They are essential for RAG applications, 
            enabling semantic search over large document collections. Popular options 
            include Pinecone, Chroma, Weaviate, and Qdrant.
            """,
            "metadata": {"source": "vector_db_info", "topic": "database"}
        }
    ]
    
    return documents

def setup_vector_store():
    """Set up a vector store with sample documents."""
    
    print("📚 Setting up vector store...")
    
    # Create sample documents
    sample_docs = create_sample_documents()
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    # Create Document objects
    from langchain_core.documents import Document
    documents = []
    for doc in sample_docs:
        # Split the document if needed
        splits = text_splitter.split_text(doc["content"])
        for split in splits:
            documents.append(Document(
                page_content=split,
                metadata=doc["metadata"]
            ))
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"✅ Vector store created with {len(documents)} document chunks")
    return vectorstore

def create_rag_chain(vectorstore):
    """Create a RAG chain with retriever and generator."""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.3  # Lower temperature for more factual responses
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 most similar documents
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer based on the context, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        Context: {context}"""),
        ("user", "{question}")
    ])
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def example_basic_rag():
    """Basic RAG example with sample questions."""
    
    print("🔍 Basic RAG Example")
    print("=" * 40)
    
    # Set up vector store and RAG chain
    vectorstore = setup_vector_store()
    rag_chain, retriever = create_rag_chain(vectorstore)
    
    # Sample questions
    questions = [
        "What is LangChain and what does it do?",
        "How does CrewAI differ from other frameworks?",
        "What is RAG and why is it useful?",
        "What are vector databases used for?",
        "Which framework is best for multi-agent systems?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 50)
        
        try:
            # Get answer from RAG chain
            answer = rag_chain.invoke(question)
            print(f"🤖 Answer: {answer}")
            
            # Show retrieved context
            docs = retriever.get_relevant_documents(question)
            print(f"\n📄 Retrieved {len(docs)} relevant documents:")
            for i, doc in enumerate(docs, 1):
                print(f"  {i}. Source: {doc.metadata.get('source', 'unknown')}")
                print(f"     Preview: {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"Error processing question: {e}")

def example_advanced_rag():
    """Advanced RAG example with custom retrieval and generation."""
    
    print("\n🎯 Advanced RAG Example")
    print("=" * 40)
    
    vectorstore = setup_vector_store()
    
    # Create custom retriever with different parameters
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 4,
            "fetch_k": 8,
            "lambda_mult": 0.7
        }
    )
    
    # Advanced LLM with specific instructions
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        max_tokens=300
    )
    
    # Detailed prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI framework consultant. Use the provided context to answer questions about AI agent frameworks.
        
        Guidelines:
        - Provide specific, actionable information
        - Compare frameworks when relevant
        - Include pros and cons when appropriate
        - Cite the source information when possible
        - If information is insufficient, clearly state what's missing
        
        Context:
        {context}
        
        Previous conversation:
        {chat_history}"""),
        ("user", "{question}")
    ])
    
    # Advanced chain with chat history
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: ""  # Simplified for this example
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Complex questions
    complex_questions = [
        "I'm building a customer service bot that needs to handle complex multi-step conversations. Which framework should I choose and why?",
        "What are the trade-offs between using LangChain vs CrewAI for a document analysis system?",
        "How do I decide whether to use RAG or fine-tuning for my knowledge-intensive application?"
    ]
    
    for question in complex_questions:
        print(f"\n🤔 Complex Question: {question}")
        print("-" * 60)
        
        try:
            answer = rag_chain.invoke(question)
            print(f"💡 Expert Answer: {answer}")
            
        except Exception as e:
            print(f"Error processing complex question: {e}")

def example_web_rag():
    """Example: RAG with web content."""
    
    print("\n🌐 Web Content RAG Example")
    print("=" * 40)
    
    try:
        # Load web content
        urls = [
            "https://python.langchain.com/docs/introduction/",
            # Add more URLs as needed
        ]
        
        print("🔄 Loading web content...")
        
        # Note: This is a simplified example
        # In practice, you'd want to handle errors and timeouts
        documents = []
        
        # For demonstration, we'll use our sample documents
        # In a real scenario, you'd use WebBaseLoader
        print("📝 Using sample documents for demonstration...")
        vectorstore = setup_vector_store()
        rag_chain, _ = create_rag_chain(vectorstore)
        
        question = "What are the main components of LangChain?"
        answer = rag_chain.invoke(question)
        
        print(f"❓ Question: {question}")
        print(f"🌐 Answer from web content: {answer}")
        
    except Exception as e:
        print(f"Web RAG error: {e}")

def cleanup():
    """Clean up resources."""
    
    print("\n🧹 Cleaning up...")
    
    # Remove the Chroma database directory
    import shutil
    try:
        shutil.rmtree("./chroma_db")
        print("✅ Cleaned up vector database")
    except Exception as e:
        print(f"Cleanup warning: {e}")

def main():
    """Main function to run RAG examples."""
    
    print("🔍 LangChain RAG System Examples")
    print("=" * 45)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Run RAG examples
        example_basic_rag()
        
        input("\nPress Enter to continue to advanced RAG...")
        example_advanced_rag()
        
        input("\nPress Enter to continue to web RAG example...")
        example_web_rag()
        
        print("\n✅ All RAG examples completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure your API key is valid and you have internet connectivity.")
    
    finally:
        # Clean up
        cleanup()

if __name__ == "__main__":
    main()