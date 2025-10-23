"""
Basic LangChain Chain Example
Demonstrates how to create a simple prompt-LLM-parser chain.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set up environment (you'll need to add your API key)
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

def create_simple_chain():
    """Create a basic chain with prompt, LLM, and output parser."""
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        max_tokens=500
    )
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that explains complex topics in simple terms. 
        Your explanations should be:
        - Clear and easy to understand
        - Use analogies when helpful
        - Provide practical examples
        - Be engaging and informative"""),
        ("user", "{topic}")
    ])
    
    # Create output parser
    output_parser = StrOutputParser()
    
    # Chain the components together using the | operator
    chain = prompt | llm | output_parser
    
    return chain

def create_formatted_chain():
    """Create a chain with additional formatting."""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # More complex prompt with multiple variables
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert {role} with years of experience."),
        ("user", """Please explain {topic} in the context of {context}.
        
        Format your response as:
        1. Brief overview
        2. Key concepts
        3. Practical applications
        4. Common challenges
        
        Keep it under {word_limit} words.""")
    ])
    
    # Chain with formatting
    chain = (
        {
            "role": RunnablePassthrough(),
            "topic": RunnablePassthrough(),
            "context": RunnablePassthrough(),
            "word_limit": RunnablePassthrough()
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

def example_simple_explanation():
    """Example: Explain a complex topic simply."""
    
    print("🔗 Simple Chain Example")
    print("=" * 40)
    
    chain = create_simple_chain()
    
    topics = [
        "quantum computing",
        "machine learning",
        "blockchain technology",
        "neural networks"
    ]
    
    for topic in topics:
        print(f"\n📚 Explaining: {topic}")
        print("-" * 30)
        
        try:
            result = chain.invoke({"topic": topic})
            print(result)
        except Exception as e:
            print(f"Error explaining {topic}: {e}")

def example_formatted_explanation():
    """Example: Use the formatted chain with multiple variables."""
    
    print("\n🎯 Formatted Chain Example")
    print("=" * 40)
    
    chain = create_formatted_chain()
    
    # Example explanation requests
    requests = [
        {
            "role": "data scientist",
            "topic": "neural networks",
            "context": "business applications",
            "word_limit": "300"
        },
        {
            "role": "software engineer",
            "topic": "microservices architecture",
            "context": "startup environments",
            "word_limit": "250"
        }
    ]
    
    for request in requests:
        print(f"\n👨‍💼 Role: {request['role']}")
        print(f"📖 Topic: {request['topic']}")
        print(f"🎯 Context: {request['context']}")
        print("-" * 30)
        
        try:
            result = chain.invoke(request)
            print(result)
        except Exception as e:
            print(f"Error processing request: {e}")

def example_batch_processing():
    """Example: Process multiple topics in batch."""
    
    print("\n⚡ Batch Processing Example")
    print("=" * 40)
    
    chain = create_simple_chain()
    
    # Batch multiple topics
    topics = [
        {"topic": "artificial intelligence"},
        {"topic": "cloud computing"},
        {"topic": "cybersecurity"}
    ]
    
    print("Processing topics in batch...")
    
    try:
        results = chain.batch(topics)
        
        for i, result in enumerate(results):
            print(f"\n📝 Topic {i+1}: {topics[i]['topic']}")
            print("-" * 30)
            print(result[:200] + "..." if len(result) > 200 else result)
            
    except Exception as e:
        print(f"Batch processing error: {e}")

def example_streaming():
    """Example: Stream responses for real-time output."""
    
    print("\n🌊 Streaming Example")
    print("=" * 40)
    
    chain = create_simple_chain()
    
    topic = "the future of artificial intelligence"
    print(f"Streaming explanation of: {topic}")
    print("-" * 30)
    
    try:
        for chunk in chain.stream({"topic": topic}):
            print(chunk, end="", flush=True)
        print("\n")  # New line after streaming
        
    except Exception as e:
        print(f"Streaming error: {e}")

def main():
    """Main function to run all chain examples."""
    
    print("🔗 LangChain Basic Chain Examples")
    print("=" * 45)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Run different chain examples
        example_simple_explanation()
        
        input("\nPress Enter to continue to formatted example...")
        example_formatted_explanation()
        
        input("\nPress Enter to continue to batch processing...")
        example_batch_processing()
        
        input("\nPress Enter to continue to streaming example...")
        example_streaming()
        
        print("\n✅ All chain examples completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure your API key is valid and you have internet connectivity.")

if __name__ == "__main__":
    main()