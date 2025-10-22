"""
Basic AutoGen Conversation Example
Demonstrates a simple two-agent conversation for problem solving.
"""

import autogen
import os

# Configuration
config_list = [
    {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY")  # Set your API key
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "timeout": 120,
}

def create_basic_agents():
    """Create assistant and user proxy agents."""
    
    # Assistant agent - the AI helper
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message="""You are a helpful AI assistant. You can help with various tasks including:
        - Answering questions
        - Problem solving
        - Writing and editing
        - Analysis and research
        
        Always provide clear, helpful responses and ask for clarification when needed."""
    )
    
    # User proxy agent - represents the human
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="TERMINATE",  # Human input to terminate
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "workspace",
            "use_docker": False,  # Set to True for better security
        },
        llm_config=llm_config,
        system_message="""Reply TERMINATE if the task is solved to your satisfaction. 
        Otherwise, reply CONTINUE, or explain why the task is not solved yet."""
    )
    
    return assistant, user_proxy

def example_math_problem():
    """Example: Solving a math problem."""
    
    print("🧮 Example: Math Problem Solving")
    print("=" * 40)
    
    assistant, user_proxy = create_basic_agents()
    
    # Start conversation
    user_proxy.initiate_chat(
        assistant,
        message="""
        I need help with this math problem:
        
        A company's revenue grows by 15% each year. If the current revenue is $1,000,000, 
        what will be the revenue after 5 years? Please show the calculation step by step.
        """
    )

def example_creative_writing():
    """Example: Creative writing assistance."""
    
    print("\n✍️ Example: Creative Writing")
    print("=" * 40)
    
    assistant, user_proxy = create_basic_agents()
    
    # Start conversation
    user_proxy.initiate_chat(
        assistant,
        message="""
        Help me write a short story opening (2-3 paragraphs) about a detective who discovers 
        that their reflection has been acting independently. Make it engaging and mysterious.
        """
    )

def example_planning_task():
    """Example: Planning and organization."""
    
    print("\n📋 Example: Planning Task")
    print("=" * 40)
    
    assistant, user_proxy = create_basic_agents()
    
    # Start conversation
    user_proxy.initiate_chat(
        assistant,
        message="""
        I'm planning a weekend camping trip for 4 people. We need to consider:
        - Equipment and gear
        - Food planning
        - Activities
        - Safety considerations
        
        Can you help me create a comprehensive checklist and timeline?
        """
    )

def interactive_conversation():
    """Interactive conversation where user can input messages."""
    
    print("\n💬 Interactive Conversation Mode")
    print("=" * 40)
    print("Type 'exit' to end the conversation")
    
    assistant, user_proxy = create_basic_agents()
    
    # Modified user proxy for interactive mode
    user_proxy_interactive = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="ALWAYS",  # Always ask for human input
        max_consecutive_auto_reply=0,
        llm_config=llm_config,
    )
    
    # Get initial message from user
    initial_message = input("\nWhat would you like help with? ")
    
    if initial_message.lower() != 'exit':
        user_proxy_interactive.initiate_chat(
            assistant,
            message=initial_message
        )

def main():
    """Main function to run examples."""
    
    print("🤖 AutoGen Basic Conversation Examples")
    print("=" * 45)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Run different examples
        example_math_problem()
        
        input("\nPress Enter to continue to the next example...")
        example_creative_writing()
        
        input("\nPress Enter to continue to the next example...")
        example_planning_task()
        
        # Optional interactive mode
        user_choice = input("\nWould you like to try interactive mode? (y/n): ")
        if user_choice.lower() == 'y':
            interactive_conversation()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure your API key is valid and you have internet connectivity.")

if __name__ == "__main__":
    main()