"""
Basic LangGraph Example
Demonstrates how to create a simple graph-based workflow with nodes and edges.
"""

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Set up environment
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Define the state structure
class WorkflowState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    user_input: str
    analysis_result: str
    processing_step: str
    final_output: str

def create_basic_graph():
    """Create a basic graph with sequential processing."""
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Define node functions
    def input_processor(state: WorkflowState):
        """Process the initial user input."""
        user_input = state.get("user_input", "")
        
        # Add processing logic
        processed_input = f"Processing: {user_input}"
        
        return {
            **state,
            "processing_step": "input_processed",
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"Input received: {user_input}")
            ]
        }
    
    def analyzer(state: WorkflowState):
        """Analyze the processed input."""
        user_input = state.get("user_input", "")
        
        # Create analysis prompt
        analysis_prompt = f"""
        Analyze the following user input and provide insights:
        Input: {user_input}
        
        Provide a brief analysis covering:
        1. Intent recognition
        2. Key topics mentioned
        3. Suggested response approach
        """
        
        try:
            response = llm.invoke([HumanMessage(content=analysis_prompt)])
            analysis_result = response.content
        except Exception as e:
            analysis_result = f"Analysis failed: {str(e)}"
        
        return {
            **state,
            "analysis_result": analysis_result,
            "processing_step": "analyzed",
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Analysis completed: {analysis_result[:100]}...")
            ]
        }
    
    def response_generator(state: WorkflowState):
        """Generate the final response based on analysis."""
        analysis = state.get("analysis_result", "")
        user_input = state.get("user_input", "")
        
        response_prompt = f"""
        Based on the following analysis, generate a helpful response to the user:
        
        Original user input: {user_input}
        Analysis: {analysis}
        
        Generate a clear, helpful, and engaging response.
        """
        
        try:
            response = llm.invoke([HumanMessage(content=response_prompt)])
            final_output = response.content
        except Exception as e:
            final_output = f"Response generation failed: {str(e)}"
        
        return {
            **state,
            "final_output": final_output,
            "processing_step": "completed",
            "messages": state.get("messages", []) + [
                AIMessage(content=final_output)
            ]
        }
    
    # Create the workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("input_processor", input_processor)
    workflow.add_node("analyzer", analyzer)
    workflow.add_node("response_generator", response_generator)
    
    # Add edges (define the flow)
    workflow.add_edge("input_processor", "analyzer")
    workflow.add_edge("analyzer", "response_generator")
    workflow.add_edge("response_generator", END)
    
    # Set entry point
    workflow.set_entry_point("input_processor")
    
    return workflow.compile()

def create_conditional_graph():
    """Create a graph with conditional routing based on input type."""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def input_classifier(state: WorkflowState):
        """Classify the type of user input."""
        user_input = state.get("user_input", "")
        
        classification_prompt = f"""
        Classify the following user input into one of these categories:
        - question: User is asking a question
        - request: User is making a request for action
        - complaint: User is expressing dissatisfaction
        - compliment: User is giving positive feedback
        
        Input: {user_input}
        
        Respond with just the category name.
        """
        
        try:
            response = llm.invoke([HumanMessage(content=classification_prompt)])
            classification = response.content.strip().lower()
        except Exception as e:
            classification = "question"  # Default fallback
        
        return {
            **state,
            "analysis_result": classification,
            "processing_step": "classified"
        }
    
    def handle_question(state: WorkflowState):
        """Handle question-type inputs."""
        user_input = state.get("user_input", "")
        
        prompt = f"""
        The user has asked a question: {user_input}
        
        Provide a comprehensive and helpful answer.
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            **state,
            "final_output": f"Answer: {response.content}",
            "processing_step": "question_handled"
        }
    
    def handle_request(state: WorkflowState):
        """Handle request-type inputs."""
        user_input = state.get("user_input", "")
        
        prompt = f"""
        The user has made a request: {user_input}
        
        Provide guidance on how to fulfill this request or explain next steps.
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            **state,
            "final_output": f"Request guidance: {response.content}",
            "processing_step": "request_handled"
        }
    
    def handle_complaint(state: WorkflowState):
        """Handle complaint-type inputs."""
        user_input = state.get("user_input", "")
        
        prompt = f"""
        The user has expressed a complaint: {user_input}
        
        Provide an empathetic response and suggest solutions.
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            **state,
            "final_output": f"Complaint response: {response.content}",
            "processing_step": "complaint_handled"
        }
    
    def handle_compliment(state: WorkflowState):
        """Handle compliment-type inputs."""
        user_input = state.get("user_input", "")
        
        return {
            **state,
            "final_output": "Thank you for your kind words! We appreciate your feedback.",
            "processing_step": "compliment_handled"
        }
    
    # Routing function
    def route_by_type(state: WorkflowState):
        """Route to appropriate handler based on classification."""
        classification = state.get("analysis_result", "question")
        
        routing_map = {
            "question": "handle_question",
            "request": "handle_request", 
            "complaint": "handle_complaint",
            "compliment": "handle_compliment"
        }
        
        return routing_map.get(classification, "handle_question")
    
    # Create workflow
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("input_classifier", input_classifier)
    workflow.add_node("handle_question", handle_question)
    workflow.add_node("handle_request", handle_request)
    workflow.add_node("handle_complaint", handle_complaint)
    workflow.add_node("handle_compliment", handle_compliment)
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "input_classifier",
        route_by_type,
        {
            "handle_question": "handle_question",
            "handle_request": "handle_request",
            "handle_complaint": "handle_complaint",
            "handle_compliment": "handle_compliment"
        }
    )
    
    # Add edges to end
    workflow.add_edge("handle_question", END)
    workflow.add_edge("handle_request", END)
    workflow.add_edge("handle_complaint", END)
    workflow.add_edge("handle_compliment", END)
    
    # Set entry point
    workflow.set_entry_point("input_classifier")
    
    return workflow.compile()

def example_basic_workflow():
    """Example: Run basic sequential workflow."""
    
    print("🔄 Basic Sequential Workflow")
    print("=" * 40)
    
    app = create_basic_graph()
    
    test_inputs = [
        "I need help understanding machine learning",
        "What are the benefits of using AI in business?",
        "Can you explain how neural networks work?"
    ]
    
    for user_input in test_inputs:
        print(f"\n📝 Input: {user_input}")
        print("-" * 30)
        
        try:
            # Run the workflow
            initial_state = {
                "user_input": user_input,
                "messages": [],
                "analysis_result": "",
                "processing_step": "",
                "final_output": ""
            }
            
            result = app.invoke(initial_state)
            
            print(f"🔍 Analysis: {result['analysis_result'][:100]}...")
            print(f"📄 Final Output: {result['final_output'][:200]}...")
            print(f"✅ Status: {result['processing_step']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

def example_conditional_workflow():
    """Example: Run conditional routing workflow."""
    
    print("\n🔀 Conditional Routing Workflow")
    print("=" * 40)
    
    app = create_conditional_graph()
    
    test_inputs = [
        "How does machine learning work?",  # Question
        "I need you to schedule a meeting",  # Request
        "Your service is too slow!",  # Complaint
        "Great job on the new features!"  # Compliment
    ]
    
    for user_input in test_inputs:
        print(f"\n📝 Input: {user_input}")
        print("-" * 30)
        
        try:
            initial_state = {
                "user_input": user_input,
                "messages": [],
                "analysis_result": "",
                "processing_step": "",
                "final_output": ""
            }
            
            result = app.invoke(initial_state)
            
            print(f"🏷️ Classified as: {result['analysis_result']}")
            print(f"📄 Response: {result['final_output'][:200]}...")
            print(f"✅ Handler: {result['processing_step']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

def example_streaming_workflow():
    """Example: Stream workflow execution to see intermediate steps."""
    
    print("\n🌊 Streaming Workflow Execution")
    print("=" * 40)
    
    app = create_basic_graph()
    
    user_input = "Explain the difference between AI and machine learning"
    
    print(f"📝 Input: {user_input}")
    print("\n🔄 Streaming execution steps:")
    print("-" * 30)
    
    try:
        initial_state = {
            "user_input": user_input,
            "messages": [],
            "analysis_result": "",
            "processing_step": "",
            "final_output": ""
        }
        
        # Stream the workflow execution
        step_count = 0
        for step in app.stream(initial_state):
            step_count += 1
            print(f"\n📍 Step {step_count}:")
            
            for node_name, node_output in step.items():
                print(f"  🔧 Node: {node_name}")
                print(f"  📊 Processing Step: {node_output.get('processing_step', 'N/A')}")
                
                if node_output.get('final_output'):
                    print(f"  ✅ Final Output Available")
        
        print(f"\n🏁 Workflow completed in {step_count} steps")
        
    except Exception as e:
        print(f"❌ Streaming error: {e}")

def main():
    """Main function to run all graph examples."""
    
    print("📊 LangGraph Basic Graph Examples")
    print("=" * 45)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Run different graph examples
        example_basic_workflow()
        
        input("\nPress Enter to continue to conditional workflow...")
        example_conditional_workflow()
        
        input("\nPress Enter to continue to streaming example...")
        example_streaming_workflow()
        
        print("\n✅ All graph examples completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure your API key is valid and you have internet connectivity.")

if __name__ == "__main__":
    main()