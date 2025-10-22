"""
Multi-Agent Team Example with CrewAI
Demonstrates collaboration between multiple specialized agents.
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileReadTool, DirectoryReadTool

# Set up environment
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["SERPER_API_KEY"] = "your-serper-api-key"

def create_research_agent():
    """Create a research specialist agent."""
    
    search_tool = SerperDevTool()
    
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Gather comprehensive information and data on assigned topics',
        backstory="""You are a senior research analyst with expertise in market 
        analysis, competitive intelligence, and trend identification. You excel at 
        finding relevant information from multiple sources and synthesizing insights.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool]
    )
    
    return researcher

def create_writer_agent():
    """Create a content writer agent."""
    
    writer = Agent(
        role='Content Writer',
        goal='Create engaging and informative content based on research',
        backstory="""You are a skilled content writer with years of experience in 
        creating compelling articles, reports, and marketing materials. You have a 
        talent for making complex information accessible and engaging.""",
        verbose=True,
        allow_delegation=False
    )
    
    return writer

def create_editor_agent():
    """Create an editor agent for quality control."""
    
    editor = Agent(
        role='Senior Editor',
        goal='Review and improve content for clarity, accuracy, and engagement',
        backstory="""You are a senior editor with a keen eye for detail, style, 
        and structure. You ensure all content meets high standards for clarity, 
        accuracy, and reader engagement.""",
        verbose=True,
        allow_delegation=False
    )
    
    return editor

def create_content_pipeline_tasks(researcher, writer, editor, topic):
    """Create a pipeline of tasks for content creation."""
    
    # Research phase
    research_task = Task(
        description=f"""
        Conduct comprehensive research on: {topic}
        
        Focus areas:
        1. Current market landscape
        2. Key trends and developments
        3. Major players and their strategies
        4. Challenges and opportunities
        5. Future predictions
        
        Compile findings into a detailed research brief.
        """,
        agent=researcher,
        expected_output="Comprehensive research brief with data and insights"
    )
    
    # Writing phase
    writing_task = Task(
        description=f"""
        Using the research brief, create an engaging article about {topic}.
        
        Article requirements:
        1. Engaging headline and introduction
        2. Well-structured content with clear sections
        3. Include key statistics and insights from research
        4. Professional yet accessible tone
        5. Strong conclusion with actionable takeaways
        6. Target length: 1500-2000 words
        """,
        agent=writer,
        expected_output="Complete article draft ready for editing",
        context=[research_task]  # This task depends on research_task completion
    )
    
    # Editing phase
    editing_task = Task(
        description="""
        Review and edit the article for:
        
        1. Grammar, spelling, and punctuation
        2. Clarity and flow of ideas
        3. Factual accuracy
        4. Engagement and readability
        5. SEO optimization opportunities
        6. Overall structure and coherence
        
        Provide the final polished version.
        """,
        agent=editor,
        expected_output="Final edited article ready for publication",
        context=[writing_task]  # This task depends on writing_task completion
    )
    
    return [research_task, writing_task, editing_task]

def main():
    """Main function demonstrating multi-agent collaboration."""
    
    print("🤖 CrewAI Multi-Agent Team Example")
    print("=" * 45)
    
    # Create specialized agents
    print("👥 Creating specialized agents...")
    researcher = create_research_agent()
    writer = create_writer_agent()
    editor = create_editor_agent()
    
    # Define the topic
    topic = "The Future of AI Agent Frameworks in Enterprise Applications"
    
    # Create task pipeline
    print(f"\n📋 Setting up content pipeline for: {topic}")
    tasks = create_content_pipeline_tasks(researcher, writer, editor, topic)
    
    # Create crew with sequential process
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=tasks,
        process=Process.sequential,  # Tasks execute in order
        verbose=2
    )
    
    # Execute the workflow
    print("\n🚀 Starting collaborative content creation...")
    print("This may take a few minutes as agents work through the pipeline.")
    
    result = crew.kickoff()
    
    print("\n📄 Final Article:")
    print("=" * 45)
    print(result)
    
    print("\n✅ Multi-agent collaboration completed!")
    print("The team successfully researched, wrote, and edited the article.")

if __name__ == "__main__":
    main()