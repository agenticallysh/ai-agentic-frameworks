"""
Basic CrewAI Agent Example
A simple example showing how to create and run a single agent with CrewAI.
"""

import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Set up environment (you'll need to add your API keys)
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["SERPER_API_KEY"] = "your-serper-api-key"

def create_basic_agent():
    """Create a basic research agent."""
    
    # Initialize tools
    search_tool = SerperDevTool()
    
    # Create the agent
    researcher = Agent(
        role='Market Researcher',
        goal='Conduct comprehensive research on given topics',
        backstory="""You are an experienced market researcher with a keen eye for 
        identifying trends and extracting valuable insights from data. You're known 
        for your thorough analysis and clear reporting.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool]
    )
    
    return researcher

def create_research_task(agent, topic):
    """Create a research task for the agent."""
    
    task = Task(
        description=f"""
        Research the topic: {topic}
        
        Your analysis should include:
        1. Current market trends
        2. Key players and competitors
        3. Opportunities and challenges
        4. Future outlook
        
        Provide a comprehensive report with actionable insights.
        """,
        agent=agent,
        expected_output="A detailed research report with findings and recommendations"
    )
    
    return task

def main():
    """Main function to run the basic agent example."""
    
    print("🤖 CrewAI Basic Agent Example")
    print("=" * 40)
    
    # Create agent
    researcher = create_basic_agent()
    
    # Define research topic
    topic = "AI agent frameworks market trends 2024"
    
    # Create task
    research_task = create_research_task(researcher, topic)
    
    # Create crew with single agent
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=2
    )
    
    # Execute the task
    print(f"\n🔍 Starting research on: {topic}")
    result = crew.kickoff()
    
    print("\n📋 Research Results:")
    print("=" * 40)
    print(result)

if __name__ == "__main__":
    main()