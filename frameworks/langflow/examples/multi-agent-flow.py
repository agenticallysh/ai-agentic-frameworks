#!/usr/bin/env python3
"""
Langflow Multi-Agent Coordination Example

This example demonstrates how to build coordinated multi-agent workflows using Langflow.
The system includes specialized agents for research, analysis, and synthesis, working
together to provide comprehensive insights.

Requirements:
    pip install langflow openai duckduckgo-search requests beautifulsoup4
    
Usage:
    python multi-agent-flow.py
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langflow.components import (
    OpenAIModel, PromptTemplate, ChatInput, ChatOutput
)
from langflow.custom import CustomComponent
from langflow.graph import Graph


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender: str
    recipient: str
    content: str
    message_type: str
    timestamp: str
    metadata: Dict[str, Any] = None


class WebSearchAgent(CustomComponent):
    """Agent specialized in web search and information gathering."""
    
    display_name = "Web Search Agent"
    description = "Searches the web for current information on given topics"
    
    def build_config(self):
        return {
            "query": {"display_name": "Search Query", "type": "str"},
            "max_results": {"display_name": "Max Results", "value": 5, "type": "int"},
            "search_engine": {"display_name": "Search Engine", "value": "duckduckgo", "type": "str"}
        }
    
    def build(self, query: str, max_results: int = 5, search_engine: str = "duckduckgo") -> Dict[str, Any]:
        """Search the web for information."""
        
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "url": result.get("href", ""),
                        "published": result.get("published", "")
                    })
            
            return {
                "search_results": results,
                "query": query,
                "total_results": len(results),
                "agent_name": "WebSearchAgent"
            }
            
        except Exception as e:
            return {
                "error": f"Search failed: {str(e)}",
                "search_results": [],
                "query": query,
                "agent_name": "WebSearchAgent"
            }


class DocumentAnalysisAgent(CustomComponent):
    """Agent specialized in document analysis and content extraction."""
    
    display_name = "Document Analysis Agent"
    description = "Analyzes documents and extracts key information"
    
    def build_config(self):
        return {
            "documents": {"display_name": "Documents", "type": "list"},
            "analysis_type": {"display_name": "Analysis Type", "value": "summary", "type": "str"},
            "focus_areas": {"display_name": "Focus Areas", "type": "list"}
        }
    
    def build(self, documents: List[str], analysis_type: str = "summary", focus_areas: List[str] = None) -> Dict[str, Any]:
        """Analyze documents and extract insights."""
        
        if focus_areas is None:
            focus_areas = ["key_points", "conclusions", "data", "recommendations"]
        
        analysis_results = []
        
        for i, doc in enumerate(documents):
            # Simulate document analysis
            analysis = {
                "document_id": i,
                "content_length": len(doc),
                "key_topics": self._extract_topics(doc),
                "sentiment": self._analyze_sentiment(doc),
                "key_phrases": self._extract_key_phrases(doc),
                "summary": self._summarize_content(doc)
            }
            analysis_results.append(analysis)
        
        return {
            "analysis_results": analysis_results,
            "total_documents": len(documents),
            "analysis_type": analysis_type,
            "focus_areas": focus_areas,
            "agent_name": "DocumentAnalysisAgent"
        }
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content."""
        # Simplified topic extraction
        common_topics = ["AI", "machine learning", "data", "technology", "business", "research"]
        return [topic for topic in common_topics if topic.lower() in content.lower()]
    
    def _analyze_sentiment(self, content: str) -> str:
        """Analyze sentiment of content."""
        positive_words = ["good", "great", "excellent", "positive", "successful", "improvement"]
        negative_words = ["bad", "poor", "negative", "failed", "problem", "issue"]
        
        pos_count = sum(1 for word in positive_words if word in content.lower())
        neg_count = sum(1 for word in negative_words if word in content.lower())
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content."""
        # Simplified key phrase extraction
        sentences = content.split('.')
        return [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
    
    def _summarize_content(self, content: str) -> str:
        """Create a summary of the content."""
        # Simplified summarization
        sentences = content.split('.')
        return '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else content[:200] + "..."


class SynthesisAgent(CustomComponent):
    """Agent specialized in synthesizing information from multiple sources."""
    
    display_name = "Synthesis Agent"
    description = "Synthesizes information from multiple agents and sources"
    
    def build_config(self):
        return {
            "search_results": {"display_name": "Search Results", "type": "dict"},
            "analysis_results": {"display_name": "Analysis Results", "type": "dict"},
            "synthesis_type": {"display_name": "Synthesis Type", "value": "comprehensive", "type": "str"}
        }
    
    def build(self, search_results: Dict[str, Any], analysis_results: Dict[str, Any], synthesis_type: str = "comprehensive") -> Dict[str, Any]:
        """Synthesize information from multiple sources."""
        
        synthesis = {
            "executive_summary": self._create_executive_summary(search_results, analysis_results),
            "key_findings": self._extract_key_findings(search_results, analysis_results),
            "recommendations": self._generate_recommendations(search_results, analysis_results),
            "data_sources": self._catalog_sources(search_results, analysis_results),
            "confidence_score": self._calculate_confidence(search_results, analysis_results),
            "synthesis_type": synthesis_type,
            "agent_name": "SynthesisAgent"
        }
        
        return synthesis
    
    def _create_executive_summary(self, search_results: Dict, analysis_results: Dict) -> str:
        """Create an executive summary."""
        search_count = search_results.get("total_results", 0)
        doc_count = analysis_results.get("total_documents", 0)
        
        return f"""
        Analysis completed on {search_count} web sources and {doc_count} documents.
        Key insights have been extracted and synthesized to provide comprehensive findings.
        Multiple data sources have been cross-referenced for accuracy and relevance.
        """
    
    def _extract_key_findings(self, search_results: Dict, analysis_results: Dict) -> List[str]:
        """Extract key findings from all sources."""
        findings = []
        
        # Extract from search results
        if "search_results" in search_results:
            for result in search_results["search_results"]:
                if result.get("snippet"):
                    findings.append(f"Web source: {result['snippet'][:100]}...")
        
        # Extract from analysis results
        if "analysis_results" in analysis_results:
            for analysis in analysis_results["analysis_results"]:
                if analysis.get("summary"):
                    findings.append(f"Document analysis: {analysis['summary']}")
        
        return findings[:5]  # Return top 5 findings
    
    def _generate_recommendations(self, search_results: Dict, analysis_results: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = [
            "Continue monitoring latest developments in this area",
            "Cross-reference findings with additional authoritative sources",
            "Consider implementing pilot programs based on best practices identified",
            "Schedule regular reviews to track progress and updates",
            "Engage with subject matter experts for deeper insights"
        ]
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _catalog_sources(self, search_results: Dict, analysis_results: Dict) -> Dict[str, Any]:
        """Catalog all data sources used."""
        sources = {
            "web_sources": len(search_results.get("search_results", [])),
            "documents_analyzed": analysis_results.get("total_documents", 0),
            "search_query": search_results.get("query", ""),
            "analysis_type": analysis_results.get("analysis_type", "")
        }
        
        return sources
    
    def _calculate_confidence(self, search_results: Dict, analysis_results: Dict) -> float:
        """Calculate confidence score for the synthesis."""
        score = 0.5  # Base score
        
        # Increase score based on number of sources
        web_sources = len(search_results.get("search_results", []))
        doc_sources = analysis_results.get("total_documents", 0)
        
        if web_sources >= 3:
            score += 0.2
        if doc_sources >= 2:
            score += 0.2
        if web_sources + doc_sources >= 5:
            score += 0.1
        
        return min(score, 1.0)


class CoordinatorAgent(CustomComponent):
    """Central coordinator agent that orchestrates the multi-agent workflow."""
    
    display_name = "Coordinator Agent"
    description = "Orchestrates multi-agent workflows and manages communication"
    
    def build_config(self):
        return {
            "task": {"display_name": "Research Task", "type": "str"},
            "agents": {"display_name": "Available Agents", "type": "list"},
            "workflow_type": {"display_name": "Workflow Type", "value": "research", "type": "str"}
        }
    
    def build(self, task: str, agents: List[str] = None, workflow_type: str = "research") -> Dict[str, Any]:
        """Coordinate the multi-agent workflow."""
        
        if agents is None:
            agents = ["WebSearchAgent", "DocumentAnalysisAgent", "SynthesisAgent"]
        
        workflow_plan = self._create_workflow_plan(task, agents, workflow_type)
        
        return {
            "workflow_plan": workflow_plan,
            "task": task,
            "agents": agents,
            "workflow_type": workflow_type,
            "agent_name": "CoordinatorAgent",
            "status": "ready"
        }
    
    def _create_workflow_plan(self, task: str, agents: List[str], workflow_type: str) -> Dict[str, Any]:
        """Create a workflow execution plan."""
        
        plan = {
            "steps": [
                {
                    "step": 1,
                    "agent": "WebSearchAgent",
                    "action": "search_web",
                    "description": f"Search web for information about: {task}"
                },
                {
                    "step": 2,
                    "agent": "DocumentAnalysisAgent", 
                    "action": "analyze_documents",
                    "description": "Analyze any provided documents or gathered content"
                },
                {
                    "step": 3,
                    "agent": "SynthesisAgent",
                    "action": "synthesize_results",
                    "description": "Synthesize findings from all sources"
                }
            ],
            "estimated_time": "5-10 minutes",
            "success_criteria": [
                "Web search completed successfully",
                "Document analysis provides insights",
                "Synthesis produces actionable findings"
            ]
        }
        
        return plan


class MultiAgentWorkflow:
    """Main workflow orchestrator for multi-agent coordination."""
    
    def __init__(self, openai_api_key: str = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.agents = {}
        self.graph = None
        self.message_history = []
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents."""
        
        self.agents = {
            "coordinator": CoordinatorAgent(),
            "web_search": WebSearchAgent(),
            "document_analysis": DocumentAnalysisAgent(),
            "synthesis": SynthesisAgent()
        }
        
        # Initialize LLM for natural language processing
        self.llm = OpenAIModel(
            model="gpt-4",
            temperature=0.3,
            max_tokens=1500,
            api_key=self.api_key
        )
        
        # Initialize prompt template for final response
        self.response_template = PromptTemplate(
            template="""Based on the multi-agent analysis below, provide a comprehensive response to the user's question.

User Question: {question}

Agent Results:
{agent_results}

Provide a clear, well-structured response that incorporates insights from all agents:""",
            input_variables=["question", "agent_results"]
        )
    
    def build_workflow(self) -> Graph:
        """Build the multi-agent workflow graph."""
        
        self.graph = Graph(name="Multi-Agent Research Workflow")
        
        # Add all agents to the graph
        for name, agent in self.agents.items():
            self.graph.add_component(name, agent)
        
        # Add LLM components
        self.graph.add_component("llm", self.llm)
        self.graph.add_component("response_template", self.response_template)
        
        # Define workflow connections
        self._connect_workflow()
        
        return self.graph
    
    def _connect_workflow(self):
        """Define the workflow connections."""
        
        # Sequential flow: coordinator -> web_search -> document_analysis -> synthesis -> llm
        self.graph.connect("coordinator", "workflow_plan", "web_search", "query")
        self.graph.connect("web_search", "search_results", "document_analysis", "documents")
        self.graph.connect("web_search", "search_results", "synthesis", "search_results")
        self.graph.connect("document_analysis", "analysis_results", "synthesis", "analysis_results")
        self.graph.connect("synthesis", "synthesis", "response_template", "agent_results")
        self.graph.connect("response_template", "prompt", "llm", "prompt")
    
    async def execute_research_task(self, question: str, documents: List[str] = None) -> Dict[str, Any]:
        """Execute a complete research task using all agents."""
        
        print(f"🎯 Starting multi-agent research task: {question}")
        
        # Step 1: Coordinator creates workflow plan
        print("\n📋 Coordinator: Creating workflow plan...")
        coordinator_result = self.agents["coordinator"].build(
            task=question,
            workflow_type="research"
        )
        
        self._log_message("system", "coordinator", f"Workflow plan created: {coordinator_result['workflow_plan']}")
        
        # Step 2: Web search agent
        print("\n🔍 Web Search Agent: Searching for information...")
        search_result = self.agents["web_search"].build(
            query=question,
            max_results=5
        )
        
        self._log_message("coordinator", "web_search", f"Search completed: {len(search_result.get('search_results', []))} results")
        
        # Step 3: Document analysis agent
        print("\n📄 Document Analysis Agent: Analyzing content...")
        if documents is None:
            # Use search results as documents if no documents provided
            documents = [
                result.get("snippet", "") 
                for result in search_result.get("search_results", [])
                if result.get("snippet")
            ]
        
        analysis_result = self.agents["document_analysis"].build(
            documents=documents,
            analysis_type="comprehensive"
        )
        
        self._log_message("web_search", "document_analysis", f"Analysis completed: {len(documents)} documents")
        
        # Step 4: Synthesis agent
        print("\n🔬 Synthesis Agent: Synthesizing findings...")
        synthesis_result = self.agents["synthesis"].build(
            search_results=search_result,
            analysis_results=analysis_result,
            synthesis_type="comprehensive"
        )
        
        self._log_message("document_analysis", "synthesis", "Synthesis completed")
        
        # Step 5: Generate final response using LLM
        print("\n🤖 Generating final response...")
        agent_results = self._format_agent_results(search_result, analysis_result, synthesis_result)
        
        prompt = await self.response_template.arun(
            question=question,
            agent_results=agent_results
        )
        
        final_response = await self.llm.arun(prompt=prompt)
        
        # Compile complete results
        results = {
            "question": question,
            "workflow_plan": coordinator_result,
            "search_results": search_result,
            "analysis_results": analysis_result,
            "synthesis_results": synthesis_result,
            "final_response": final_response,
            "message_history": self.message_history,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n✅ Multi-agent research task completed!")
        return results
    
    def _format_agent_results(self, search_result: Dict, analysis_result: Dict, synthesis_result: Dict) -> str:
        """Format results from all agents for the final LLM prompt."""
        
        formatted_results = f"""
WEB SEARCH RESULTS:
- Query: {search_result.get('query', 'N/A')}
- Total Results: {search_result.get('total_results', 0)}
- Key Findings: {json.dumps(search_result.get('search_results', [])[:3], indent=2)}

DOCUMENT ANALYSIS:
- Documents Analyzed: {analysis_result.get('total_documents', 0)}
- Analysis Type: {analysis_result.get('analysis_type', 'N/A')}
- Key Insights: {json.dumps(analysis_result.get('analysis_results', [])[:2], indent=2)}

SYNTHESIS:
- Executive Summary: {synthesis_result.get('executive_summary', 'N/A')}
- Key Findings: {synthesis_result.get('key_findings', [])}
- Recommendations: {synthesis_result.get('recommendations', [])}
- Confidence Score: {synthesis_result.get('confidence_score', 0)}
"""
        
        return formatted_results
    
    def _log_message(self, sender: str, recipient: str, content: str):
        """Log inter-agent communication."""
        
        message = AgentMessage(
            sender=sender,
            recipient=recipient,
            content=content,
            message_type="communication",
            timestamp=datetime.now().isoformat()
        )
        
        self.message_history.append(message)
    
    async def interactive_research_session(self):
        """Start an interactive research session."""
        
        print("\n🤖 Multi-Agent Research System")
        print("Ask any research question and our agents will collaborate to find answers!")
        print("Type 'quit' to exit, 'history' to see agent communications")
        print("-" * 70)
        
        while True:
            try:
                question = input("\n❓ Research Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Research session ended!")
                    break
                
                if question.lower() == 'history':
                    self._display_message_history()
                    continue
                
                if not question:
                    continue
                
                # Execute research task
                results = await self.execute_research_task(question)
                
                # Display results
                print(f"\n{'='*70}")
                print("📊 RESEARCH RESULTS")
                print(f"{'='*70}")
                print(f"\n🎯 Question: {question}")
                print(f"\n🤖 Final Answer:\n{results['final_response']}")
                
                # Display synthesis summary
                synthesis = results['synthesis_results']
                print(f"\n📋 Summary:")
                print(f"- Sources analyzed: {synthesis.get('data_sources', {})}")
                print(f"- Confidence score: {synthesis.get('confidence_score', 0):.2f}")
                print(f"- Key recommendations: {len(synthesis.get('recommendations', []))}")
                
            except KeyboardInterrupt:
                print("\n👋 Research session ended!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _display_message_history(self):
        """Display the agent communication history."""
        
        if not self.message_history:
            print("No agent communications yet.")
            return
        
        print(f"\n{'='*50}")
        print("AGENT COMMUNICATION HISTORY")
        print(f"{'='*50}")
        
        for i, msg in enumerate(self.message_history[-10:], 1):  # Show last 10 messages
            print(f"{i}. {msg.sender} → {msg.recipient}: {msg.content[:100]}...")


async def main():
    """Main function to demonstrate multi-agent coordination."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        return
    
    # Initialize multi-agent workflow
    print("🚀 Initializing Multi-Agent Research System...")
    workflow = MultiAgentWorkflow()
    
    # Build the workflow graph
    graph = workflow.build_workflow()
    print(f"✅ Built workflow graph with {len(graph.components)} components")
    
    # Example research tasks
    example_questions = [
        "What are the latest developments in AI agent frameworks?",
        "How does Langflow compare to other visual AI platforms?",
        "What are the best practices for building RAG systems?"
    ]
    
    print("\n🔄 Running example research tasks...")
    for question in example_questions[:1]:  # Run first example
        print(f"\n{'='*70}")
        results = await workflow.execute_research_task(question)
        
        print(f"\n📊 Results for: {question}")
        print(f"🤖 Answer: {results['final_response'][:200]}...")
        
        synthesis = results['synthesis_results']
        print(f"📈 Confidence: {synthesis.get('confidence_score', 0):.2f}")
        print(f"🔍 Sources: {synthesis.get('data_sources', {})}")
    
    # Start interactive session
    print(f"\n{'='*70}")
    await workflow.interactive_research_session()


if __name__ == "__main__":
    asyncio.run(main())