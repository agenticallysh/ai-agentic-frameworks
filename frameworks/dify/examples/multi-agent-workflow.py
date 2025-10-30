#!/usr/bin/env python3
"""
Multi-Agent Workflow Example - Dify
====================================

This example demonstrates how to create coordinated multi-agent workflows
using Dify's workflow capabilities. Multiple specialized agents work together
to complete complex tasks requiring different expertise areas.

Requirements:
    pip install requests python-dotenv

Usage:
    python multi-agent-workflow.py
"""

import os
import json
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests


@dataclass
class WorkflowStep:
    """Represents a step in the multi-agent workflow."""
    step_id: str
    agent_name: str
    task: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    status: str = "pending"
    execution_time: float = 0.0
    error_message: str = None


class DifyMultiAgentWorkflow:
    """
    Multi-agent workflow orchestrator using Dify's workflow capabilities.
    
    This class manages the creation and execution of complex workflows that
    involve multiple specialized agents working in coordination.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.dify.ai/v1"
    ):
        """
        Initialize the multi-agent workflow system.
        
        Args:
            api_key: Your Dify API key
            base_url: Dify API base URL
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.workflow_apps: Dict[str, str] = {}  # agent_name -> app_id
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def create_research_agent(self) -> str:
        """Create a research specialist agent."""
        
        agent_config = {
            "name": "Research Agent",
            "mode": "workflow",
            "icon": "🔍",
            "description": "Specialized agent for research and information gathering",
            "workflow_config": {
                "type": "workflow",
                "nodes": [
                    {
                        "id": "start",
                        "type": "start",
                        "data": {
                            "title": "Research Request",
                            "variables": [
                                {
                                    "variable": "research_topic",
                                    "label": "Research Topic",
                                    "type": "text-input",
                                    "required": True
                                },
                                {
                                    "variable": "research_depth", 
                                    "label": "Research Depth",
                                    "type": "select",
                                    "options": ["basic", "detailed", "comprehensive"],
                                    "default": "detailed"
                                }
                            ]
                        }
                    },
                    {
                        "id": "llm_research",
                        "type": "llm",
                        "data": {
                            "title": "Research Analysis",
                            "model": {
                                "provider": "openai",
                                "name": "gpt-4",
                                "parameters": {
                                    "temperature": 0.3,
                                    "max_tokens": 2000
                                }
                            },
                            "prompt_template": [
                                {
                                    "role": "system",
                                    "text": """You are a research specialist AI agent. Your role is to conduct thorough research on given topics and provide comprehensive, accurate information.

Research Guidelines:
1. Provide factual, well-structured information
2. Include key concepts, definitions, and context
3. Identify important trends and developments
4. Suggest areas for further investigation
5. Cite reliable sources when possible

Research Depth Levels:
- Basic: Overview and key points
- Detailed: Comprehensive analysis with examples
- Comprehensive: In-depth analysis with multiple perspectives"""
                                },
                                {
                                    "role": "user", 
                                    "text": "Research Topic: {{#start.research_topic#}}\nResearch Depth: {{#start.research_depth#}}\n\nPlease conduct research on this topic according to the specified depth level."
                                }
                            ]
                        }
                    },
                    {
                        "id": "end",
                        "type": "end",
                        "data": {
                            "title": "Research Results",
                            "outputs": [
                                {
                                    "variable": "research_findings",
                                    "type": "text",
                                    "value_selector": ["llm_research", "text"]
                                }
                            ]
                        }
                    }
                ],
                "edges": [
                    {
                        "id": "start-llm_research",
                        "source": "start",
                        "target": "llm_research"
                    },
                    {
                        "id": "llm_research-end",
                        "source": "llm_research", 
                        "target": "end"
                    }
                ]
            }
        }
        
        return self._create_agent_app("research_agent", agent_config)
    
    def create_analysis_agent(self) -> str:
        """Create an analysis specialist agent."""
        
        agent_config = {
            "name": "Analysis Agent",
            "mode": "workflow",
            "icon": "📊",
            "description": "Specialized agent for data analysis and insights",
            "workflow_config": {
                "type": "workflow",
                "nodes": [
                    {
                        "id": "start",
                        "type": "start",
                        "data": {
                            "title": "Analysis Request",
                            "variables": [
                                {
                                    "variable": "data_input",
                                    "label": "Data to Analyze",
                                    "type": "paragraph",
                                    "required": True
                                },
                                {
                                    "variable": "analysis_type",
                                    "label": "Analysis Type",
                                    "type": "select",
                                    "options": ["trend_analysis", "comparative_analysis", "risk_assessment", "opportunity_identification"],
                                    "default": "trend_analysis"
                                }
                            ]
                        }
                    },
                    {
                        "id": "llm_analysis",
                        "type": "llm",
                        "data": {
                            "title": "Data Analysis",
                            "model": {
                                "provider": "openai",
                                "name": "gpt-4",
                                "parameters": {
                                    "temperature": 0.2,
                                    "max_tokens": 1500
                                }
                            },
                            "prompt_template": [
                                {
                                    "role": "system",
                                    "text": """You are a data analysis specialist AI agent. Your role is to analyze information and extract meaningful insights, patterns, and recommendations.

Analysis Capabilities:
- Trend Analysis: Identify patterns and trends in data
- Comparative Analysis: Compare different options or scenarios
- Risk Assessment: Identify potential risks and mitigation strategies
- Opportunity Identification: Find opportunities and potential benefits

Provide structured analysis with:
1. Key findings and patterns
2. Supporting evidence
3. Implications and significance
4. Actionable recommendations
5. Confidence levels for conclusions"""
                                },
                                {
                                    "role": "user",
                                    "text": "Analysis Type: {{#start.analysis_type#}}\n\nData to Analyze:\n{{#start.data_input#}}\n\nPlease conduct a thorough analysis based on the specified type."
                                }
                            ]
                        }
                    },
                    {
                        "id": "end",
                        "type": "end",
                        "data": {
                            "title": "Analysis Results",
                            "outputs": [
                                {
                                    "variable": "analysis_results",
                                    "type": "text",
                                    "value_selector": ["llm_analysis", "text"]
                                }
                            ]
                        }
                    }
                ],
                "edges": [
                    {
                        "id": "start-llm_analysis",
                        "source": "start",
                        "target": "llm_analysis"
                    },
                    {
                        "id": "llm_analysis-end",
                        "source": "llm_analysis",
                        "target": "end"
                    }
                ]
            }
        }
        
        return self._create_agent_app("analysis_agent", agent_config)
    
    def create_synthesis_agent(self) -> str:
        """Create a synthesis specialist agent."""
        
        agent_config = {
            "name": "Synthesis Agent",
            "mode": "workflow", 
            "icon": "⚗️",
            "description": "Specialized agent for synthesizing information from multiple sources",
            "workflow_config": {
                "type": "workflow",
                "nodes": [
                    {
                        "id": "start",
                        "type": "start",
                        "data": {
                            "title": "Synthesis Request",
                            "variables": [
                                {
                                    "variable": "research_findings",
                                    "label": "Research Findings",
                                    "type": "paragraph",
                                    "required": True
                                },
                                {
                                    "variable": "analysis_results",
                                    "label": "Analysis Results", 
                                    "type": "paragraph",
                                    "required": True
                                },
                                {
                                    "variable": "synthesis_goal",
                                    "label": "Synthesis Goal",
                                    "type": "text-input",
                                    "required": True
                                }
                            ]
                        }
                    },
                    {
                        "id": "llm_synthesis",
                        "type": "llm",
                        "data": {
                            "title": "Information Synthesis",
                            "model": {
                                "provider": "openai",
                                "name": "gpt-4",
                                "parameters": {
                                    "temperature": 0.4,
                                    "max_tokens": 2500
                                }
                            },
                            "prompt_template": [
                                {
                                    "role": "system",
                                    "text": """You are a synthesis specialist AI agent. Your role is to combine information from multiple sources into coherent, actionable insights and recommendations.

Synthesis Process:
1. Integrate findings from research and analysis
2. Identify connections and relationships
3. Resolve conflicts or contradictions
4. Create unified understanding
5. Generate actionable recommendations

Output Structure:
- Executive Summary
- Key Insights (synthesis of all inputs)
- Interconnections and Dependencies
- Strategic Recommendations
- Implementation Considerations
- Risk Factors and Mitigation
- Success Metrics"""
                                },
                                {
                                    "role": "user",
                                    "text": "Synthesis Goal: {{#start.synthesis_goal#}}\n\nResearch Findings:\n{{#start.research_findings#}}\n\nAnalysis Results:\n{{#start.analysis_results#}}\n\nPlease synthesize this information into a comprehensive report."
                                }
                            ]
                        }
                    },
                    {
                        "id": "end",
                        "type": "end",
                        "data": {
                            "title": "Synthesis Report",
                            "outputs": [
                                {
                                    "variable": "synthesis_report",
                                    "type": "text",
                                    "value_selector": ["llm_synthesis", "text"]
                                }
                            ]
                        }
                    }
                ],
                "edges": [
                    {
                        "id": "start-llm_synthesis",
                        "source": "start",
                        "target": "llm_synthesis"
                    },
                    {
                        "id": "llm_synthesis-end",
                        "source": "llm_synthesis",
                        "target": "end"
                    }
                ]
            }
        }
        
        return self._create_agent_app("synthesis_agent", agent_config)
    
    def create_coordinator_workflow(self) -> str:
        """Create the main coordinator workflow that orchestrates all agents."""
        
        workflow_config = {
            "name": "Multi-Agent Research Coordinator",
            "mode": "workflow",
            "icon": "🎯",
            "description": "Coordinates multi-agent research and analysis workflow",
            "workflow_config": {
                "type": "workflow",
                "nodes": [
                    {
                        "id": "start",
                        "type": "start",
                        "data": {
                            "title": "Research Project Request",
                            "variables": [
                                {
                                    "variable": "project_topic",
                                    "label": "Research Topic",
                                    "type": "text-input",
                                    "required": True
                                },
                                {
                                    "variable": "project_scope",
                                    "label": "Project Scope",
                                    "type": "select",
                                    "options": ["market_analysis", "competitive_analysis", "technology_assessment", "strategic_planning"],
                                    "default": "market_analysis"
                                },
                                {
                                    "variable": "urgency_level",
                                    "label": "Urgency Level",
                                    "type": "select",
                                    "options": ["low", "medium", "high"],
                                    "default": "medium"
                                }
                            ]
                        }
                    },
                    {
                        "id": "research_step",
                        "type": "http-request",
                        "data": {
                            "title": "Execute Research Agent",
                            "method": "POST",
                            "url": f"{self.base_url}/workflows/run",
                            "headers": {
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            "body": {
                                "inputs": {
                                    "research_topic": "{{#start.project_topic#}}",
                                    "research_depth": "comprehensive"
                                },
                                "response_mode": "blocking",
                                "user": "coordinator_workflow"
                            }
                        }
                    },
                    {
                        "id": "analysis_step",
                        "type": "http-request", 
                        "data": {
                            "title": "Execute Analysis Agent",
                            "method": "POST",
                            "url": f"{self.base_url}/workflows/run",
                            "headers": {
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            "body": {
                                "inputs": {
                                    "data_input": "{{#research_step.research_findings#}}",
                                    "analysis_type": "trend_analysis"
                                },
                                "response_mode": "blocking",
                                "user": "coordinator_workflow"
                            }
                        }
                    },
                    {
                        "id": "synthesis_step",
                        "type": "http-request",
                        "data": {
                            "title": "Execute Synthesis Agent",
                            "method": "POST",
                            "url": f"{self.base_url}/workflows/run",
                            "headers": {
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            "body": {
                                "inputs": {
                                    "research_findings": "{{#research_step.research_findings#}}",
                                    "analysis_results": "{{#analysis_step.analysis_results#}}",
                                    "synthesis_goal": "Create comprehensive strategic recommendations for {{#start.project_topic#}}"
                                },
                                "response_mode": "blocking",
                                "user": "coordinator_workflow"
                            }
                        }
                    },
                    {
                        "id": "end",
                        "type": "end",
                        "data": {
                            "title": "Multi-Agent Report",
                            "outputs": [
                                {
                                    "variable": "final_report",
                                    "type": "text",
                                    "value_selector": ["synthesis_step", "synthesis_report"]
                                },
                                {
                                    "variable": "research_data",
                                    "type": "text",
                                    "value_selector": ["research_step", "research_findings"]
                                },
                                {
                                    "variable": "analysis_data",
                                    "type": "text",
                                    "value_selector": ["analysis_step", "analysis_results"]
                                }
                            ]
                        }
                    }
                ],
                "edges": [
                    {
                        "id": "start-research_step",
                        "source": "start",
                        "target": "research_step"
                    },
                    {
                        "id": "research_step-analysis_step",
                        "source": "research_step",
                        "target": "analysis_step"
                    },
                    {
                        "id": "analysis_step-synthesis_step",
                        "source": "analysis_step",
                        "target": "synthesis_step"
                    },
                    {
                        "id": "synthesis_step-end",
                        "source": "synthesis_step",
                        "target": "end"
                    }
                ]
            }
        }
        
        return self._create_agent_app("coordinator", workflow_config)
    
    def _create_agent_app(self, agent_name: str, config: Dict[str, Any]) -> str:
        """Create an agent application in Dify."""
        
        try:
            response = self.session.post(
                f"{self.base_url}/apps",
                json=config,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            app_id = result.get('id')
            
            if app_id:
                self.workflow_apps[agent_name] = app_id
                print(f"✅ Created {agent_name}: {config['name']}")
                print(f"   App ID: {app_id}")
            
            return app_id
            
        except requests.RequestException as e:
            print(f"❌ Error creating {agent_name}: {e}")
            return None
    
    def execute_workflow_step(
        self,
        app_id: str,
        inputs: Dict[str, Any],
        user_id: str = None
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        
        if not user_id:
            user_id = f"workflow_user_{uuid.uuid4().hex[:8]}"
        
        payload = {
            "inputs": inputs,
            "response_mode": "blocking",
            "user": user_id
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/workflows/run",
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                "success": True,
                "outputs": result.get('data', {}).get('outputs', {}),
                "workflow_id": result.get('workflow_run_id'),
                "usage": result.get('data', {}).get('total_tokens', 0)
            }
            
        except requests.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "outputs": {},
                "workflow_id": None,
                "usage": 0
            }
    
    def run_multi_agent_workflow(
        self,
        project_topic: str,
        project_scope: str = "market_analysis",
        urgency_level: str = "medium"
    ) -> Dict[str, Any]:
        """Execute the complete multi-agent workflow."""
        
        print(f"🚀 Starting multi-agent workflow for: {project_topic}")
        print(f"   Scope: {project_scope}")
        print(f"   Urgency: {urgency_level}")
        
        workflow_steps = []
        start_time = time.time()
        
        # Step 1: Research Agent
        print("\n🔍 Step 1: Research Agent")
        research_inputs = {
            "research_topic": project_topic,
            "research_depth": "comprehensive"
        }
        
        research_step = WorkflowStep(
            step_id="research",
            agent_name="Research Agent",
            task="Conduct comprehensive research",
            inputs=research_inputs,
            outputs={}
        )
        
        step_start = time.time()
        research_result = self.execute_workflow_step(
            self.workflow_apps.get("research_agent"),
            research_inputs
        )
        research_step.execution_time = time.time() - step_start
        
        if research_result["success"]:
            research_step.status = "completed"
            research_step.outputs = research_result["outputs"]
            research_findings = research_result["outputs"].get("research_findings", "")
            print(f"   ✅ Research completed in {research_step.execution_time:.1f}s")
        else:
            research_step.status = "failed"
            research_step.error_message = research_result["error"]
            print(f"   ❌ Research failed: {research_result['error']}")
            return {"success": False, "steps": [research_step]}
        
        workflow_steps.append(research_step)
        
        # Step 2: Analysis Agent
        print("\n📊 Step 2: Analysis Agent")
        analysis_inputs = {
            "data_input": research_findings,
            "analysis_type": "trend_analysis"
        }
        
        analysis_step = WorkflowStep(
            step_id="analysis",
            agent_name="Analysis Agent", 
            task="Analyze research findings",
            inputs=analysis_inputs,
            outputs={}
        )
        
        step_start = time.time()
        analysis_result = self.execute_workflow_step(
            self.workflow_apps.get("analysis_agent"),
            analysis_inputs
        )
        analysis_step.execution_time = time.time() - step_start
        
        if analysis_result["success"]:
            analysis_step.status = "completed"
            analysis_step.outputs = analysis_result["outputs"]
            analysis_results = analysis_result["outputs"].get("analysis_results", "")
            print(f"   ✅ Analysis completed in {analysis_step.execution_time:.1f}s")
        else:
            analysis_step.status = "failed"
            analysis_step.error_message = analysis_result["error"]
            print(f"   ❌ Analysis failed: {analysis_result['error']}")
            return {"success": False, "steps": workflow_steps}
        
        workflow_steps.append(analysis_step)
        
        # Step 3: Synthesis Agent
        print("\n⚗️ Step 3: Synthesis Agent")
        synthesis_inputs = {
            "research_findings": research_findings,
            "analysis_results": analysis_results,
            "synthesis_goal": f"Create comprehensive strategic recommendations for {project_topic}"
        }
        
        synthesis_step = WorkflowStep(
            step_id="synthesis",
            agent_name="Synthesis Agent",
            task="Synthesize findings into recommendations",
            inputs=synthesis_inputs,
            outputs={}
        )
        
        step_start = time.time()
        synthesis_result = self.execute_workflow_step(
            self.workflow_apps.get("synthesis_agent"),
            synthesis_inputs
        )
        synthesis_step.execution_time = time.time() - step_start
        
        if synthesis_result["success"]:
            synthesis_step.status = "completed"
            synthesis_step.outputs = synthesis_result["outputs"]
            print(f"   ✅ Synthesis completed in {synthesis_step.execution_time:.1f}s")
        else:
            synthesis_step.status = "failed"
            synthesis_step.error_message = synthesis_result["error"]
            print(f"   ❌ Synthesis failed: {synthesis_result['error']}")
            return {"success": False, "steps": workflow_steps}
        
        workflow_steps.append(synthesis_step)
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Multi-agent workflow completed in {total_time:.1f}s")
        
        return {
            "success": True,
            "total_execution_time": total_time,
            "steps": workflow_steps,
            "final_report": synthesis_step.outputs.get("synthesis_report", ""),
            "metadata": {
                "project_topic": project_topic,
                "project_scope": project_scope,
                "urgency_level": urgency_level,
                "total_tokens": sum(step.outputs.get("usage", 0) for step in workflow_steps)
            }
        }
    
    def setup_workflow_system(self) -> bool:
        """Set up the complete multi-agent workflow system."""
        
        print("🏗️ Setting up multi-agent workflow system...")
        
        # Create all agent applications
        agents_created = []
        
        print("\n📝 Creating specialized agents...")
        
        research_app_id = self.create_research_agent()
        if research_app_id:
            agents_created.append("research_agent")
        
        analysis_app_id = self.create_analysis_agent()
        if analysis_app_id:
            agents_created.append("analysis_agent")
        
        synthesis_app_id = self.create_synthesis_agent()
        if synthesis_app_id:
            agents_created.append("synthesis_agent")
        
        if len(agents_created) == 3:
            print("\n✅ All specialized agents created successfully!")
            print("   🔍 Research Agent - For information gathering")
            print("   📊 Analysis Agent - For data analysis") 
            print("   ⚗️ Synthesis Agent - For report generation")
            return True
        else:
            print(f"\n❌ Failed to create all agents. Created: {agents_created}")
            return False
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get the status of all workflow components."""
        
        status = {
            "agents_created": len(self.workflow_apps),
            "agent_details": {},
            "system_ready": len(self.workflow_apps) >= 3
        }
        
        for agent_name, app_id in self.workflow_apps.items():
            status["agent_details"][agent_name] = {
                "app_id": app_id,
                "status": "ready"
            }
        
        return status


def run_workflow_demo():
    """Run a demonstration of the multi-agent workflow."""
    
    print("🤖 Dify Multi-Agent Workflow Demo")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('DIFY_API_KEY')
    if not api_key:
        api_key = input("Enter your Dify API key: ").strip()
    
    if not api_key:
        print("❌ API key is required")
        return
    
    # Initialize workflow system
    workflow = DifyMultiAgentWorkflow(api_key=api_key)
    
    # Set up the workflow system
    setup_success = workflow.setup_workflow_system()
    if not setup_success:
        print("❌ Failed to set up workflow system")
        return
    
    # Demo projects
    demo_projects = [
        {
            "topic": "AI-powered customer service automation",
            "scope": "technology_assessment",
            "urgency": "high"
        },
        {
            "topic": "Sustainable energy solutions for small businesses",
            "scope": "market_analysis", 
            "urgency": "medium"
        }
    ]
    
    for i, project in enumerate(demo_projects, 1):
        print(f"\n{'='*60}")
        print(f"🎯 Demo Project {i}: {project['topic']}")
        print(f"{'='*60}")
        
        result = workflow.run_multi_agent_workflow(
            project_topic=project["topic"],
            project_scope=project["scope"],
            urgency_level=project["urgency"]
        )
        
        if result["success"]:
            print(f"\n📋 Final Report:")
            print("-" * 40)
            print(result["final_report"])
            
            print(f"\n📊 Workflow Statistics:")
            print(f"   Total time: {result['total_execution_time']:.1f}s")
            print(f"   Steps completed: {len(result['steps'])}")
            print(f"   Total tokens: {result['metadata']['total_tokens']}")
            
        else:
            print(f"\n❌ Workflow failed")
            for step in result["steps"]:
                if step.status == "failed":
                    print(f"   Failed step: {step.agent_name} - {step.error_message}")
        
        # Small delay between projects
        if i < len(demo_projects):
            time.sleep(3)
    
    # Show system status
    print(f"\n📊 Workflow System Status:")
    status = workflow.get_workflow_status()
    print(f"   Agents created: {status['agents_created']}")
    print(f"   System ready: {status['system_ready']}")
    
    for agent_name, details in status["agent_details"].items():
        print(f"   {agent_name}: {details['status']} (ID: {details['app_id']})")


def main():
    """Main function to run the multi-agent workflow example."""
    
    print("🤖 Dify Multi-Agent Workflow Example")
    print("=" * 45)
    
    # Check for environment setup
    if not os.getenv('DIFY_API_KEY'):
        print("💡 Tip: Set DIFY_API_KEY environment variable for easier usage")
        print("   export DIFY_API_KEY='your_api_key_here'")
    
    print("\nThis example demonstrates:")
    print("• Research Agent - Gathers comprehensive information")
    print("• Analysis Agent - Analyzes data and identifies trends")
    print("• Synthesis Agent - Creates actionable recommendations")
    print("• Coordinator - Orchestrates the entire workflow")
    
    try:
        run_workflow_demo()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()