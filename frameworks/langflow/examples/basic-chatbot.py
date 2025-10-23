#!/usr/bin/env python3
"""
Basic Chatbot Example - Langflow
=================================

This example demonstrates how to create a simple chatbot using Langflow's API.
The chatbot uses OpenAI's GPT model and can maintain conversation context.

Prerequisites:
- langflow installed: pip install langflow
- OpenAI API key set in environment: OPENAI_API_KEY
- Langflow server running: langflow run

Usage:
    python basic-chatbot.py
"""

import os
import json
import requests
import asyncio
from typing import Dict, Any

class LangflowChatbot:
    """Simple chatbot using Langflow API"""
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.session_id = "demo_session"
        
        # Verify API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    def create_flow_json(self) -> Dict[str, Any]:
        """Create a basic chatbot flow configuration"""
        flow_config = {
            "data": {
                "nodes": [
                    {
                        "id": "ChatInput-1",
                        "type": "ChatInput",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "node": {
                                "template": {
                                    "sender": {"value": "User"},
                                    "sender_name": {"value": "User"},
                                    "message": {"value": ""},
                                    "session_id": {"value": self.session_id}
                                }
                            }
                        }
                    },
                    {
                        "id": "OpenAI-1", 
                        "type": "OpenAIModel",
                        "position": {"x": 400, "y": 100},
                        "data": {
                            "node": {
                                "template": {
                                    "model_name": {"value": "gpt-3.5-turbo"},
                                    "openai_api_key": {"value": os.getenv("OPENAI_API_KEY")},
                                    "temperature": {"value": 0.7},
                                    "max_tokens": {"value": 1000}
                                }
                            }
                        }
                    },
                    {
                        "id": "ChatOutput-1",
                        "type": "ChatOutput", 
                        "position": {"x": 700, "y": 100},
                        "data": {
                            "node": {
                                "template": {
                                    "sender": {"value": "AI"},
                                    "sender_name": {"value": "Assistant"},
                                    "session_id": {"value": self.session_id}
                                }
                            }
                        }
                    }
                ],
                "edges": [
                    {
                        "id": "edge-1",
                        "source": "ChatInput-1",
                        "target": "OpenAI-1",
                        "sourceHandle": "message",
                        "targetHandle": "input"
                    },
                    {
                        "id": "edge-2", 
                        "source": "OpenAI-1",
                        "target": "ChatOutput-1",
                        "sourceHandle": "output",
                        "targetHandle": "message"
                    }
                ]
            }
        }
        return flow_config
    
    def send_message(self, message: str) -> str:
        """Send message to chatbot and get response"""
        try:
            # Create the flow
            flow_config = self.create_flow_json()
            
            # Send request to Langflow API
            response = requests.post(
                f"{self.base_url}/api/v1/run/flow",
                json={
                    "flow": flow_config,
                    "inputs": {
                        "message": message,
                        "session_id": self.session_id
                    },
                    "tweaks": {}
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("outputs", {}).get("message", "No response")
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Langflow server. Make sure it's running with 'langflow run'"
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    """Main chat loop"""
    print("🤖 Langflow Chatbot")
    print("=" * 50)
    print("Type 'quit' to exit")
    print()
    
    try:
        # Initialize chatbot
        chatbot = LangflowChatbot()
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Get chatbot response
            print("🤖 Thinking...")
            response = chatbot.send_message(user_input)
            print(f"Bot: {response}")
            print()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()