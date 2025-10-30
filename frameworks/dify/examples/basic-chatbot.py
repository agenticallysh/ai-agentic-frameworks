#!/usr/bin/env python3
"""
Basic Chatbot Example - Dify
=============================

This example demonstrates how to create a simple conversational AI chatbot
using Dify's API. The chatbot can maintain conversation context and provide
intelligent responses using various LLM providers.

Requirements:
    pip install requests python-dotenv

Usage:
    python basic-chatbot.py
"""

import os
import json
import time
import uuid
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    message_id: str = None

    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())


class DifyChatbot:
    """
    A simple chatbot implementation using Dify's Chat API.
    
    This class provides a clean interface for creating conversational AI
    applications with Dify's platform capabilities.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.dify.ai/v1",
        app_id: str = None
    ):
        """
        Initialize the Dify chatbot.
        
        Args:
            api_key: Your Dify API key
            base_url: Dify API base URL (default: cloud endpoint)
            app_id: Your Dify application ID (if None, will be prompted)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.app_id = app_id
        self.conversation_id = None
        self.user_id = f"user_{uuid.uuid4().hex[:8]}"
        self.chat_history: List[ChatMessage] = []
        
        # HTTP session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        if not self.app_id:
            self.app_id = input("Enter your Dify App ID: ").strip()
    
    def create_chatbot_app(self, name: str = "Python Chatbot") -> Dict[str, Any]:
        """
        Create a new chatbot application in Dify.
        
        Args:
            name: Name for the new application
            
        Returns:
            Dictionary containing app creation response
        """
        
        app_config = {
            "name": name,
            "mode": "chat",
            "icon": "🤖",
            "description": "A simple chatbot created via Python API",
            "model_config": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            },
            "user_input_form": [
                {
                    "paragraph": {
                        "label": "Message",
                        "variable": "message",
                        "required": True,
                        "default": ""
                    }
                }
            ],
            "opening_statement": "Hello! I'm your AI assistant. How can I help you today?",
            "suggested_questions": [
                "What can you help me with?",
                "Tell me about artificial intelligence",
                "How do I get started with Dify?",
                "What are some best practices for AI development?"
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/apps",
                json=app_config,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            self.app_id = result.get('id')
            
            print(f"✅ Created chatbot app: {name}")
            print(f"   App ID: {self.app_id}")
            
            return result
            
        except requests.RequestException as e:
            print(f"❌ Error creating app: {e}")
            return {}
    
    def send_message(
        self, 
        message: str, 
        stream: bool = False,
        auto_generate_name: bool = True
    ) -> Dict[str, Any]:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            message: The user's message
            stream: Whether to use streaming response
            auto_generate_name: Whether to auto-generate conversation name
            
        Returns:
            Dictionary containing the response
        """
        
        if not self.app_id:
            raise ValueError("No app_id provided. Create an app first or provide app_id.")
        
        # Prepare the request payload
        payload = {
            "inputs": {"message": message},
            "query": message,
            "response_mode": "streaming" if stream else "blocking",
            "conversation_id": self.conversation_id,
            "user": self.user_id,
            "auto_generate_name": auto_generate_name
        }
        
        try:
            # Record user message
            user_msg = ChatMessage(
                role="user",
                content=message,
                timestamp=datetime.now()
            )
            self.chat_history.append(user_msg)
            
            # Send request to Dify
            response = self.session.post(
                f"{self.base_url}/chat-messages",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_blocking_response(response)
                
        except requests.RequestException as e:
            error_msg = f"API request failed: {e}"
            print(f"❌ {error_msg}")
            return {
                "answer": f"Sorry, I encountered an error: {error_msg}",
                "conversation_id": self.conversation_id,
                "message_id": str(uuid.uuid4())
            }
    
    def _handle_blocking_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle non-streaming response from Dify."""
        
        result = response.json()
        
        # Extract response information
        answer = result.get('answer', 'No response received')
        conversation_id = result.get('conversation_id')
        message_id = result.get('id')
        
        # Update conversation ID if this is the first message
        if conversation_id and not self.conversation_id:
            self.conversation_id = conversation_id
        
        # Record assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=answer,
            timestamp=datetime.now(),
            message_id=message_id
        )
        self.chat_history.append(assistant_msg)
        
        return {
            "answer": answer,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "metadata": result.get('metadata', {}),
            "usage": result.get('usage', {})
        }
    
    def _handle_streaming_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle streaming response from Dify."""
        
        answer_parts = []
        conversation_id = None
        message_id = None
        
        print("🤖 Assistant: ", end="", flush=True)
        
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        event = data.get('event')
                        
                        if event == 'message':
                            chunk = data.get('answer', '')
                            if chunk:
                                print(chunk, end="", flush=True)
                                answer_parts.append(chunk)
                        
                        elif event == 'message_end':
                            conversation_id = data.get('conversation_id')
                            message_id = data.get('id')
                            
                    except json.JSONDecodeError:
                        continue
            
            print()  # New line after streaming
            
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
        
        # Combine all answer parts
        answer = ''.join(answer_parts)
        
        # Update conversation ID
        if conversation_id and not self.conversation_id:
            self.conversation_id = conversation_id
        
        # Record assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=answer,
            timestamp=datetime.now(),
            message_id=message_id
        )
        self.chat_history.append(assistant_msg)
        
        return {
            "answer": answer,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "streaming": True
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation messages
        """
        
        if not self.conversation_id:
            return []
        
        try:
            response = self.session.get(
                f"{self.base_url}/messages",
                params={
                    "conversation_id": self.conversation_id,
                    "limit": 100
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
            
        except requests.RequestException as e:
            print(f"❌ Error fetching conversation history: {e}")
            return []
    
    def get_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get list of conversations for the current user.
        
        Args:
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversations
        """
        
        try:
            response = self.session.get(
                f"{self.base_url}/conversations",
                params={
                    "user": self.user_id,
                    "limit": limit
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
            
        except requests.RequestException as e:
            print(f"❌ Error fetching conversations: {e}")
            return []
    
    def rename_conversation(self, name: str) -> bool:
        """
        Rename the current conversation.
        
        Args:
            name: New name for the conversation
            
        Returns:
            True if successful, False otherwise
        """
        
        if not self.conversation_id:
            print("❌ No active conversation to rename")
            return False
        
        try:
            response = self.session.patch(
                f"{self.base_url}/conversations/{self.conversation_id}",
                json={"name": name},
                timeout=30
            )
            response.raise_for_status()
            
            print(f"✅ Conversation renamed to: {name}")
            return True
            
        except requests.RequestException as e:
            print(f"❌ Error renaming conversation: {e}")
            return False
    
    def clear_conversation(self):
        """Clear the current conversation context."""
        self.conversation_id = None
        self.chat_history.clear()
        print("🧹 Conversation cleared")
    
    def export_chat_history(self, filename: str = None) -> str:
        """
        Export chat history to a JSON file.
        
        Args:
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to the exported file
        """
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dify_chat_history_{timestamp}.json"
        
        # Convert chat history to serializable format
        history_data = []
        for msg in self.chat_history:
            history_data.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "message_id": msg.message_id
            })
        
        export_data = {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "app_id": self.app_id,
            "export_timestamp": datetime.now().isoformat(),
            "message_count": len(history_data),
            "messages": history_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Chat history exported to: {filename}")
        return filename
    
    def print_stats(self):
        """Print conversation statistics."""
        
        if not self.chat_history:
            print("📊 No conversation data available")
            return
        
        user_messages = [msg for msg in self.chat_history if msg.role == "user"]
        assistant_messages = [msg for msg in self.chat_history if msg.role == "assistant"]
        
        print("\n📊 Conversation Statistics:")
        print(f"   Total messages: {len(self.chat_history)}")
        print(f"   User messages: {len(user_messages)}")
        print(f"   Assistant messages: {len(assistant_messages)}")
        print(f"   Conversation ID: {self.conversation_id or 'None'}")
        print(f"   Duration: {self._get_conversation_duration()}")
    
    def _get_conversation_duration(self) -> str:
        """Get the duration of the current conversation."""
        
        if len(self.chat_history) < 2:
            return "N/A"
        
        start_time = self.chat_history[0].timestamp
        end_time = self.chat_history[-1].timestamp
        duration = end_time - start_time
        
        minutes = int(duration.total_seconds() // 60)
        seconds = int(duration.total_seconds() % 60)
        
        return f"{minutes}m {seconds}s"


def interactive_chat_session():
    """Run an interactive chat session with the Dify chatbot."""
    
    print("🤖 Dify Chatbot Interactive Session")
    print("=" * 50)
    
    # Get API credentials
    api_key = os.getenv('DIFY_API_KEY')
    if not api_key:
        api_key = input("Enter your Dify API key: ").strip()
    
    if not api_key:
        print("❌ API key is required")
        return
    
    # Initialize chatbot
    try:
        chatbot = DifyChatbot(api_key=api_key)
        print(f"✅ Chatbot initialized with App ID: {chatbot.app_id}")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return
    
    # Chat session
    print("\n💬 Chat Session Started")
    print("Commands: 'quit' to exit, 'clear' to start new conversation, 'stats' for statistics")
    print("          'export' to save history, 'stream' to toggle streaming mode")
    print("-" * 60)
    
    streaming_mode = False
    
    while True:
        try:
            user_input = input("\n💭 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'quit':
                print("\n👋 Thanks for chatting! Goodbye!")
                break
            
            elif user_input.lower() == 'clear':
                chatbot.clear_conversation()
                continue
            
            elif user_input.lower() == 'stats':
                chatbot.print_stats()
                continue
            
            elif user_input.lower() == 'export':
                chatbot.export_chat_history()
                continue
            
            elif user_input.lower() == 'stream':
                streaming_mode = not streaming_mode
                print(f"🔄 Streaming mode: {'ON' if streaming_mode else 'OFF'}")
                continue
            
            # Send message to chatbot
            print("🤖 Assistant: ", end="" if streaming_mode else "", flush=True)
            
            response = chatbot.send_message(user_input, stream=streaming_mode)
            
            if not streaming_mode:
                print(response['answer'])
            
            # Show usage info if available
            if 'usage' in response and response['usage']:
                usage = response['usage']
                if 'total_tokens' in usage:
                    print(f"   📊 Tokens used: {usage['total_tokens']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    # Final statistics
    chatbot.print_stats()


def demo_conversation():
    """Demonstrate basic chatbot functionality with predefined messages."""
    
    print("🤖 Dify Chatbot Demo")
    print("=" * 30)
    
    # Get API credentials
    api_key = os.getenv('DIFY_API_KEY')
    if not api_key:
        print("❌ DIFY_API_KEY environment variable not set")
        print("💡 Set it with: export DIFY_API_KEY='your_api_key_here'")
        return
    
    # Initialize chatbot
    chatbot = DifyChatbot(api_key=api_key)
    
    # Demo messages
    demo_messages = [
        "Hello! Can you introduce yourself?",
        "What can you help me with?",
        "Tell me about artificial intelligence and machine learning.",
        "How can I get started building AI applications?",
        "What are some best practices for developing with LLMs?"
    ]
    
    print(f"\n✅ Starting demo with {len(demo_messages)} messages...")
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n[{i}/{len(demo_messages)}] 💭 User: {message}")
        
        try:
            response = chatbot.send_message(message)
            print(f"🤖 Assistant: {response['answer']}")
            
            # Small delay between messages
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Show final statistics
    chatbot.print_stats()


def main():
    """Main function to run the chatbot example."""
    
    print("🤖 Dify Basic Chatbot Example")
    print("=" * 40)
    
    # Check for environment setup
    if not os.getenv('DIFY_API_KEY'):
        print("💡 Tip: Set DIFY_API_KEY environment variable for easier usage")
        print("   export DIFY_API_KEY='your_api_key_here'")
    
    # Choose mode
    print("\nChoose an option:")
    print("1. Interactive chat session")
    print("2. Demo conversation")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            interactive_chat_session()
        elif choice == '2':
            demo_conversation()
        elif choice == '3':
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()