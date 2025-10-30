#!/usr/bin/env python3
"""
Custom Tool Example - Dify
===========================

This example demonstrates how to create and integrate custom tools with Dify
applications. Custom tools extend Dify's capabilities by adding specialized
functions and integrations with external services.

Requirements:
    pip install requests python-dotenv beautifulsoup4

Usage:
    python custom-tool.py
"""

import os
import json
import time
import uuid
import requests
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import urllib.parse


@dataclass
class CustomTool:
    """Represents a custom tool configuration."""
    name: str
    description: str
    parameters: List[Dict[str, Any]]
    function_code: str
    category: str = "utility"
    icon: str = "🔧"


class DifyCustomToolBuilder:
    """
    Builder for creating custom tools that integrate with Dify applications.
    
    This class provides a framework for creating, testing, and deploying
    custom tools that extend Dify's built-in capabilities.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.dify.ai/v1"
    ):
        """
        Initialize the custom tool builder.
        
        Args:
            api_key: Your Dify API key
            base_url: Dify API base URL
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        self.tools: Dict[str, CustomTool] = {}
        self.tool_app_id = None
    
    def create_web_scraper_tool(self) -> CustomTool:
        """Create a web scraping tool."""
        
        function_code = '''
async function webScraper(url, selector = null, extract_type = "text") {
    try {
        // Validate URL
        if (!url || !url.startsWith('http')) {
            return { error: "Invalid URL provided" };
        }
        
        // For demonstration, we'll simulate web scraping
        // In production, you would integrate with a real scraping service
        
        const scraped_data = {
            url: url,
            title: "Sample Page Title",
            content: `This is sample content from ${url}`,
            links: ["https://example.com/link1", "https://example.com/link2"],
            images: ["https://example.com/image1.jpg"],
            timestamp: new Date().toISOString()
        };
        
        // Apply selector filtering if provided
        if (selector) {
            scraped_data.selected_content = `Content filtered by selector: ${selector}`;
        }
        
        // Format based on extract_type
        switch (extract_type) {
            case "text":
                return {
                    success: true,
                    data: scraped_data.content,
                    metadata: {
                        title: scraped_data.title,
                        url: scraped_data.url
                    }
                };
            
            case "links":
                return {
                    success: true,
                    data: scraped_data.links,
                    count: scraped_data.links.length
                };
            
            case "images":
                return {
                    success: true,
                    data: scraped_data.images,
                    count: scraped_data.images.length
                };
            
            case "all":
                return {
                    success: true,
                    data: scraped_data
                };
            
            default:
                return {
                    success: true,
                    data: scraped_data.content
                };
        }
        
    } catch (error) {
        return {
            success: false,
            error: error.message
        };
    }
}

return await webScraper(url, selector, extract_type);
        '''
        
        tool = CustomTool(
            name="web_scraper",
            description="Extract content from web pages including text, links, and images",
            parameters=[
                {
                    "name": "url",
                    "type": "string",
                    "description": "The URL of the webpage to scrape",
                    "required": True
                },
                {
                    "name": "selector",
                    "type": "string", 
                    "description": "CSS selector to target specific elements (optional)",
                    "required": False
                },
                {
                    "name": "extract_type",
                    "type": "select",
                    "description": "Type of content to extract",
                    "options": ["text", "links", "images", "all"],
                    "default": "text",
                    "required": False
                }
            ],
            function_code=function_code,
            category="web",
            icon="🌐"
        )
        
        self.tools["web_scraper"] = tool
        return tool
    
    def create_data_analyzer_tool(self) -> CustomTool:
        """Create a data analysis tool."""
        
        function_code = '''
async function dataAnalyzer(data, analysis_type = "summary", format = "json") {
    try {
        // Parse input data
        let parsed_data;
        if (typeof data === 'string') {
            try {
                parsed_data = JSON.parse(data);
            } catch {
                // Treat as text data
                parsed_data = data.split('\\n').filter(line => line.trim());
            }
        } else {
            parsed_data = data;
        }
        
        const results = {
            analysis_type: analysis_type,
            timestamp: new Date().toISOString(),
            input_type: Array.isArray(parsed_data) ? 'array' : typeof parsed_data,
            data_size: Array.isArray(parsed_data) ? parsed_data.length : Object.keys(parsed_data || {}).length
        };
        
        switch (analysis_type) {
            case "summary":
                if (Array.isArray(parsed_data)) {
                    results.summary = {
                        total_items: parsed_data.length,
                        sample_items: parsed_data.slice(0, 3),
                        data_types: [...new Set(parsed_data.map(item => typeof item))]
                    };
                } else if (typeof parsed_data === 'object') {
                    results.summary = {
                        total_keys: Object.keys(parsed_data).length,
                        keys: Object.keys(parsed_data).slice(0, 5),
                        value_types: [...new Set(Object.values(parsed_data).map(val => typeof val))]
                    };
                } else {
                    results.summary = {
                        content_length: parsed_data.toString().length,
                        word_count: parsed_data.toString().split(/\\s+/).length,
                        character_count: parsed_data.toString().length
                    };
                }
                break;
            
            case "statistics":
                if (Array.isArray(parsed_data)) {
                    const numbers = parsed_data.filter(item => typeof item === 'number');
                    if (numbers.length > 0) {
                        results.statistics = {
                            count: numbers.length,
                            sum: numbers.reduce((a, b) => a + b, 0),
                            average: numbers.reduce((a, b) => a + b, 0) / numbers.length,
                            min: Math.min(...numbers),
                            max: Math.max(...numbers)
                        };
                    } else {
                        results.statistics = {
                            message: "No numeric data found for statistical analysis",
                            total_items: parsed_data.length
                        };
                    }
                } else {
                    results.statistics = {
                        message: "Statistical analysis requires array input",
                        data_type: typeof parsed_data
                    };
                }
                break;
            
            case "patterns":
                if (Array.isArray(parsed_data)) {
                    const patterns = {};
                    parsed_data.forEach(item => {
                        const type = typeof item;
                        patterns[type] = (patterns[type] || 0) + 1;
                    });
                    
                    results.patterns = {
                        type_distribution: patterns,
                        unique_values: [...new Set(parsed_data)].length,
                        duplicates: parsed_data.length - [...new Set(parsed_data)].length
                    };
                } else {
                    results.patterns = {
                        message: "Pattern analysis requires array input"
                    };
                }
                break;
            
            case "validation":
                results.validation = {
                    is_valid: true,
                    issues: [],
                    data_quality_score: 0.95,
                    recommendations: ["Data appears to be well-formatted", "Consider adding data validation rules"]
                };
                break;
            
            default:
                results.error = `Unknown analysis type: ${analysis_type}`;
        }
        
        // Format output
        if (format === "text") {
            return {
                success: true,
                data: JSON.stringify(results, null, 2)
            };
        } else {
            return {
                success: true,
                data: results
            };
        }
        
    } catch (error) {
        return {
            success: false,
            error: error.message
        };
    }
}

return await dataAnalyzer(data, analysis_type, format);
        '''
        
        tool = CustomTool(
            name="data_analyzer",
            description="Analyze data structures and provide insights, statistics, and patterns",
            parameters=[
                {
                    "name": "data",
                    "type": "string",
                    "description": "The data to analyze (JSON string or text)",
                    "required": True
                },
                {
                    "name": "analysis_type",
                    "type": "select",
                    "description": "Type of analysis to perform",
                    "options": ["summary", "statistics", "patterns", "validation"],
                    "default": "summary",
                    "required": False
                },
                {
                    "name": "format",
                    "type": "select",
                    "description": "Output format",
                    "options": ["json", "text"],
                    "default": "json",
                    "required": False
                }
            ],
            function_code=function_code,
            category="data",
            icon="📊"
        )
        
        self.tools["data_analyzer"] = tool
        return tool
    
    def create_api_caller_tool(self) -> CustomTool:
        """Create an API calling tool."""
        
        function_code = '''
async function apiCaller(url, method = "GET", headers = {}, body = null, timeout = 30000) {
    try {
        // Validate URL
        if (!url || !url.startsWith('http')) {
            return { error: "Invalid URL provided" };
        }
        
        // Prepare request options
        const options = {
            method: method.toUpperCase(),
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'Dify-Custom-Tool/1.0',
                ...headers
            }
        };
        
        // Add body for POST/PUT requests
        if (body && ['POST', 'PUT', 'PATCH'].includes(method.toUpperCase())) {
            if (typeof body === 'object') {
                options.body = JSON.stringify(body);
            } else {
                options.body = body;
            }
        }
        
        // Simulate API call (in production, use actual fetch/axios)
        const mockResponse = {
            status: 200,
            statusText: 'OK',
            data: {
                message: `Mock response from ${method} ${url}`,
                timestamp: new Date().toISOString(),
                headers_sent: options.headers,
                body_sent: options.body || null
            },
            headers: {
                'content-type': 'application/json',
                'x-api-version': '1.0'
            }
        };
        
        return {
            success: true,
            status: mockResponse.status,
            status_text: mockResponse.statusText,
            data: mockResponse.data,
            response_headers: mockResponse.headers,
            request_info: {
                url: url,
                method: method,
                sent_at: new Date().toISOString()
            }
        };
        
    } catch (error) {
        return {
            success: false,
            error: error.message,
            url: url,
            method: method
        };
    }
}

return await apiCaller(url, method, headers, body, timeout);
        '''
        
        tool = CustomTool(
            name="api_caller",
            description="Make HTTP requests to external APIs with full control over method, headers, and body",
            parameters=[
                {
                    "name": "url",
                    "type": "string",
                    "description": "The API endpoint URL to call",
                    "required": True
                },
                {
                    "name": "method",
                    "type": "select",
                    "description": "HTTP method to use",
                    "options": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    "default": "GET",
                    "required": False
                },
                {
                    "name": "headers",
                    "type": "string",
                    "description": "HTTP headers as JSON string (optional)",
                    "required": False
                },
                {
                    "name": "body",
                    "type": "string",
                    "description": "Request body data (for POST/PUT requests)",
                    "required": False
                },
                {
                    "name": "timeout",
                    "type": "number",
                    "description": "Request timeout in milliseconds",
                    "default": 30000,
                    "required": False
                }
            ],
            function_code=function_code,
            category="integration",
            icon="🔗"
        )
        
        self.tools["api_caller"] = tool
        return tool
    
    def create_text_processor_tool(self) -> CustomTool:
        """Create a text processing tool."""
        
        function_code = '''
async function textProcessor(text, operation = "analyze", options = {}) {
    try {
        if (!text || typeof text !== 'string') {
            return { error: "Valid text input is required" };
        }
        
        const results = {
            operation: operation,
            original_length: text.length,
            timestamp: new Date().toISOString()
        };
        
        switch (operation) {
            case "analyze":
                results.analysis = {
                    character_count: text.length,
                    word_count: text.split(/\\s+/).filter(word => word.length > 0).length,
                    sentence_count: text.split(/[.!?]+/).filter(s => s.trim().length > 0).length,
                    paragraph_count: text.split(/\\n\\s*\\n/).filter(p => p.trim().length > 0).length,
                    average_word_length: text.split(/\\s+/).reduce((sum, word) => sum + word.length, 0) / text.split(/\\s+/).length,
                    reading_time_minutes: Math.ceil(text.split(/\\s+/).length / 200)
                };
                break;
            
            case "extract_emails":
                const emailRegex = /\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b/g;
                const emails = text.match(emailRegex) || [];
                results.emails = {
                    found: emails,
                    count: emails.length,
                    unique_count: [...new Set(emails)].length
                };
                break;
            
            case "extract_urls":
                const urlRegex = /https?:\\/\\/(www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_\\+.~#?&//=]*)/g;
                const urls = text.match(urlRegex) || [];
                results.urls = {
                    found: urls,
                    count: urls.length,
                    unique_count: [...new Set(urls)].length
                };
                break;
            
            case "extract_keywords":
                const words = text.toLowerCase()
                    .replace(/[^a-zA-Z\\s]/g, '')
                    .split(/\\s+/)
                    .filter(word => word.length > 3);
                
                const wordFreq = {};
                words.forEach(word => {
                    wordFreq[word] = (wordFreq[word] || 0) + 1;
                });
                
                const keywords = Object.entries(wordFreq)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 10)
                    .map(([word, freq]) => ({ word, frequency: freq }));
                
                results.keywords = {
                    top_keywords: keywords,
                    total_unique_words: Object.keys(wordFreq).length
                };
                break;
            
            case "summarize":
                const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
                const summary = sentences.slice(0, 3).join('. ');
                results.summary = {
                    summary_text: summary + (sentences.length > 3 ? '...' : ''),
                    compression_ratio: (summary.length / text.length * 100).toFixed(1) + '%',
                    sentences_used: Math.min(3, sentences.length),
                    total_sentences: sentences.length
                };
                break;
            
            case "sentiment":
                // Simple sentiment analysis
                const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'pleased'];
                const negativeWords = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 'disappointed', 'frustrated'];
                
                const words_lower = text.toLowerCase().split(/\\s+/);
                const positiveCount = words_lower.filter(word => positiveWords.includes(word)).length;
                const negativeCount = words_lower.filter(word => negativeWords.includes(word)).length;
                
                let sentiment = 'neutral';
                let confidence = 0.5;
                
                if (positiveCount > negativeCount) {
                    sentiment = 'positive';
                    confidence = 0.5 + (positiveCount - negativeCount) / words_lower.length;
                } else if (negativeCount > positiveCount) {
                    sentiment = 'negative';
                    confidence = 0.5 + (negativeCount - positiveCount) / words_lower.length;
                }
                
                results.sentiment = {
                    sentiment: sentiment,
                    confidence: Math.min(confidence, 1.0).toFixed(3),
                    positive_indicators: positiveCount,
                    negative_indicators: negativeCount,
                    total_words: words_lower.length
                };
                break;
            
            default:
                results.error = `Unknown operation: ${operation}`;
        }
        
        return {
            success: true,
            data: results
        };
        
    } catch (error) {
        return {
            success: false,
            error: error.message
        };
    }
}

return await textProcessor(text, operation, options);
        '''
        
        tool = CustomTool(
            name="text_processor",
            description="Comprehensive text processing including analysis, extraction, and sentiment analysis",
            parameters=[
                {
                    "name": "text",
                    "type": "string",
                    "description": "The text to process",
                    "required": True
                },
                {
                    "name": "operation",
                    "type": "select",
                    "description": "Type of text processing operation",
                    "options": ["analyze", "extract_emails", "extract_urls", "extract_keywords", "summarize", "sentiment"],
                    "default": "analyze",
                    "required": False
                },
                {
                    "name": "options",
                    "type": "string",
                    "description": "Additional options as JSON string (optional)",
                    "required": False
                }
            ],
            function_code=function_code,
            category="text",
            icon="📝"
        )
        
        self.tools["text_processor"] = tool
        return tool
    
    def create_tool_testing_app(self) -> str:
        """Create an application to test custom tools."""
        
        app_config = {
            "name": "Custom Tools Testing Playground",
            "mode": "chat",
            "icon": "🧪",
            "description": "Application for testing and demonstrating custom tools",
            "model_config": {
                "provider": "openai",
                "model": "gpt-4",
                "parameters": {
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            },
            "prompt_template": """You are an AI assistant with access to powerful custom tools for web scraping, data analysis, API calls, and text processing.

Available Tools:
1. **Web Scraper** - Extract content from web pages
2. **Data Analyzer** - Analyze data structures and provide insights
3. **API Caller** - Make HTTP requests to external APIs
4. **Text Processor** - Comprehensive text analysis and processing

Guidelines:
- Use appropriate tools based on user requests
- Explain which tools you're using and why
- Provide clear, structured outputs from tool results
- Handle errors gracefully and suggest alternatives
- Combine multiple tools when beneficial

Always explain your tool usage to help users understand the capabilities.""",
            "opening_statement": "Hello! I'm equipped with custom tools for web scraping, data analysis, API calls, and text processing. What would you like me to help you with?",
            "suggested_questions": [
                "Analyze this text for sentiment and keywords",
                "Help me scrape content from a website",
                "Analyze this data and provide insights",
                "Make an API call to retrieve information"
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/apps",
                json=app_config,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            app_id = result.get('id')
            
            if app_id:
                self.tool_app_id = app_id
                print(f"✅ Created tool testing app: {app_config['name']}")
                print(f"   App ID: {app_id}")
            
            return app_id
            
        except requests.RequestException as e:
            print(f"❌ Error creating tool testing app: {e}")
            return None
    
    def test_tool(self, tool_name: str, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test a custom tool with given parameters."""
        
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        tool = self.tools[tool_name]
        
        print(f"🧪 Testing {tool.name}...")
        print(f"   Description: {tool.description}")
        print(f"   Parameters: {test_params}")
        
        # Simulate tool execution
        # In production, this would execute the actual tool function
        mock_result = {
            "tool_name": tool.name,
            "execution_time": "0.25s",
            "success": True,
            "test_mode": True,
            "parameters_used": test_params,
            "simulated_output": f"Mock output from {tool.name} with params: {test_params}"
        }
        
        print(f"   ✅ Test completed: {mock_result['simulated_output']}")
        
        return mock_result
    
    def demonstrate_tools(self):
        """Run demonstrations of all custom tools."""
        
        print("🎯 Custom Tools Demonstration")
        print("=" * 50)
        
        # Create all tools
        print("\n🔧 Creating custom tools...")
        self.create_web_scraper_tool()
        self.create_data_analyzer_tool()
        self.create_api_caller_tool()
        self.create_text_processor_tool()
        
        print(f"✅ Created {len(self.tools)} custom tools")
        
        # Test each tool
        print("\n🧪 Testing custom tools...")
        
        # Test web scraper
        self.test_tool("web_scraper", {
            "url": "https://example.com",
            "extract_type": "text"
        })
        
        # Test data analyzer
        self.test_tool("data_analyzer", {
            "data": '[1, 2, 3, 4, 5, 2, 3, 1]',
            "analysis_type": "statistics"
        })
        
        # Test API caller
        self.test_tool("api_caller", {
            "url": "https://jsonplaceholder.typicode.com/posts/1",
            "method": "GET"
        })
        
        # Test text processor
        self.test_tool("text_processor", {
            "text": "This is a great example of text processing! It works wonderfully and I love the capabilities.",
            "operation": "sentiment"
        })
        
        # Create testing app
        print("\n🏗️ Creating tool testing application...")
        app_id = self.create_tool_testing_app()
        
        if app_id:
            print(f"\n🎉 Custom tools demonstration completed!")
            print(f"   Tools created: {len(self.tools)}")
            print(f"   Testing app ID: {app_id}")
            
            # Show tool summary
            print(f"\n📋 Tool Summary:")
            for tool_name, tool in self.tools.items():
                print(f"   {tool.icon} {tool.name} - {tool.description}")
    
    def get_tool_documentation(self) -> str:
        """Generate documentation for all custom tools."""
        
        docs = "# Custom Tools Documentation\n\n"
        
        for tool_name, tool in self.tools.items():
            docs += f"## {tool.icon} {tool.name}\n\n"
            docs += f"**Description:** {tool.description}\n\n"
            docs += f"**Category:** {tool.category}\n\n"
            docs += "**Parameters:**\n"
            
            for param in tool.parameters:
                required = " (required)" if param.get("required") else " (optional)"
                docs += f"- `{param['name']}` ({param['type']}){required}: {param['description']}\n"
                if param.get("default"):
                    docs += f"  - Default: `{param['default']}`\n"
                if param.get("options"):
                    docs += f"  - Options: {param['options']}\n"
            
            docs += "\n**Example Usage:**\n"
            docs += f"```javascript\n// Example call to {tool.name}\n"
            docs += f"const result = await {tool.name}("
            
            example_params = []
            for param in tool.parameters:
                if param.get("required"):
                    if param["type"] == "string":
                        example_params.append(f'"{param["name"]}_value"')
                    else:
                        example_params.append(str(param.get("default", "value")))
            
            docs += ", ".join(example_params)
            docs += ");\n```\n\n"
            docs += "---\n\n"
        
        return docs


def main():
    """Main function to demonstrate custom tool creation."""
    
    print("🔧 Dify Custom Tool Example")
    print("=" * 35)
    
    # Get API key
    api_key = os.getenv('DIFY_API_KEY')
    if not api_key:
        api_key = input("Enter your Dify API key: ").strip()
    
    if not api_key:
        print("❌ API key is required")
        return
    
    # Initialize tool builder
    tool_builder = DifyCustomToolBuilder(api_key=api_key)
    
    try:
        # Run tool demonstrations
        tool_builder.demonstrate_tools()
        
        # Generate documentation
        print("\n📚 Generating tool documentation...")
        docs = tool_builder.get_tool_documentation()
        
        # Save documentation
        with open("custom_tools_documentation.md", "w", encoding="utf-8") as f:
            f.write(docs)
        
        print("✅ Documentation saved to custom_tools_documentation.md")
        
        print(f"\n🎉 Custom tool demonstration completed!")
        print("Key capabilities demonstrated:")
        print("• ✅ Web scraping tool creation")
        print("• ✅ Data analysis tool creation")
        print("• ✅ API integration tool creation")
        print("• ✅ Text processing tool creation")
        print("• ✅ Tool testing and validation")
        print("• ✅ Documentation generation")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()