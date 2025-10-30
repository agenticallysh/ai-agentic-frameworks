#!/usr/bin/env python3
"""
Langflow Custom Component Development Example

This example demonstrates how to create custom components for Langflow,
including input/output handling, configuration options, and integration
with external APIs and services.

Requirements:
    pip install langflow requests beautifulsoup4 pandas pillow
    
Usage:
    python custom-component.py
"""

import asyncio
import json
import os
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from langflow.custom import CustomComponent
from langflow.components import OpenAIModel
from langflow.graph import Graph


class WebScrapingComponent(CustomComponent):
    """Custom component for web scraping and content extraction."""
    
    display_name = "Web Scraper"
    description = "Scrapes web content and extracts structured information"
    icon = "🕷️"
    
    def build_config(self):
        """Define the component's configuration interface."""
        return {
            "url": {
                "display_name": "URL",
                "type": "str",
                "required": True,
                "info": "URL to scrape content from"
            },
            "extract_type": {
                "display_name": "Extract Type",
                "type": "str",
                "options": ["text", "links", "images", "tables", "all"],
                "value": "text",
                "info": "Type of content to extract"
            },
            "css_selector": {
                "display_name": "CSS Selector",
                "type": "str",
                "required": False,
                "info": "Optional CSS selector to target specific elements"
            },
            "max_length": {
                "display_name": "Max Content Length",
                "type": "int",
                "value": 5000,
                "info": "Maximum length of extracted content"
            },
            "include_metadata": {
                "display_name": "Include Metadata",
                "type": "bool",
                "value": True,
                "info": "Include page metadata (title, description, etc.)"
            }
        }
    
    def build(
        self, 
        url: str, 
        extract_type: str = "text",
        css_selector: str = None,
        max_length: int = 5000,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Main component logic for web scraping."""
        
        try:
            # Fetch the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content based on type
            extracted_content = self._extract_content(soup, extract_type, css_selector, max_length)
            
            # Get metadata if requested
            metadata = self._extract_metadata(soup) if include_metadata else {}
            
            return {
                "url": url,
                "content": extracted_content,
                "metadata": metadata,
                "extract_type": extract_type,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except requests.RequestException as e:
            return {
                "url": url,
                "error": f"Request failed: {str(e)}",
                "status": "error"
            }
        except Exception as e:
            return {
                "url": url,
                "error": f"Scraping failed: {str(e)}",
                "status": "error"
            }
    
    def _extract_content(self, soup, extract_type: str, css_selector: str, max_length: int):
        """Extract content based on the specified type."""
        
        if css_selector:
            elements = soup.select(css_selector)
            if not elements:
                return "No elements found for the given CSS selector"
        else:
            elements = [soup]
        
        if extract_type == "text":
            text = " ".join([elem.get_text(strip=True) for elem in elements])
            return text[:max_length] if len(text) > max_length else text
        
        elif extract_type == "links":
            links = []
            for elem in elements:
                for link in elem.find_all('a', href=True):
                    links.append({
                        "text": link.get_text(strip=True),
                        "url": link['href']
                    })
            return links[:50]  # Limit to 50 links
        
        elif extract_type == "images":
            images = []
            for elem in elements:
                for img in elem.find_all('img', src=True):
                    images.append({
                        "alt": img.get('alt', ''),
                        "src": img['src']
                    })
            return images[:20]  # Limit to 20 images
        
        elif extract_type == "tables":
            tables = []
            for elem in elements:
                for table in elem.find_all('table'):
                    rows = []
                    for row in table.find_all('tr'):
                        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                        rows.append(cells)
                    tables.append(rows)
            return tables[:5]  # Limit to 5 tables
        
        elif extract_type == "all":
            return {
                "text": self._extract_content(soup, "text", css_selector, max_length),
                "links": self._extract_content(soup, "links", css_selector, max_length),
                "images": self._extract_content(soup, "images", css_selector, max_length),
                "tables": self._extract_content(soup, "tables", css_selector, max_length)
            }
        
        return ""
    
    def _extract_metadata(self, soup) -> Dict[str, str]:
        """Extract page metadata."""
        
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # Meta description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag:
            metadata['description'] = desc_tag.get('content', '')
        
        # Meta keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            metadata['keywords'] = keywords_tag.get('content', '')
        
        # Open Graph data
        og_title = soup.find('meta', property='og:title')
        if og_title:
            metadata['og_title'] = og_title.get('content', '')
        
        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            metadata['og_description'] = og_desc.get('content', '')
        
        return metadata


class DataProcessingComponent(CustomComponent):
    """Custom component for advanced data processing and transformation."""
    
    display_name = "Data Processor"
    description = "Process and transform data with various operations"
    icon = "⚙️"
    
    def build_config(self):
        return {
            "data": {
                "display_name": "Input Data",
                "type": "any",
                "required": True,
                "info": "Data to process (can be dict, list, or string)"
            },
            "operation": {
                "display_name": "Operation",
                "type": "str",
                "options": ["clean", "transform", "aggregate", "filter", "sort", "validate"],
                "value": "clean",
                "info": "Type of processing operation"
            },
            "parameters": {
                "display_name": "Parameters",
                "type": "dict",
                "value": {},
                "info": "Additional parameters for the operation"
            },
            "output_format": {
                "display_name": "Output Format",
                "type": "str",
                "options": ["json", "csv", "text", "dict"],
                "value": "dict",
                "info": "Format for the processed output"
            }
        }
    
    def build(
        self, 
        data: Any, 
        operation: str = "clean",
        parameters: Dict[str, Any] = None,
        output_format: str = "dict"
    ) -> Dict[str, Any]:
        """Process data according to the specified operation."""
        
        if parameters is None:
            parameters = {}
        
        try:
            # Convert data to a workable format
            processed_data = self._prepare_data(data)
            
            # Apply the requested operation
            result = self._apply_operation(processed_data, operation, parameters)
            
            # Format the output
            formatted_result = self._format_output(result, output_format)
            
            return {
                "original_data_type": type(data).__name__,
                "operation": operation,
                "parameters": parameters,
                "result": formatted_result,
                "output_format": output_format,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": f"Data processing failed: {str(e)}",
                "operation": operation,
                "status": "error"
            }
    
    def _prepare_data(self, data: Any) -> Any:
        """Prepare data for processing."""
        
        if isinstance(data, str):
            try:
                # Try to parse as JSON
                return json.loads(data)
            except json.JSONDecodeError:
                # Return as is if not JSON
                return data
        
        return data
    
    def _apply_operation(self, data: Any, operation: str, parameters: Dict[str, Any]) -> Any:
        """Apply the specified operation to the data."""
        
        if operation == "clean":
            return self._clean_data(data, parameters)
        elif operation == "transform":
            return self._transform_data(data, parameters)
        elif operation == "aggregate":
            return self._aggregate_data(data, parameters)
        elif operation == "filter":
            return self._filter_data(data, parameters)
        elif operation == "sort":
            return self._sort_data(data, parameters)
        elif operation == "validate":
            return self._validate_data(data, parameters)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _clean_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Clean data by removing nulls, duplicates, etc."""
        
        if isinstance(data, list):
            # Remove None values and duplicates
            cleaned = [item for item in data if item is not None]
            if parameters.get("remove_duplicates", True):
                cleaned = list(dict.fromkeys(cleaned))  # Preserve order
            return cleaned
        
        elif isinstance(data, dict):
            # Remove None values
            cleaned = {k: v for k, v in data.items() if v is not None}
            return cleaned
        
        elif isinstance(data, str):
            # Clean text data
            cleaned = data.strip()
            if parameters.get("remove_extra_spaces", True):
                cleaned = " ".join(cleaned.split())
            return cleaned
        
        return data
    
    def _transform_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Transform data structure or values."""
        
        transform_type = parameters.get("type", "uppercase")
        
        if isinstance(data, str):
            if transform_type == "uppercase":
                return data.upper()
            elif transform_type == "lowercase":
                return data.lower()
            elif transform_type == "title":
                return data.title()
        
        elif isinstance(data, list):
            if transform_type == "reverse":
                return data[::-1]
            elif transform_type == "flatten" and all(isinstance(item, list) for item in data):
                return [item for sublist in data for item in sublist]
        
        elif isinstance(data, dict):
            if transform_type == "keys_to_lowercase":
                return {k.lower(): v for k, v in data.items()}
            elif transform_type == "values_to_string":
                return {k: str(v) for k, v in data.items()}
        
        return data
    
    def _aggregate_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Aggregate data with statistical operations."""
        
        if not isinstance(data, list):
            return {"error": "Aggregation requires list data"}
        
        numeric_data = [item for item in data if isinstance(item, (int, float))]
        
        if not numeric_data:
            return {"error": "No numeric data found for aggregation"}
        
        return {
            "count": len(numeric_data),
            "sum": sum(numeric_data),
            "average": sum(numeric_data) / len(numeric_data),
            "min": min(numeric_data),
            "max": max(numeric_data),
            "unique_values": len(set(numeric_data))
        }
    
    def _filter_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Filter data based on criteria."""
        
        if isinstance(data, list):
            filter_type = parameters.get("type", "length")
            
            if filter_type == "length":
                min_length = parameters.get("min_length", 0)
                max_length = parameters.get("max_length", float('inf'))
                return [item for item in data if min_length <= len(str(item)) <= max_length]
            
            elif filter_type == "contains":
                search_term = parameters.get("search_term", "")
                return [item for item in data if search_term.lower() in str(item).lower()]
        
        return data
    
    def _sort_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Sort data according to criteria."""
        
        if isinstance(data, list):
            reverse = parameters.get("reverse", False)
            
            try:
                return sorted(data, reverse=reverse)
            except TypeError:
                # If items aren't comparable, sort by string representation
                return sorted(data, key=str, reverse=reverse)
        
        elif isinstance(data, dict):
            sort_by = parameters.get("sort_by", "keys")
            reverse = parameters.get("reverse", False)
            
            if sort_by == "keys":
                return dict(sorted(data.items(), reverse=reverse))
            elif sort_by == "values":
                return dict(sorted(data.items(), key=lambda x: x[1], reverse=reverse))
        
        return data
    
    def _validate_data(self, data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against specified criteria."""
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "data_type": type(data).__name__,
            "size": len(data) if hasattr(data, '__len__') else 1
        }
        
        # Type validation
        expected_type = parameters.get("expected_type")
        if expected_type and not isinstance(data, eval(expected_type)):
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Expected {expected_type}, got {type(data).__name__}")
        
        # Size validation
        min_size = parameters.get("min_size")
        max_size = parameters.get("max_size")
        
        if hasattr(data, '__len__'):
            size = len(data)
            if min_size is not None and size < min_size:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Size {size} is less than minimum {min_size}")
            
            if max_size is not None and size > max_size:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Size {size} exceeds maximum {max_size}")
        
        return validation_results
    
    def _format_output(self, data: Any, format_type: str) -> Any:
        """Format the output according to the specified format."""
        
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        elif format_type == "csv":
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # Convert list of dicts to CSV format
                import io
                import csv
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
                return output.getvalue()
            else:
                return str(data)
        elif format_type == "text":
            return str(data)
        else:  # dict
            return data


class APIIntegrationComponent(CustomComponent):
    """Custom component for integrating with external APIs."""
    
    display_name = "API Integrator"
    description = "Make API calls and handle responses"
    icon = "🔌"
    
    def build_config(self):
        return {
            "endpoint": {
                "display_name": "API Endpoint",
                "type": "str",
                "required": True,
                "info": "Full URL of the API endpoint"
            },
            "method": {
                "display_name": "HTTP Method",
                "type": "str",
                "options": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "value": "GET",
                "info": "HTTP method for the request"
            },
            "headers": {
                "display_name": "Headers",
                "type": "dict",
                "value": {"Content-Type": "application/json"},
                "info": "HTTP headers to include"
            },
            "payload": {
                "display_name": "Request Payload",
                "type": "dict",
                "value": {},
                "info": "Data to send with the request"
            },
            "timeout": {
                "display_name": "Timeout (seconds)",
                "type": "int",
                "value": 30,
                "info": "Request timeout in seconds"
            },
            "auth_token": {
                "display_name": "Auth Token",
                "type": "str",
                "password": True,
                "info": "Authentication token (optional)"
            }
        }
    
    def build(
        self,
        endpoint: str,
        method: str = "GET",
        headers: Dict[str, str] = None,
        payload: Dict[str, Any] = None,
        timeout: int = 30,
        auth_token: str = None
    ) -> Dict[str, Any]:
        """Make an API call and return the response."""
        
        if headers is None:
            headers = {"Content-Type": "application/json"}
        
        if payload is None:
            payload = {}
        
        # Add authentication if provided
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        try:
            # Make the API call
            response = requests.request(
                method=method.upper(),
                url=endpoint,
                headers=headers,
                json=payload if method.upper() != "GET" else None,
                params=payload if method.upper() == "GET" else None,
                timeout=timeout
            )
            
            # Try to parse JSON response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = response.text
            
            return {
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": response_data,
                "success": response.status_code < 400,
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.RequestException as e:
            return {
                "endpoint": endpoint,
                "method": method,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }


class CustomComponentDemo:
    """Demonstration of custom components in a Langflow workflow."""
    
    def __init__(self):
        self.components = {
            "web_scraper": WebScrapingComponent(),
            "data_processor": DataProcessingComponent(),
            "api_integrator": APIIntegrationComponent()
        }
        
        self.graph = None
    
    def build_demo_workflow(self) -> Graph:
        """Build a demonstration workflow using custom components."""
        
        self.graph = Graph(name="Custom Component Demo")
        
        # Add custom components
        for name, component in self.components.items():
            self.graph.add_component(name, component)
        
        return self.graph
    
    async def demo_web_scraping(self):
        """Demonstrate web scraping component."""
        
        print("\n🕷️  Web Scraping Component Demo")
        print("-" * 40)
        
        # Example URLs to scrape
        test_urls = [
            "https://httpbin.org/html",  # Test HTML page
            "https://jsonplaceholder.typicode.com/posts/1",  # JSON API
        ]
        
        for url in test_urls:
            print(f"\n📄 Scraping: {url}")
            
            result = self.components["web_scraper"].build(
                url=url,
                extract_type="all",
                include_metadata=True
            )
            
            if result.get("status") == "success":
                print(f"✅ Success! Extracted content from {url}")
                print(f"   Metadata: {result.get('metadata', {})}")
                if isinstance(result.get('content'), dict):
                    print(f"   Text length: {len(result['content'].get('text', ''))}")
                    print(f"   Links found: {len(result['content'].get('links', []))}")
                else:
                    print(f"   Content length: {len(str(result.get('content', '')))}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    async def demo_data_processing(self):
        """Demonstrate data processing component."""
        
        print("\n⚙️  Data Processing Component Demo")
        print("-" * 40)
        
        # Test data
        test_datasets = [
            {
                "name": "Number List",
                "data": [1, 5, 3, 8, 2, 5, 1, 9, 7, 3],
                "operation": "aggregate"
            },
            {
                "name": "Text List",
                "data": ["hello world", "  PYTHON  ", "JavaScript", "hello world", None, ""],
                "operation": "clean"
            },
            {
                "name": "Dictionary",
                "data": {"Name": "john", "AGE": 30, "city": "NEW YORK"},
                "operation": "transform",
                "parameters": {"type": "keys_to_lowercase"}
            }
        ]
        
        for dataset in test_datasets:
            print(f"\n📊 Processing: {dataset['name']}")
            print(f"   Input: {dataset['data']}")
            
            result = self.components["data_processor"].build(
                data=dataset["data"],
                operation=dataset["operation"],
                parameters=dataset.get("parameters", {}),
                output_format="dict"
            )
            
            if result.get("status") == "success":
                print(f"✅ Result: {result['result']}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    async def demo_api_integration(self):
        """Demonstrate API integration component."""
        
        print("\n🔌 API Integration Component Demo")
        print("-" * 40)
        
        # Test API endpoints
        test_apis = [
            {
                "name": "JSONPlaceholder GET",
                "endpoint": "https://jsonplaceholder.typicode.com/posts/1",
                "method": "GET"
            },
            {
                "name": "HTTPBin POST",
                "endpoint": "https://httpbin.org/post",
                "method": "POST",
                "payload": {"message": "Hello from Langflow!", "timestamp": datetime.now().isoformat()}
            }
        ]
        
        for api_test in test_apis:
            print(f"\n🌐 Testing: {api_test['name']}")
            print(f"   Endpoint: {api_test['endpoint']}")
            
            result = self.components["api_integrator"].build(
                endpoint=api_test["endpoint"],
                method=api_test["method"],
                payload=api_test.get("payload", {})
            )
            
            if result.get("success"):
                print(f"✅ Status: {result['status_code']}")
                print(f"   Response preview: {str(result['data'])[:100]}...")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    async def demo_component_integration(self):
        """Demonstrate how components can work together."""
        
        print("\n🔄 Component Integration Demo")
        print("-" * 50)
        
        # Step 1: Scrape data from a webpage
        print("Step 1: Scraping data...")
        scrape_result = self.components["web_scraper"].build(
            url="https://httpbin.org/json",
            extract_type="text"
        )
        
        if scrape_result.get("status") != "success":
            print("❌ Scraping failed, using mock data...")
            scraped_data = '{"name": "John Doe", "age": 30, "skills": ["Python", "JavaScript", "AI"]}'
        else:
            scraped_data = scrape_result.get("content", "")
        
        # Step 2: Process the scraped data
        print("Step 2: Processing scraped data...")
        process_result = self.components["data_processor"].build(
            data=scraped_data,
            operation="clean",
            output_format="dict"
        )
        
        processed_data = process_result.get("result", {})
        
        # Step 3: Send processed data to an API
        print("Step 3: Sending to API...")
        api_result = self.components["api_integrator"].build(
            endpoint="https://httpbin.org/post",
            method="POST",
            payload={
                "processed_data": processed_data,
                "processing_info": {
                    "operation": process_result.get("operation"),
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
        
        print(f"\n🎯 Integration Results:")
        print(f"   Scraping: {'✅' if scrape_result.get('status') == 'success' else '❌'}")
        print(f"   Processing: {'✅' if process_result.get('status') == 'success' else '❌'}")
        print(f"   API Call: {'✅' if api_result.get('success') else '❌'}")
        
        if api_result.get("success"):
            response_data = api_result.get("data", {})
            if isinstance(response_data, dict) and "json" in response_data:
                print(f"   Final payload sent: {response_data['json']}")
    
    async def interactive_demo(self):
        """Run an interactive demo of all custom components."""
        
        print("\n🎨 Custom Component Interactive Demo")
        print("=" * 50)
        
        demos = {
            "1": ("Web Scraping", self.demo_web_scraping),
            "2": ("Data Processing", self.demo_data_processing),
            "3": ("API Integration", self.demo_api_integration),
            "4": ("Component Integration", self.demo_component_integration),
            "5": ("All Demos", self.run_all_demos)
        }
        
        while True:
            print("\nAvailable Demos:")
            for key, (name, _) in demos.items():
                print(f"  {key}. {name}")
            print("  q. Quit")
            
            choice = input("\nSelect demo (1-5, q): ").strip().lower()
            
            if choice == 'q':
                print("👋 Demo ended!")
                break
            
            if choice in demos:
                try:
                    await demos[choice][1]()
                except Exception as e:
                    print(f"❌ Demo failed: {e}")
            else:
                print("❌ Invalid choice. Please try again.")
    
    async def run_all_demos(self):
        """Run all component demos in sequence."""
        
        print("\n🚀 Running All Component Demos")
        print("=" * 50)
        
        await self.demo_web_scraping()
        await self.demo_data_processing()
        await self.demo_api_integration()
        await self.demo_component_integration()
        
        print(f"\n{'=' * 50}")
        print("✅ All demos completed!")


async def main():
    """Main function to demonstrate custom components."""
    
    print("🎨 Langflow Custom Component Development Demo")
    print("=" * 60)
    
    # Create demo instance
    demo = CustomComponentDemo()
    
    # Build the workflow
    graph = demo.build_demo_workflow()
    print(f"✅ Built demo workflow with {len(graph.components)} custom components")
    
    # Show component information
    print("\n📋 Available Custom Components:")
    for name, component in demo.components.items():
        print(f"   {component.icon} {component.display_name}: {component.description}")
    
    # Run interactive demo
    await demo.interactive_demo()


if __name__ == "__main__":
    asyncio.run(main())