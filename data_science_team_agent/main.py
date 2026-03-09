"""Data Science Team Agent - AI data analysis and visualization agent."""

# data_science_team_agent/main.py
import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from textwrap import dedent
from typing import Any

from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv
from openai import AsyncOpenAI
import pandas as pd
import re

# Load environment variables from .env file
load_dotenv()

# Global agent instance
agent: Any = None
_initialized = False
_init_lock = asyncio.Lock()


def load_config() -> dict:
    """Load agent configuration from project root."""
    # Try multiple possible locations for agent_config.json
    possible_paths = [
        Path(__file__).parent.parent / "agent_config.json",  # Project root
        Path(__file__).parent / "agent_config.json",  # Same directory as main.py
        Path.cwd() / "agent_config.json",  # Current working directory
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except (PermissionError, json.JSONDecodeError) as e:
                print(f"⚠️  Error reading {config_path}: {type(e).__name__}")
                continue
            except Exception as e:
                print(f"⚠️  Unexpected error reading {config_path}: {type(e).__name__}")
                continue

    # If no config found or readable, create a minimal default
    print("⚠️  No agent_config.json found, using default configuration")
    return {
        "name": "data-science-team-agent",
        "description": "AI Data Science Team Agent with comprehensive data analysis capabilities",
        "version": "1.0.0",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
            "proxy_urls": ["127.0.0.1"],
            "cors_origins": ["*"],
        },
        "environment_variables": [
            {"key": "OPENROUTER_API_KEY", "description": "OpenRouter API key for LLM calls", "required": True},
            {"key": "MEM0_API_KEY", "description": "Mem0 API key for memory", "required": True},
        ],
    }


async def initialize_agent() -> None:
    """Initialize the data science agent with proper model and tools."""
    global agent

    # Get API keys from environment
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    mem0_api_key = os.getenv("MEM0_API_KEY")
    model_name = os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet")

    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY required. Get your API key from: https://openrouter.ai/keys")

    # Initialize OpenAI client via OpenRouter
    client = AsyncOpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    print(f"✅ Using OpenRouter model: {model_name}")

    # Create the data science agent
    agent = DataScienceAgent(client, model_name, mem0_api_key)
    print("✅ Data Science Team Agent initialized")


class DataScienceAgent:
    """Data Science Agent that handles all data science tasks."""
    
    def __init__(self, client: AsyncOpenAI, model_name: str, mem0_api_key: str | None = None):
        self.client = client
        self.model_name = model_name
        self.mem0_api_key = mem0_api_key
        self.description = dedent("""\
            You are an elite data science team with decades of experience in comprehensive data analysis.
            Your expertise encompasses: 📊

            - Data cleaning and preprocessing
            - Exploratory data analysis (EDA)
            - Data visualization and plotting
            - Feature engineering and selection
            - Machine learning model development
            - Statistical analysis and hypothesis testing
            - Data wrangling and transformation
            - SQL database operations
            - Workflow planning and automation
            - Pandas data manipulation
            - MLflow experiment tracking

            You excel at:
            - Loading data from various sources (CSV, databases, APIs)
            - Identifying and handling missing values
            - Detecting and treating outliers
            - Creating insightful visualizations
            - Building predictive models
            - Generating comprehensive reports
            - Optimizing data pipelines
            - Providing actionable insights\
        """)
        
        self.instructions = dedent("""\
            1. Data Loading & Inspection 🔍
               - Load data from provided URL or source
               - Examine data structure, types, and quality
               - Identify missing values, duplicates, and anomalies

            2. Data Cleaning & Preprocessing 🧹
               - Handle missing values appropriately (imputation, removal)
               - Remove duplicates and correct data types
               - Detect and treat outliers
               - Standardize and normalize features

            3. Exploratory Data Analysis 📊
               - Generate descriptive statistics
               - Create correlation matrices and heatmaps
               - Visualize distributions and relationships
               - Identify patterns and insights

            4. Advanced Analysis (if requested) 🔬
               - Feature engineering for ML tasks
               - Build and evaluate machine learning models
               - Create statistical tests and hypothesis validation
               - Generate comprehensive reports

            5. Results & Documentation 📋
               - Provide clear, actionable insights
               - Include relevant code snippets
               - Suggest next steps and recommendations
               - Format results professionally

            Always:
            - Explain your methodology clearly
            - Include relevant code examples
            - Provide statistical evidence
            - Suggest practical applications
            - Format output for readability\
        """)
    
    async def arun(self, messages: list[dict[str, str]]) -> Any:
        """Run the agent with the given messages using standard format."""
        # Extract user text from standard message format [{"role": "user", "content": "..."}]
        user_text = ""
        
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                user_text = message.get("content", "")
                break
        
        if not user_text:
            return {"text": "No user message found. Please provide a data science task or question."}
        
        print(f"🔍 Processing request: '{user_text[:100]}...'")
        
        try:
            # Check if URL is provided
            url_match = re.search(r'https://[^\s]+', user_text)
            df = None
            
            if url_match:
                url = url_match.group(0)
                try:
                    df = pd.read_csv(url)
                    print(f"📊 Loaded DataFrame with shape: {df.shape}")
                except Exception as e:
                    return {"text": f"❌ Error loading data from URL: {str(e)}"}
            
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(user_text, df)
            
            # Use the model to analyze the request
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"{self.description}\n\n{self.instructions}"},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            # Extract the analysis content
            analysis_content = response.choices[0].message.content
            
            # Return response with content field for frontend compatibility
            class AgentResponse:
                def __init__(self, content):
                    self.content = content
            
            return AgentResponse(f"🧹 **Data Science Analysis Completed**\n\n{analysis_content}")
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            print(error_msg)
            
            class ErrorResponse:
                def __init__(self, content):
                    self.content = content
            
            return ErrorResponse(error_msg)
    
    def _create_analysis_prompt(self, user_request: str, df: pd.DataFrame | None) -> str:
        """Create a comprehensive analysis prompt."""
        if df is not None:
            data_info = f"""
            Dataset Information:
            - Shape: {df.shape}
            - Columns: {list(df.columns)}
            - Data types: {df.dtypes.to_string()}
            - Missing values: {df.isnull().sum().to_string()}
            - First few rows:
            {df.head().to_string()}
            - Descriptive statistics:
            {df.describe().to_string()}
            """
        else:
            data_info = "No dataset provided. Working with general data science guidance."
        
        prompt = f"""
        User Request: {user_request}
        
        {data_info}
        
        Please provide a comprehensive data science response including:
        1. Clear explanation of your approach
        2. Step-by-step methodology
        3. Relevant code examples (in Python/pandas)
        4. Statistical insights and interpretations
        5. Visualization recommendations
        6. Next steps and best practices
        
        Format your response professionally with clear sections and code blocks where appropriate.
        """
        
        return prompt


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages with lazy initialization."""
    global _initialized

    # Lazy initialization on first call
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Data Science Team Agent...")
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages."""
    global agent
    if not agent:
        error_msg = "Agent not initialized"
        raise RuntimeError(error_msg)

    # Run the agent and get response
    response = await agent.arun(messages)
    return response


async def cleanup() -> None:
    """Clean up any resources."""
    print("🧹 Cleaning up Data Science Team Agent resources...")


def main():
    """Run the main entry point for Data Science Team Agent."""
    parser = argparse.ArgumentParser(description="Bindu Data Science Team Agent")
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--mem0-api-key",
        type=str,
        default=os.getenv("MEM0_API_KEY"),
        help="Mem0 API key (env: MEM0_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet"),
        help="Model ID for OpenRouter (env: MODEL_NAME)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to agent_config.json (optional)",
    )
    args = parser.parse_args()

    # Set environment variables if provided via CLI
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.mem0_api_key:
        os.environ["MEM0_API_KEY"] = args.mem0_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Data Science Team Agent - AI Data Analysis & Visualization")
    print("📊 Capabilities: EDA, cleaning, visualization, ML, SQL, workflow planning")

    # Load configuration
    config = load_config()

    try:
        # Bindufy and start the agent server
        print("🚀 Starting Bindu Data Science Team Agent server...")
        print(f"🌐 Server will run on: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3773')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Data Science Team Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup on exit
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
