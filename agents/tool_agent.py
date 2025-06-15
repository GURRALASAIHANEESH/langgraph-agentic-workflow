from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

# ✅ Tool 1: Calculator
@tool
def calculator_tool(expression: str) -> str:
    """Useful for solving basic math expressions like 2+2 or 10/5."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Math error: {str(e)}"

# ✅ Tool 2: Tavily Search
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise EnvironmentError("TAVILY_API_KEY not set in environment")

search_tool = TavilySearchResults(api_key=tavily_api_key)

# ✅ Tool list
tools = [
    Tool.from_function(
        func=calculator_tool,
        name="Calculator",
        description="Use this to solve basic math problems.",
    ),
    search_tool
]

# ✅ Create Agent
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# ✅ Tool Agent Logic
def tool_agent(task: str) -> str:
    try:
        response = agent_executor.run(task)
        return f"Result: {response}"
    except Exception as e:
        return f"Tool error: {e}"
