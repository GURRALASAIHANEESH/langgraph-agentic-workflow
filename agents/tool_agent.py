from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("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")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",
    openai_api_base=os.environ["OPENAI_API_BASE"]
)

# ✅ Tool 1: Simple calculator
@tool
def calculator_tool(expression: str) -> str:
    """Performs basic arithmetic like addition, subtraction, etc."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

#\ Get from https://app.tavily.com/
tavily_key = os.getenv("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")

search = TavilySearchResults()

# ✅ Define tool list
tools = [
    Tool.from_function(func=calculator_tool, name="Calculator", description="For math problems"),
    search
]

# ✅ Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# ✅ The new smart ToolAgent
def tool_agent(task: str) -> str:
    try:
        response = agent.run(task)
        return f"Result: {response}"
    except Exception as e:
        return f"Error during execution: {e}"
