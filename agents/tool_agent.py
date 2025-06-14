from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

@tool
def calculator_tool(expression: str) -> str:
    """Performs basic arithmetic like addition, subtraction, etc."""  # ✅ This is what it needs
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


search = TavilySearchResults()

tools = [
    Tool.from_function(func=calculator_tool, name="Calculator", description="For math problems"),
    search
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

def tool_agent(task: str) -> str:
    try:
        response = agent.run(task)
        return f"Result: {response}"
    except Exception as e:
        return f"Error during execution: {e}"
