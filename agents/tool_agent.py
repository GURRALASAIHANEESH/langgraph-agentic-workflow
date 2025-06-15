from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

@tool
def calculator_tool(expression: str) -> str:
    """Performs basic arithmetic like 2+2 or 5*3"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

tools = [
    Tool.from_function(func=calculator_tool, name="Calculator", description="Handles basic math operations"),
    search
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False
)

def tool_agent(task: str) -> str:
    try:
        return agent.run(task)
    except Exception as e:
        return f"Tool error: {e}"
