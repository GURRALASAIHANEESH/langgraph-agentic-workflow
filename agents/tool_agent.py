from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

@tool
def calculator_tool(expression: str) -> str:
    """Performs basic math operations like addition or subtraction."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

# Web search tool
search = TavilySearchResults()

tools = [
    Tool.from_function(func=calculator_tool, name="Calculator", description="Useful for simple math problems."),
    search
]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

def tool_agent(task: str) -> str:
    try:
        response = agent_executor.run(task)
        return f"Result: {response}"
    except Exception as e:
        return f"Tool error: {e}"
