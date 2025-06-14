from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

# ✅ Correct calculator with docstring (required)
@tool
def calculator_tool(expression: str) -> str:
    """Performs basic math operations like addition, subtraction, etc."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# ✅ Define tools list
tools = [
    Tool.from_function(
        func=calculator_tool,
        name="Calculator",
        description="Useful for basic math problems like add, subtract, multiply, divide"
    )
]

# ✅ Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def tool_agent(task: str) -> str:
    try:
        return agent.run(task)
    except Exception as e:
        return f"Tool error: {e}"
