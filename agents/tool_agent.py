from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

@tool
def calculator_tool(expression: str) -> str:
    """Performs arithmetic operations like add, subtract, multiply."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def knowledge_helper(question: str) -> str:
    """Useful for coding or general questions (Java, Python, logic, etc)."""
    return llm.predict(question)

tools = [
    Tool.from_function(func=calculator_tool, name="Calculator", description="Performs basic arithmetic."),
    Tool.from_function(func=knowledge_helper, name="KnowledgeHelper", description="Handles general/coding questions.")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

def tool_agent(task: str) -> str:
    try:
        return agent.run(task)
    except Exception as e:
        return f"Tool error: {e}"
