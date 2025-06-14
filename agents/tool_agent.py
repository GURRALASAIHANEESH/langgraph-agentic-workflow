from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

@tool
def calculator_tool(expression: str) -> str:
    """Evaluate basic arithmetic expressions like 3+2 or 10/5."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Math Error: {e}"

@tool
def knowledge_helper(query: str) -> str:
    """Answer general, logic, or programming questions."""
    return llm.predict(query)

tools = [
    Tool.from_function(func=calculator_tool, name="Calculator", description="Handles basic math operations"),
    Tool.from_function(func=knowledge_helper, name="KnowledgeHelper", description="Answers general questions")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

def tool_agent(task: str) -> str:
    try:
        return agent.run(task)
    except Exception as e:
        return f"Tool Error: {e}"
