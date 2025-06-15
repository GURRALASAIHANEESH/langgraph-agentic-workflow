from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ LLM setup
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

# ✅ Calculator tool
@tool
def calculator(expression: str) -> str:
    """Useful for solving math problems like 2+2, 7*3, etc."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# ✅ Search tool (Tavily)
tavily_key = os.getenv("TAVILY_API_KEY")
search = TavilySearchResults(api_key=tavily_key)

# ✅ Tool list
tools = [
    Tool.from_function(
        func=calculator,
        name="Calculator",
        description="Use for arithmetic or math expressions. Input should be like '2+2' or '15 / 3'"
    ),
    search
]

# ✅ Force tools to be used via system message
system_instruction = (
    "You are an intelligent agent. You MUST always respond in the following format:\n"
    "Thought: describe what you are thinking\n"
    "Action: the tool to use (Calculator or TavilySearchResults)\n"
    "Action Input: the input to the tool\n"
    "When you get a result, reply like this:\n"
    "Observation: tool output\n"
    "Final Answer: the final result\n"
)

# ✅ Custom prompt wrapper for tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": system_instruction}
)

# ✅ Tool Agent function
def tool_agent(task: str) -> str:
    try:
        result = agent.run(task)
        return f"Result: {result}"
    except Exception as e:
        return f"Tool error: {e}"
