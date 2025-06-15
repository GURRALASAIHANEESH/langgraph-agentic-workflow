from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ LLM Setup
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

# ✅ Calculator Tool
@tool
def calculator(expression: str) -> str:
    """Useful for solving math problems like 2+2, 7*3, etc."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# ✅ CodeHelper Tool
@tool
def code_helper(task: str) -> str:
    """Use this for programming-related questions: Java, Python, etc."""
    prompt = f"You are a helpful programming assistant. Help with: {task}"
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"CodeHelper error: {e}"

# ✅ Tavily Search Tool
tavily_key = os.getenv("TAVILY_API_KEY")
search = TavilySearchResults(api_key=tavily_key)

# ✅ Tool List
tools = [
    Tool.from_function(
        func=calculator,
        name="Calculator",
        description="Use for arithmetic or math expressions like '2+2' or '15 / 3'."
    ),
    Tool.from_function(
        func=code_helper,
        name="CodeHelper",
        description="Use for logic, programming, or code-based questions like 'create array in Java'."
    ),
    search
]

# ✅ System Prompt to Guide the Agent
system_instruction = (
    "You are an intelligent assistant who solves tasks using the following tools:\n\n"
    "- Calculator: Use for numeric or arithmetic tasks.\n"
    "- CodeHelper: Use for programming or code-based questions (e.g., Java, Python).\n"
    "- TavilySearchResults: Use for general search or informational lookups from the web.\n\n"
    "Always reason step-by-step. Use this format:\n"
    "Thought: ...\n"
    "Action: ...\n"
    "Action Input: ...\n"
    "Observation: ...\n"
    "Final Answer: ..."
)

# ✅ Agent Initialization
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"system_message": system_instruction},
    verbose=True,
    handle_parsing_errors=True,
)

# ✅ ToolAgent Entry Point
def tool_agent(task: str) -> str:
    try:
        result = agent.run(task)
        return f"Result: {result}"
    except Exception as e:
        return f"Tool error: {e}"
