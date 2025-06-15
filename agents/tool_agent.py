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
    "You are an intelligent assistant that uses tools.\n"
    "When solving a problem, always use this format:\n\n"
    "Thought: Think about the task.\n"
    "Action: Tool name to use (Calculator or TavilySearchResults)\n"
    "Action Input: The input you pass to the tool\n"
    "Observation: Output from the tool\n"
    "Final Answer: Your final result\n\n"
    "Example:\n"
    "Thought: I need to calculate 2+2\n"
    "Action: Calculator\n"
    "Action Input: 2+2\n"
    "Observation: 4\n"
    "Final Answer: 4"
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"system_message": system_instruction},
    verbose=True,
    handle_parsing_errors=True,
)


# ✅ Tool Agent function
def tool_agent(task: str) -> str:
    try:
        result = agent.run(task)
        return f"Result: {result}"
    except Exception as e:
        return f"Tool error: {e}"
