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

# ✅ Tool 1: Calculator
@tool
def calculator(expression: str) -> str:
    """Useful for solving math problems like 2+2, 7*3, 2*2*3 etc."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# ✅ Tool 2: Code Helper for programming queries
@tool
def code_helper(task: str) -> str:
    """Helpful for programming questions like Java/Python/C++ syntax or logic."""
    try:
        prompt = f"Give a clear example or explanation for this programming task:\n\n{task}"
        return llm.invoke(prompt).content
    except Exception as e:
        return f"CodeHelper error: {e}"

# ✅ Tool 3: Web search using Tavily
tavily_key = os.getenv("TAVILY_API_KEY")
search = TavilySearchResults(api_key=tavily_key)

# ✅ Define tools list with priority descriptions
tools = [
    Tool.from_function(
        func=code_helper,
        name="CodeHelper",
        description="Highly recommended for programming/code-related questions like 'create array in Java' or 'Python sorting example'"
    ),
    Tool.from_function(
        func=calculator,
        name="Calculator",
        description="Use for arithmetic or math expressions. Input like '15 / 3' or '2+2'"
    ),
    search
]

# ✅ System message to guide tool selection
system_instruction = (
    "You are a helpful assistant who must always choose a tool to solve tasks.\n"
    "Use this reasoning format:\n"
    "Thought: Your reasoning\n"
    "Action: The tool to use (CodeHelper, Calculator, TavilySearchResults)\n"
    "Action Input: What to pass to the tool\n"
    "Observation: What the tool returned\n"
    "Final Answer: What the user needs\n\n"
    "Always pick CodeHelper for any coding-related or syntax questions."
)

# ✅ Create the AgentExecutor
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"system_message": system_instruction},
    verbose=False,
    handle_parsing_errors=True,
)

# ✅ The agent wrapper function
def tool_agent(task: str) -> str:
    # Optional shortcut for coding tasks
    keywords = ["array in java", "create array", "java array", "python", "syntax", "loop", "code"]
    if any(k in task.lower() for k in keywords):
        return f"Result: {code_helper(task)}"
    try:
        result = agent.run(task)
        return f"Result: {result}"
    except Exception as e:
        return f"Tool error: {e}"
