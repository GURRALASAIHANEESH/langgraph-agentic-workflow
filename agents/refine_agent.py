from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load .env or Streamlit Cloud secrets
load_dotenv()
openai_key = os.getenv("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")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# Initialize the LLM
llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",
    openai_api_key=openai_key,
    openai_api_base=os.environ["OPENAI_API_BASE"]
)

# Prompt template
prompt = PromptTemplate(
    input_variables=["input", "subtasks"],
    template="""
You are a task refinement agent. A user asked: "{input}"

Here are the initial subtasks:
{subtasks}

Your job is to improve this task list by:
- Reordering if needed
- Removing irrelevant tasks
- Adding missing steps
- Making it more precise and actionable

Return the refined subtasks as a numbered list:
"""
)

# Create refinement chain
refine_chain = LLMChain(llm=llm, prompt=prompt)

# Function to refine initial subtasks (used after PlanAgent)
def refine_tasks(state):
    response = refine_chain.invoke({
        "input": state["input"],
        "subtasks": "\n".join(state["subtasks"])
    })

    response_text = response["text"] if isinstance(response, dict) and "text" in response else str(response)
    refined = [line.strip("-• \n") for line in response_text.split("\n") if line.strip()]
    state["subtasks"] = refined
    return state

# Function to refine subtasks based on results (used by ReflectAgent)
def refine_agent(state):
    response = refine_chain.invoke({
        "input": state["input"],
        "subtasks": "\n".join(state["results"])
    })

    response_text = response["text"] if isinstance(response, dict) and "text" in response else str(response)
    refined = [line.strip("-• \n") for line in response_text.split("\n") if line.strip()]
    state["subtasks"] = refined
    return state
