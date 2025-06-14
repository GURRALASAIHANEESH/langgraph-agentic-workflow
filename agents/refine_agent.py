from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

prompt = PromptTemplate(
    input_variables=["input", "subtasks", "results"],
    template="""
You're a refinement agent. The user wants to: {input}

Current subtasks:
{subtasks}

Completed results so far:
{results}

Refine the subtasks list. Keep, remove, or add subtasks to make the plan more complete.
Return the updated subtask list as a bullet list.
"""
)

refine_chain = LLMChain(llm=llm, prompt=prompt)

def refine_tasks(state):
    response = refine_chain.invoke({
        "input": state["input"],
        "subtasks": "\n".join(state["subtasks"]),
        "results": "\n".join(state["results"])
    })
    response_text = response["text"] if isinstance(response, dict) and "text" in response else str(response)
    subtasks = [line.strip("-• \n") for line in response_text.split("\n") if line.strip()]
    return {**state, "subtasks": subtasks}

