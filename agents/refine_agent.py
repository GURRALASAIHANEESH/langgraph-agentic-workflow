from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Use OpenRouter
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",  # or Claude if preferred
    openai_api_base=os.environ["OPENAI_API_BASE"]
)

# Prompt to refine the subtasks
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

refine_chain = LLMChain(llm=llm, prompt=prompt)

def refine_tasks(state):
    response = refine_chain.invoke({
        "input": state["input"],
        "subtasks": "\n".join(state["subtasks"])
    })

    if isinstance(response, dict) and "text" in response:
        response_text = response["text"]
    else:
        response_text = str(response)

    refined = [line.strip("-• \n") for line in response_text.split("\n") if line.strip()]
    state["subtasks"] = refined
    return state

def refine_agent(state):
    response = refine_chain.invoke({
        "input": state["input"],
        "results": "\n".join(state["results"])
    })

    if isinstance(response, dict) and "text" in response:
        response_text = response["text"]
    else:
        response_text = str(response)

    refined = [line.strip("-• \n") for line in response_text.split("\n") if line.strip()]
    state["subtasks"] = refined
    return state
