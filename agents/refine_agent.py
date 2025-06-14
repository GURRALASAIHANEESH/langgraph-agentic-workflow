from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ Load API config
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

# ✅ Prompt for refining subtasks
prompt = PromptTemplate(
    input_variables=["input", "subtasks", "results"],
    template="""
You're a smart refinement agent helping improve a task execution plan.

Original request: {input}

Current Subtasks:
{subtasks}

Results Completed So Far:
{results}

Please refine the subtasks:
- Improve clarity
- Remove duplicates
- Add any missing tasks
Return ONLY the refined subtasks as a bullet list.
"""
)

refine_chain = LLMChain(llm=llm, prompt=prompt)

def refine_tasks(state):
    """Refines the subtask list based on current input and results."""
    try:
        response = refine_chain.invoke({
            "input": state["input"],
            "subtasks": "\n".join(state.get("subtasks", [])),
            "results": "\n".join(state.get("results", [])),
        })
        text = response.get("text", str(response))
        refined_subtasks = [line.strip("-• \n") for line in text.split("\n") if line.strip()]
        return {**state, "subtasks": refined_subtasks}
    except Exception as e:
        return {**state, "subtasks": state.get("subtasks", []), "results": state.get("results", []) + [f"Refine Error: {e}"]}
