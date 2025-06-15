from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

prompt = PromptTemplate(
    input_variables=["input", "results"],
    template="""
You are a reflection agent. The original user request was:
"{input}"

The system completed these subtasks:
{results}

Question: Does the result fully satisfy the original request?

Reply with:
- "YES" if complete.
- "NO" if more steps are needed.
"""
)

reflect_chain = LLMChain(llm=llm, prompt=prompt)

def reflect_on_results(state):
    response = reflect_chain.run({
        "input": state["input"],
        "results": "\n".join(state["results"])
    })
    if "NO" in response.upper():
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["done"] = state["retry_count"] > 2
    else:
        state["done"] = True
    return state
