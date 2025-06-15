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

prompt =PromptTemplate(
    input_variables=["input", "results"],
    template="""
You are a reflection agent. The user wanted to: {input}

Here are the completed results:
{results}

Is the user's request fully completed?
- Reply with "YES" if all subtasks are completed correctly.
- Reply with "NO" if anything is missing or wrong.
"""
)

reflect_chain = LLMChain(llm=llm, prompt=prompt)

def reflect_on_results(state):
    response = reflect_chain.invoke({
        "input": state["input"],
        "results": "\n".join(state["results"])
    })

    answer = response.get("text", str(response)).strip().upper()

    state["retry_count"] = state.get("retry_count", 0) + 1

    if "NO" in answer:
        if state["retry_count"] >= 2:
            state["done"] = True
        else:
            state["done"] = False
    else:
        state["done"] = True

    return state
