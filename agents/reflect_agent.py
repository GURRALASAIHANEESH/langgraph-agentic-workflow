from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Setup OpenRouter
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",
    openai_api_base=os.environ["OPENAI_API_BASE"]
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

    # Decide what to do
    if "NO" in response.upper():
        # Add a retry limit counter
        state["retry_count"] = state.get("retry_count", 0) + 1
        if state["retry_count"] > 2:  # Don't loop more than 2 times
            state["done"] = True  # Force end
        else:
            state["done"] = False
    else:
        state["done"] = True
