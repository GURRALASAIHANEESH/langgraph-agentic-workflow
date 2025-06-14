from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
# Load API keys
load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",
    openai_api_key=os.getenv("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"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

# Prompt for reflection
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
    response = reflect_chain.invoke({
        "input": state["input"],
        "results": "\n".join(state["results"])
    })

    response_text = response["text"] if isinstance(response, dict) and "text" in response else str(response)

    if "NO" in response_text.upper():
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["done"] = state["retry_count"] > 2  # Stop after 2 retries
    else:
        state["done"] = True

    return state
