from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",
    openai_api_key=os.getenv("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"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

# Prompt template
prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a task planner AI. Break the following high-level request into 3 to 5 clear and actionable subtasks:

User Query: {input}

Subtasks (as a list):
"""
)

# LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# PlanAgent function
def plan_agent(user_input: str) -> list:
    response = chain.invoke({"input": user_input})

    response_text = response["text"] if isinstance(response, dict) and "text" in response else str(response)
    subtasks = [line.strip("-• \n") for line in response_text.split("\n") if line.strip()]
    return subtasks
