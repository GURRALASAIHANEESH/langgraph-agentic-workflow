from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os, re
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a task planner AI. Break the user's high-level request into 3–5 clear and actionable subtasks.

Request: {input}

Subtasks:
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

def plan_agent(user_input: str) -> list:
    if re.fullmatch(r"\s*\d+\s*[\+\-\*/]\s*\d+\s*", user_input):
        return [user_input.strip()]
    response = chain.invoke({"input": user_input})
    text = response.get("text", str(response))
    return [line.strip("-• \n") for line in text.split("\n") if line.strip()]
