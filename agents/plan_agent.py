from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("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")

os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Choose a model supported by OpenRouter
llm = ChatOpenAI(
    temperature=0,
    model_name="openai/gpt-3.5-turbo",  # or try "anthropic/claude-3-haiku-20240307"
    openai_api_base=os.environ["OPENAI_API_BASE"]
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

chain = LLMChain(llm=llm, prompt=prompt)

def plan_agent(user_input: str) -> list:
    response = chain.invoke({"input": user_input})

    if isinstance(response, dict) and "text" in response:
        response_text = response["text"]
    else:
        response_text = str(response)

    subtasks = [line.strip("-• \n") for line in response_text.split("\n") if line.strip()]
    return subtasks




