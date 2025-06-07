import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class LLM:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"), temperature=0)

    def query(self, system_prompt: str, user_query: str, input: dict = {}) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_query)
        ])
        chain = prompt | self.llm
        response = chain.invoke(input).content
        return response
    
    def query_json(self, system_prompt: str, user_query: str) -> dict:
        response = self.query(system_prompt, user_query)
        return eval(response)
