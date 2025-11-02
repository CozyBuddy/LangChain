from dotenv import load_dotenv
load_dotenv('.env')
import os
api_key = os.getenv("MISTRALAI_API_KEY")

print(api_key)

from langchain_mistralai.chat_models import ChatMistralAI

model = ChatMistralAI(model='mistral-medium' ,api_key=api_key)

# print(model.invoke("안녕!"))
# from typing import List
# def call_chat_model(messages : List[dict]):
#     response = model.chat.completions.create(
#         model
#     )
# prompt_template = "주제 {input}에 대해 짧은 설명해주세요"

# def invoke_chain(topic : str):
#     prompt_value = prompt_template.format(topic=topic)
#     messages = [{'role' : 'user' , "content" : prompt_value}]
#     return call_chat_model(messages)

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

prompt = ChatPromptTemplate.from_template(
    "주제 {topic}에 대해 짧은 설명해주세요."
)

output_parser = StrOutputParser()
model = ChatMistralAI(model='mistral-medium' , api_key=api_key)



# print(chain.invoke({'topic' : '더블딥'}))

from langchain_mistralai.chat_models import ChatMistralAI

llm = ChatMistralAI( api_key=api_key,
    temperature=0.5 , max_tokens=500 , model_name='mistral-large-2411'
)

chain = (
    {"topic" : RunnablePassthrough()} | prompt | llm | output_parser
)

print(chain.invoke({'topic' : '더블딥'}))

