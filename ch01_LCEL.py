from langchain_mistralai.chat_models import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv('.env')
api_key = os.getenv('GEMINI_KEY')
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash" , api_key=api_key)

prompt = ChatPromptTemplate.from_template("주제 {topic}에 대해 짧은 설명 부탁한다")

parser = StrOutputParser()

chain = prompt | model | parser

# print(chain.invoke({"topic":'더블딥'}))

# print(chain.batch([{'topic' : "더블딥"} , {'topic' : '인플레이션'}]))

# for token in chain.stream({'topic' : '더블딥'}):
#     print(token , flush=True)
    
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

analysis_prompt = ChatPromptTemplate.from_template("이 대답을 영어로 번역해주세요 : {answer}")

composed_chain = {"answer" : chain} | analysis_prompt | model | StrOutputParser()

# print(composed_chain.invoke({'topic' : "더블딥"})) 

composed_chain_with_lambda = (
    chain | (lambda input : {'answer' : input}) | analysis_prompt | model | StrOutputParser() 
)

# print(composed_chain_with_lambda.invoke({'topic' : '더블딥'}))

composed_chain_with_pipe = (
    chain.pipe(lambda input : {'answer' : input}).pipe(analysis_prompt).pipe(model).pipe(StrOutputParser())
)

# composed_chain_with_pipe.invoke({'topic' : "더블딥"})

from langchain_core.runnables import RunnableParallel

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash' , api_key=api_key)

kor_chain = (
    ChatPromptTemplate.from_template("{topic}에 대해 짧은 설명 해주세요.") |
     model | StrOutputParser()
)

eng_chain = (
    ChatPromptTemplate.from_template("{topic}에 대해 짧게 영어로 설명 부탁해요.") | model | StrOutputParser()
)

parallel_chain = RunnableParallel(kor=kor_chain , eng=eng_chain)

result = parallel_chain.invoke({'topic' : "더블딥"})

print(result['kor'])
print(result['eng'])



