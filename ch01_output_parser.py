from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

instructions = parser.get_format_instructions()
print(instructions)

ai_response = '{"이름" : "김철수" , "나이" : 30}'
parsed_response = parser.parse(ai_response)

# print(parsed_response)

from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv('.env')
import os
api_key = os.getenv('GEMINI_KEY')
parser = RetryWithErrorOutputParser.from_llm(parser=JsonOutputParser() , llm=GoogleGenerativeAI(api_key=api_key , model='gemini-2.5-flash'))

question = '가장 큰 대륙은?'
ai_response = '아시아입니다.'

try :
    result = parser.parse_with_prompt(ai_response, question)
    print(result)

except Exception as e :
    print(f'오류 발생: {e}')
    

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
from pydantic import BaseModel , Field , model_validator

model = GoogleGenerativeAI(model='gemini-2.5-flash' , api_key=api_key , temperature=0.0)
api_key2 = os.getenv('MISTRALAI_API_KEY')
model2 = ChatMistralAI(model='mistral-medium' , api_key=api_key2 , temperature=0.0)

from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "다음 질문에 대한 답변이 포함된 JSON 객체를 반환하십시오 : {question}"
)

json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser

# print(list(json_chain.stream({'question' : '비트코인에 대한 짧은 한 문장 설명'})))

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_mistralai.chat_models import ChatMistralAI

from pydantic import BaseModel , Field

model = ChatMistralAI(temperature=0.0 , api_key=api_key2)

class FinancialAdvice(BaseModel):
    setup : str = Field(description='금융 조언 상황을 설정하기 위한 질문')
    advice : str = Field(description='질문을 해결하기 위한 금융 답변')
    

parser = JsonOutputParser(pydantic_object=FinancialAdvice)
prompt = PromptTemplate(
    template="다음 금융 관련 질문에 답변해 주세요. \n{format_instructions}\n{query}\n" ,
    input_variables=['query'] ,
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

chain = prompt | model | parser

print(chain.invoke({'query' : "부동산에 관련하여 금융 조언을 받을 수 있게 질문해라."}))