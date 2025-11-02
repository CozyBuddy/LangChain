from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "주제 {topic}에 대해 금융 관련 짧은 조언을 해주세요."
)

print(prompt_template.invoke({'topic' : "투자"}))

from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
prompt_template = ChatPromptTemplate.from_messages([
    ('system' , '당신은 유능한 금융 조언가입니다.') ,
    ('user' , "주제 {topic}에 대해 금융 관련 조언을 해주세요.")
])

# print(prompt_template.invoke({'topic' : '주식'}))

from langchain_core.messages import HumanMessage

prompt_template = ChatPromptTemplate.from_messages([
    ('system' , "당신은 유명한 금융 조언가입니다."),
    MessagesPlaceholder('msgs')
])

print(prompt_template.invoke({'msgs' : [HumanMessage(content='안녕하세요!!')]}))

prompt_template = ChatPromptTemplate.from_messages([
    ('system' , '당신은 유능한 금융 전문가입니다.'),
    ('placeholder' , "{msgs}")
])

print(prompt_template.invoke({'msgs' : [HumanMessage(content='안녕하세요!!')]}))

from langchain_core.prompts import PromptTemplate
example_prompt = PromptTemplate.from_template("질문 : {question}\n답변: {answer}")

examples = [
    {'question' : "주식 투자와 예금 중 어느 것이 더 수익률이 높은가?" ,
     "answer" : """
      후속 질문이 필요한가요 : 네.
      후속 질문 : 주식 투자의 평균 수익률은 얼마인가요?
      중간 답변 : 주식 투자의 평균 수익률은 연 7%입니다.
      후속 질문 : 에금의 평균 이자율은 얼마인가요?
      중간 답변 : 예금의 평균 이자율은 연 1%입니다.
      따라서 최종 답변은 : 주식 투자
     
     """ }, {
         "question" : '부동산과 채권 중 어느 것이 더 안정적인 투자처인가?' ,
         'answer' : """
         후속 질문이 필요한가요 : 네 .
         후속 질문 : 부동산 투자의 위험도는 어느 정도인가요?
         중간 답변 : 부동산 투자의 위험도는 중간 수준입니다.
         후속 질문 : 채권의 위험도는 어느 정도 인가요?
         중간 답변 : 채권의 위험도는 낮은 편입니다.
         따라서 최종 답변은 : 채권
         """
     }
]

# print(example_prompt.invoke(examples[0]).to_string())

from langchain_core.prompts import FewShotPromptTemplate

prompt = FewShotPromptTemplate(
    examples=examples , example_prompt=example_prompt , suffix='질문 : {input}' , input_variables=['input']
)

# print(prompt.invoke({'input' : '부동산 투자의 장점은 무엇인가?'}).to_string())

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv('.env')
import os
api_key = os.getenv('GEMINI_KEY')
os.environ["GOOGLE_API_KEY"] = api_key
embedding =GoogleGenerativeAIEmbeddings(model='gemini-embedding-001' , api_key=api_key)


example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples , embedding , Chroma , k=1
)
question = '부동산 투자의 장점은 무엇인가?'

selected_examples = example_selector.select_examples({'question' : question})

# for example in selected_examples:
#     print('\n')
#     print('입력과 가장 유사한 예제')
#     for k, v in reversed(example.items()):
#         print(f"{k}: {v}")
        
from langchain_core.prompts import FewShotPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

example_prompt = PromptTemplate(
    input_variables=['question' , 'answer'],
    template="질문 : {question}\n 답변 : {answer}"
)

prompt = FewShotPromptTemplate(
    example_selector=example_selector, example_prompt=example_prompt, prefix='다음은 금융 관련 질문과 답변의 예입니다.' , suffix='질문 : {input}\n 답변:' , input_variables=['input']
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash' ,api_key=api_key)

chain = prompt | model

# response = chain.invoke({'input' : '부동산 투자의 장점은 무엇인가요?'})

# print(response.content)


from langchain import hub

# prompt = hub.pull('hardkothari/prompt-maker:c5db8eee')

# print(prompt)



