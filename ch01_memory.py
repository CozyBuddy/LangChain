from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from dotenv import load_dotenv

load_dotenv('.env')
import os
api_key = os.getenv('MISTRALAI_API_KEY') 

model = ChatMistralAI(model="mistral-small-latest" , api_key=api_key , temperature=0.0)

prompt = ChatPromptTemplate.from_messages(
    [('system' , '당신은 금융 상담사입니다. 사용자에게 최선의 금융 조언을 제공합니다.') ,
     ('placeholder' , "{messages}")]
)

chain = prompt | model

# ai_msg = chain.invoke({
#     "messages" : [
#         ('human' , "저축을 늘리기 위해 무엇을 할 수 있나요?") ,
#         ('ai' , '저축 목표를 설정하고 , 매달 자동 이체로 일정 금액을 저축하세요') , 
#         ('human' , "아니 방금 너 뭐라고 했는지 다시 말해봐")
#     ]
# })


# print(ai_msg.content)

from langchain_community.chat_message_histories import ChatMessageHistory

chat_history = ChatMessageHistory()

chat_history.add_user_message('저축을 늘리기 위해 무엇을 할 수 있나요?')
chat_history.add_ai_message('저축 목표를 설정하고 ,매달 자동이체로 일정 금액을 저축하세요')
chat_history.add_user_message('너 방금 뭐라고 했나요?')

# ai_response = chain.invoke({'messages' : chat_history.messages})

# print(ai_response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

prompt = ChatPromptTemplate.from_messages(
    [
        ('system' , "당신은 금융 전문가입니다. 모든 질문에 최선을 다해 답변하십시오") ,
        ('placeholder' , "{chat_history}") ,
        ('human' , "{input}")
    ]
)

chat_history = ChatMessageHistory()
chain = prompt | model

chain_with_message_history = RunnableWithMessageHistory(
    chain , lambda session_id : chat_history ,
    input_messages_key="input" , history_messages_key="chat_history"
)

# print(chain_with_message_history.invoke(
#     {'input' : "저축을 늘리기 위해 무엇을 할 수 있나요?"} ,
#     {'configurable' : {'session_id' : "unused"}}
# ).content)

# print(chain_with_message_history.invoke(
#     {'input' : "내가 방금 뭐라고 질문했나요?"},
#     {'configurable' : {'session_id' : "unused"}}
# ).content)

from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

trimmer = trim_messages(strategy='last' , max_tokens=2 , token_counter=len)

chain_with_trimming = (
    RunnablePassthrough.assign(chat_history=itemgetter('chat_history') | trimmer) | prompt | model
)

chain_with_trimmed_history = RunnableWithMessageHistory(
    chain_with_trimming , lambda session_id : chat_history , input_message_key='input' , history_messages_key='chat_history'
)

print(chain_with_trimmed_history.invoke({
    'input' : "저는 5년 내에 집을 사기 위해 어떤 재정 계획을 세워야 하나요?"
}, {'configurable' : {'session_id' : 'finance_session_1'}}).content)

# print(chain_with_trimmed_history.invoke({
#     'input' : "내가방금 뭐라고 했더라? "
# }, {'configurable' : {'session_id' : 'finance_session_1'}}).content)


def summarize_messages(chain_input):
    stored_messages = chat_history.messages
    if len(stored_messages) == 0 :
        return False
    
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            ('placeholder' , "{chat_history}") ,
            ('user' , "이전 대화를 요약해주세요 .가능한 한 많은 세부 정보를 포함하십시오")
        ]
    )
    
    summarization_chain = summarization_prompt | model
    summary_message = summarization_chain.invoke({'chat_history' : stored_messages})
    
    chat_history.clear()
    chat_history.add_message(summary_message)
    
    return True

chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages) |chain_with_message_history
)

print(chain_with_summarization.invoke({'input' : "저에게 어떤 재정적 조언을 해주셨나요?"}, {'configurable' : {'session_id' : "unused"}}).content)
