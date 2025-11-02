import os
from dotenv import load_dotenv
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv('.env')
api_key = os.getenv('MISTRALAI_API_KEY')

embedding = MistralAIEmbeddings(model='mistral-embed' , api_key=api_key)

query_result = embedding.embed_query('저는 배가 고파요')

# print(query_result)

data = [
    '주식 시장이 급등했어요' , '시장 물가가 올랐어요' , '전통 시장에는 다양한 물품들을 팔아요' , '부동산 시장이 점점 더 복잡해지고 있어요' , '저는 빠른 비트를 좋아해요' , '최근 비트코인 가격이 많이 변동했어요'
]

df = pd.DataFrame(data ,columns=['text'])


def get_embedding(text):
    return embedding.embed_query(text)

df['embedding'] = df['text'].apply(get_embedding)

# print(df.head())

def cos_sim(A,B):
    return dot(A,B) / (norm(A)*norm(B))

def return_answer_candidate(df, query):
    query_embedding = get_embedding(query)
    
    df['similarity'] = df['embedding'].apply(lambda x : cos_sim(np.array(x) , np.array(query_embedding)))
    
    top_three_doc = df.sort_values("similarity" , ascending=False).head(3)
    
    return top_three_doc

sim_result = return_answer_candidate(df, '과일 값이 비싸다')

# print(sim_result)

from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings

embeddings = HuggingFaceBgeEmbeddings(model_name='BAAI/bge-m3')

def get_embedding(text):
    return embeddings.embed_query(text)

data = [
    '주식 시장이 급등했어요' , '시장 물가가 올랐어요' , '전통 시장에는 다양한 물품들을 팔아요' , '부동산 시장이 점점 더 복잡해지고 있어요' , '저는 빠른 비트를 좋아해요' , '최근 비트코인 가격이 많이 변동했어요'
]

df = pd.DataFrame(data , columns=['text'])

df['embedding'] = df['text'].apply(get_embedding)

print(df.head())

sim_result = return_answer_candidate(df, '과일 값이 비싸다')

print(sim_result)