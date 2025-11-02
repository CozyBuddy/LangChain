import os 
os.environ['USER_AGENT'] = 'MyApp/1.0 (Custom Langchain Application)'

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader('https://docs.smith.langchain.com')

loader_multiple_pages = WebBaseLoader(
    ['https://python.langchain.com/docs/introduction/' , 'https://langchain-ai.github.io/langgraph/']
)

# single_doc = loader.load()
# print(single_doc[0].metadata)

docs = loader_multiple_pages.load()

# print(docs[0].page_content)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader

loader = PyPDFLoader("2024_KB_부동산_보고서_최종.pdf")

pages = loader.load_and_split()
# print('청크 수' , len(pages))

# print(pages[10])

loader = PyMuPDFLoader("2024_KB_부동산_보고서_최종.pdf")

pages = loader.load_and_split()
# print("청크수" , len(pages))

# print(pages[10])

loader = PDFPlumberLoader("2024_KB_부동산_보고서_최종.pdf")

# pages = loader.load_and_split()
# print("청크수" , len(pages))

# print(pages[10])

from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredCSVLoader

loader = CSVLoader('서울시_부동산_실거래가정보.csv')

# documents = loader.load()
# print('청크수' , len(documents))

# print(documents[5])

loader = UnstructuredCSVLoader('서울시_부동산_실거래가정보.csv' , mode="elements" ,encoding="cp949")

documents = loader.load()
print('청크수' , len(documents))
print(str(documents[0].metadata)[:500])
print(str(documents[0].page_content)[:500])