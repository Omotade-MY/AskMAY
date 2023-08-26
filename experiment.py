import os
import pypdf


class ExcelLoader():
    def __init__(self, name=None):
        self.ext = 'xlsx'
        self.name =  name
        import pandas as pd
        self.loader = pd.read_excel
    
    def load(self):
        import pandas as pd
        df = self.loader(self.name)
        print(df)
        df.to_csv('./tempfile.csv', index=False)

        loader = CSVLoader('./tempfile.csv')
        docs =  loader.load()

        os.remove('./tempfile.csv')

        return docs

        
# Import all language models and tools fro langchain
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader, unstructured
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse

os.environ['OPENAI_API_KEY'] = "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"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()


welcome_message = """Welcome to the **AskMAY** ! To get started:
1. Upload a PDF or text file
2. Ask a question about the file you uploaded
3. Save yourself of having to go through a whole document to get a specific information
"""

def process_file(file: dict):
    
    loader = ExcelLoader('./names.xlsx')
    documents = loader.load()
    print(documents)
    docs = text_splitter.split_documents(documents)
    return docs

def get_docsearch(file: dict):
    docs = process_file(file)

    # Create a unique namespace for the file
    
    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch

def start():
    # Sending an image with the local file path
    file = {'name':'./names.csv'}
    
    print("Processing doc")

    # No async implementation in the Pinecone client, fallback to sync
    docsearch = get_docsearch(file)
    print(docsearch)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=2000),
    )
    return chain


chain = start()

message = "What number is Muhammad King Yakub on the list? Print out all his details"
res = chain({'question':message})
print(res['answer'])