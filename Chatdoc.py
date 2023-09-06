import os

#pip install pypdf
#export HNSWLIB_NO_NATIVE = 1

#os.environ["LANGCHAIN_TRACING"] = "true"

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.memory import ConversationBufferMemory

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType, create_csv_agent
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_sql_agent 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 
from langchain import PromptTemplate




os.environ['OPENAI_API_KEY'] = "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"


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()


files = ["Assignment 516 Group_4.pdf", 'ENGINEERING_CONTRACT_LAW.pdf']
for file in files:
    docs = []
    if file.split('.')[-1] == "txt":
            Loader = TextLoader
    elif file.split('.')[-1] == "pdf":
        Loader = PyPDFLoader
    loader = Loader(file)
    docs.extend(loader.load())


documents = text_splitter.split_documents(docs)
docsearch = Chroma.from_documents(
        docs, embeddings
    )


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0, streaming=True)

chain = RetrievalQAWithSourcesChain.from_chain_type(
llm,
retriever = docsearch.as_retriever(max_tokens_limit=4097)
)


def run_qa_chain(question):
    results = chain(question)
    return str(results['answer']+'\nSOURCES: ' + results['sources'])

tool = [
     Tool.from_function(
          name = 'llm_retrieval_chain',
          func= run_qa_chain,
          description= "This tol should be used to retriev answers from documents/content"
     )
     ]

agent = initialize_agent(
        tools= tool,
        llm=llm,
        agent= AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_error=True,
        verbose = True,
    )
message = "What are the water regulatory Acts?"


import langchain
while True:
    message = input('User:> ')
    try:
        response = chain.run(message)
    except Exception as e:
        response = str(e)
        if not response.startswith("Could not parse tool input: ") or\
            response.startswith("Could not parse LLM output: "):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    print('Chatbot:> ', response)