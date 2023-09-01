import os
import pypdf
os.environ["LANGCHAIN_TRACING"] = "true"

# Import all language models and tools fro langchain
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.agents import create_csv_agent, AgentType, initialize_agent, Tool
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse

from langchain.agents import create_sql_agent 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 

os.environ['OPENAI_API_KEY'] = "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"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo-0613", streaming=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()
username = 'postgres'
host = 'localhost'
port = '5432'
mydatabase = 'postgres'
password = 'Assalaam'

pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"

db = SQLDatabase.from_uri(pg_uri)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

llm_sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

while True:
    message = input('User:> ')
    try:
        response = llm_sql_agent.run(input=message)
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse tool input: "):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")

    print('Chatbot:> ', response)