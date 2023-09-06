
import os
import pypdf
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.agents import create_csv_agent, AgentType
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_sql_agent 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 
from langchain.llms import OpenAI

os.environ["LANGCHAIN_TRACING"] = "true"
username = 'postgres'
host = 'localhost'
port = '5432'
mydatabase = 'postgres'
password = 'Assalaam'

pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"
os.environ['OPENAI_API_KEY'] = "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"

memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
llm = OpenAI(temperature=0,model="text-davinci-003", streaming=True)


from SQLagent import build_sql_agent, sql_as_tool
from csv_chat import build_csv_agent, csv_as_tool
llm = OpenAI(temperature=0,model="text-davinci-003", streaming=True)
from utility import process_csv_file
file_paths = process_csv_file('ChatGPT_Learning_Data.xlsx')
file_paths.append(process_csv_file('namesCopy.csv'))

sql_agent = build_sql_agent(llm=llm)
csv_agent = build_csv_agent(llm=llm, file_path=file_paths)
tools = [
    csv_as_tool(csv_agent),
    sql_as_tool(sql_agent),
        ]
    
agent = initialize_agent(
    tools = tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory = memory
    )

agent.run('what sectors are available?')