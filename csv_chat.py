import os

#pip install pypdf
#export HNSWLIB_NO_NATIVE = 1

#os.environ["LANGCHAIN_TRACING"] = "true"
import langchain
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_csv_agent, AgentType
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_sql_agent 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 
from langchain import PromptTemplate




os.environ['OPENAI_API_KEY'] = "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"

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""
from utility import process_csv_file
#llm = OpenAI(temperature=0, model="text-davinci-003")
#'namesCopy.csv', 
#file_paths = process_csv_file('ChatGPT_Learning_Data.xlsx')

def build_csv_agent(llm, file_path):
    assert isinstance(file_path, list)
    if len(file_path) == 1:
         file_path = file_path[0]
    

    csv_agent = create_csv_agent(
        llm,
        file_path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return csv_agent

#csv_agent = build_csv_agent(llm, file_paths)
#message = "What are the sectors available"
#csv_agent.run(input=message)

def csv_as_tool(agent):
     return Tool.from_function(
                    name = "csv_retrieval_tool",
                    func= agent.run,
                    description= 'This tool useful for statistics, calculations, plotting and as well as data aggregation'
                )
"""while True:
    message = input('User:> ')
    try:
        response = 
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse tool input: "):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")

    print('Chatbot:> ', response)"""
