
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
os.environ["LANGCHAIN_TRACING"] = "true"
username = 'postgres'
host = 'localhost'
port = '5432'
mydatabase = 'postgres'
password = 'Assalaam'

pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"
os.environ['OPENAI_API_KEY'] = "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"

memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo-0613", streaming=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

global csv_files
csv_files = []

class ExcelLoader():
    def __init__(self, file):
        import pandas as pd
        self.status = False
        self.name =  'ExcelLoader'
        self.file = file
        self.loader = pd.ExcelFile
        self.ext = ['xlsx']
    
    def load(self):
        from langchain.document_loaders.csv_loader import CSVLoader

        ssheet = self.loader(self.file)
        try:
            os.mkdir('temp')

        except FileExistsError:
            pass
        docs = []
        from randomize import randomName
        for i,sheet in enumerate(ssheet.sheet_names):
            df = ssheet.parse(sheet)
            #file_name = randomName()
            temp_path = f'./temp/{sheet}.csv'
            csv_files.append(temp_path)
            df.to_csv(temp_path, index=False)
            loader = CSVLoader(temp_path)
            doc = loader.load()
            docs.extend(doc)
            #os.remove(temp_path)

        return docs


def process_file(file: AskFileResponse):
    import tempfile

    if file.split('.')[-1] == "txt":
        Loader = TextLoader
    elif file.split('.')[-1] == "pdf":
        Loader = PyPDFLoader
    elif file.split('.')[-1] == "csv":
        Loader = CSVLoader
        csv_files.append(file)

    elif file.split('.')[-1] == "xlsx":
        Loader = ExcelLoader

    else:
        raise ValueError('File type is not supported')
    
    
    #print('This is what file holds', file.content)
    loader = Loader(file)
    documents = loader.load()

    docs = text_splitter.split_documents(documents)


    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    
    return docs
    
def get_docsearch(file: AskFileResponse):
    
    docs = process_file(file)
    
    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch


files = input('Enter file name: ')

# No async implementation in the Pinecone client, fallback to sync
files = files.split(',')
csv_files = []
for file in files:
    file = file.strip()
    docsearch = get_docsearch(file)
    retriever=docsearch.as_retriever(max_tokens_limit=2000)

    
    llm_retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type = 'stuff',
        retriever = retriever,
        memory = memory
    )
    llm_qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=docsearch.as_retriever(max_tokens_limit=4097),
        )


if len(csv_files) == 1:
    csv_files = csv_files[0]

# Create the CSV agent.
llm_csv_agent_chain = create_csv_agent(
    llm,
    csv_files,
    verbose=False,
    handle_parsing_errors=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )


db = SQLDatabase.from_uri(pg_uri)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

llm_sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
tools = [
    
        # Create the CSV agent tool.
    Tool.from_function(
            name="csv_agent",
            func=llm_csv_agent_chain.run,
            description="useful for when you need to give statistics. Example questions could be 'what is the the total number of rows'. Use this tool when you are asked to find things like average, most, highest etc. Use this function first if you need to anser question from a CSV document",

        
        ),
    
    Tool.from_function(
        name="converse_retrieve_agent",
        func=llm_qa_chain,
        description="This tool should be used when you need to answer questions from other documents and also for chatting and conversing and responding to greetings",
    ),
    Tool.from_function(
        name = 'sql_database_query_agent',
        func= llm_sql_agent.run,
        description= 'Use this tool when you need run a query against a database. You should use this tool if you did not get an answer after using llm_csv_agent'
    )
]


# Use the agent tool.
agent = initialize_agent(
                         tools=tools, llm=llm, 
                         agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, 
                         memory=memory,
                         handle_parsing_errors=True)

while True:
    message = input('User:> ')
    try:
        response = agent.run(input=message)
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse tool input: "):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")

    print('Chatbot:> ', response)
    ##Errors to handle
    # 1. APIError
