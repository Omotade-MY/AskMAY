
import os
import pypdf
#os.environ["LANGCHAIN_TRACING"] = "true"

# Import all language models and tools fro langchain
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.agents import create_csv_agent, AgentType
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

os.environ['OPENAI_API_KEY'] = "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"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

csv_files = []

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





file = input('Enter file name: ')

# No async implementation in the Pinecone client, fallback to sync
docsearch = get_docsearch(file)
retriever=docsearch.as_retriever(max_tokens_limit=2000)

# create memory 

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
llm = ChatOpenAI(temperature=0, streaming=True)
# create chains tools

llm_csv_agent_chain = create_csv_agent(
    llm,
    file,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
llm_retrieval_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm = llm,
    chain_type = 'stuff',
    retriever = retriever
)
llm_converse_chain = ConversationalRetrievalChain.from_llm(
    llm,
    chain_type="stuff",
    retriever=retriever,
)

tools = [

    #Tool(
     #   name="converse_retrieve_agent",
      #  func=llm_converse_chain,
       # description="This tool should be used first. Useful for conversation and giving descriptive answers and definitions."
    #),
    
    #Tool(
    #    name="converse_retrieve_agent",
    #    func=llm_retrieval_chain,
    #    description="This tool should be used when you need to answer questions from the documents."
    #),

    Tool(
        name="csv_agent",
        func=llm_csv_agent_chain.run,
        description="useful for when you need to give statistics. Example questions could be 'who is the first'"
    ),    


]


agent = initialize_agent(tools, llm, 
                         agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, memory=memory,
                         handle_parsing_errors=True)

msg = f" `{file}` processed. Here you go. Ask me about your data"

while True:
    message = input('User:> ')
    answer = agent.run(message)

    #answer = res["answer"]
    print('Chatbot:> ', answer)
