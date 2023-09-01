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


os.environ['OPENAI_API_KEY'] = "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"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

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

welcome_message = """Welcome to the **AskMAY** ! To get started:
1. Upload a PDF or text file
2. Ask a question about the file you uploaded
3. Save yourself of having to go through a whole document to get a specific information
"""
global csv_files
csv_files = []

def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    elif file.type == "text/csv":
        Loader = CSVLoader
        csv_files.append(file)

    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        Loader = ExcelLoader

    else:
        raise ValueError('File type is not supported')
    
    tempfile_name = './tempfile.txt'
    with open(tempfile_name, 'wb') as tmpfile:
        
        tmpfile.write(file.content)
    #print('This is what file holds', file.content)
    loader = Loader(tempfile_name)
    documents = loader.load()
    
    os.remove(tempfile_name)
    docs = text_splitter.split_documents(documents)


    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    
    return docs
    

    

def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)
    # Create a unique namespace for the file

    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch



@cl.on_chat_start
async def start():
    # Sending an image with the local file path
    await cl.Message(content="You are just a step away from interracting with your document.").send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "text/csv", "application/pdf", ".xlsx"],
            max_size_mb=200,
            timeout=180,
            max_files= 10,
        ).send()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(temperature=0, streaming=True)
    csv_files = []

    for file in files:    
        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()

        # No async implementation in the Pinecone client, fallback to sync
        docsearch = await cl.make_async(get_docsearch)(file)

        retriever = docsearch.as_retriever(max_tokens_limit=4000)
        
        llm_retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm = llm,
            chain_type = 'stuff',
            retriever = retriever,
            memory = memory
        )
        

        # Let the user know that the system is ready
        msg.content = f"`{file.name}` processed"
        await msg.update()
    
    if len(csv_files) == 1:
        csv_files = csv_files[0]

    llm_csv_agent_chain = create_csv_agent(
            llm,
            csv_files,
            verbose=True,
            handle_parsing_errors=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    

    tools = [
    
        # Create the CSV agent tool.
            Tool.from_function(
                   name="csv_agent",
                  func=llm_csv_agent_chain.run,
                 description="useful for when you need to give statistics. Example questions could be 'who is the the total number of rows'",

                
             ),
            
            Tool.from_function(
                name="converse_retrieve_agent",
                func=llm_retrieval_chain.run,
                description="This tool should be used when you need to answer questions from the documents and also for chatting and conversing",
            ),

                ]
        
    # Use the agent tool.
    chain = initialize_agent(tools=tools, llm=llm, 
                        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                        verbose=True, 
                        memory=memory,
                        handle_parsing_errors=True)

        

    msg.content = f"all `{len(files)}` files processed. Here you go. Ask me about your data"
    await msg.update()


    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    print('I reached here   1!!!')

    res = await chain.acall(message, callbacks=[cb])
    print('I reached here!!!')
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the documents from the user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()