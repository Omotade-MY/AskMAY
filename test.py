import os

#pip install pypdf
#export HNSWLIB_NO_NATIVE = 1

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType, create_csv_agent
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


os.environ['OPENAI_API_KEY'] = "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"


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""


def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile(delete=False) as tempfile:
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        documents = loader.load()
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
    await cl.Message(content="You can now chat with your pdfs.").send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
            max_files=10,
        ).send()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(temperature=0, streaming=True)

    for file in files:

        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()

        # No async implementation in the Pinecone client, fallback to sync
        docsearch = await cl.make_async(get_docsearch)(file)

        llm_retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type = 'stuff',
        retriever = docsearch.as_retriever(max_tokens_limit=4097),
        memory = memory
        )

        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()


    tools = [
    
            
            Tool.from_function(
                name="llm_retrieve_agent",
                func=llm_retrieval_chain.run,
                description="This tool should be used when you need to answer questions from the documents. If request is not a question then don't use this tool",

            )
    ]

    # Use the agent tool.
    chain = initialize_agent(tools=tools, llm=llm, 
                        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                        verbose=True, 
                        memory=memory,
                        handle_parsing_errors=True)

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.run(message, callbacks=[cb])

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