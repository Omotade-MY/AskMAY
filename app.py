from SQLagent import build_sql_agent, sql_as_tool
from csv_chat import build_csv_agent, csv_as_tool
from utility import ExcelLoader
# app.py
from typing import List, Union, Optional
from langchain.document_loaders import PyPDFLoader, TextLoader
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant
from PyPDF2 import PdfReader
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

PROMPT_TEMPLATE = """
Use the following pieces of context enclosed by triple backquotes to answer the question at the end.
\n\n
Context:
```
{context}
```
\n\n
Question: [][][][]{question}[][][][]
\n
Answer:"""


def init_page() -> None:
    st.set_page_config(
    )
    st.sidebar.title("Options")
    icon, title = st.columns([3, 20])
    with icon:
        st.image('image.png')
    with title:
        st.title('AskMAY Chatbot')

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content=(
                    "You are a helpful AI QA assistant. "
                    "When answering questions, use the context enclosed by triple backquotes if it is relevant. "
                    "If you don't know the answer, just say that you don't know, "
                    "don't try to make up an answer. "
                    )
            )
        ]
        st.session_state.costs = []

def get_csv_file() -> Optional[str]:
    """
    Function to load PDF text and split it into chunks.
    """
    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        label="Here, upload your csv file you want AskMAY to use to answer",
        type= ["csv", 'xlsx']
    )
    import pandas as pd
    import os
    file = uploaded_file
    if uploaded_file:
        
    
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


        print('FIlE NAME', uploaded_file.type)
        pd.read_csv(uploaded_file).to_csv('temp.csv', index=False)
        
        docs = CSVLoader('temp.csv').load()
        #text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
        documents = text_splitter.split_documents(docs)
        return documents
    else:
        return None

def build_vectore_store(
    docs: str, embeddings: Union[OpenAIEmbeddings, LlamaCppEmbeddings]) \
        -> Optional[Qdrant]:
    """
    Store the embedding vectors of text chunks into vector store (Qdrant).
    """
    if docs:
        with st.spinner("Loading CSV FIle ..."):
            chroma = Chroma.from_documents(
             docs, embeddings
            )
    
        st.success("File Loaded Successfully!!")
    else:
        chroma = None
    return chroma


# Select model 

def select_llm() -> Union[ChatOpenAI, LlamaCpp]:
    """
    Read user selection of parameters in Streamlit sidebar.
    """
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gpt-3.5-turbo-0613",
                                   "gpt-3.5-turbo-16k-0613",
                                   "gpt-4",
                                   "text-davinci-003",
                                   "llama-2-7b-chat.ggmlv3.q2_K",
                                   "llama-2-13b-chat.ggmlv3.q2_K.bin"))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    
    return model_name, temperature


def init_agent(model_name: str, temperature: float) -> Union[ChatOpenAI, LlamaCpp]:
    """
    Load LLM.
    """
    if model_name.startswith("gpt-"):
        llm =  ChatOpenAI(temperature=temperature, model_name=model_name)
    
    elif model_name.startswith("text-dav"):
        llm =  OpenAI(temperature=temperature, model_name=model_name)
    
    elif model_name.startswith("llama-2-"):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
            model_path=f"./models/{model_name}.bin",
            input={"temperature": temperature,
                   "max_length": 2048,
                   "top_p": 1
                   },
            n_ctx=2048,
            callback_manager=callback_manager,
            verbose=False,  # True
        )
    csv_agent = build_csv_agent(llm=llm, file_path='namesCopy.csv')
    sql_agent = build_sql_agent(llm=llm)
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
    return agent, llm

def load_embeddings(model_name: str) -> Union[OpenAIEmbeddings, LlamaCppEmbeddings]:
    """
    Load embedding model.
    """
    if model_name.startswith("gpt-") or model_name.startswith("text-dav"):
        return OpenAIEmbeddings()
    elif model_name.startswith("llama-2-"):
        return LlamaCppEmbeddings(model_path=f"./models/{model_name}.bin")

def get_answer(llm_agent,llm, message) -> tuple[str, float]:
    """
    Get the AI answer to user questions.
    """

    if isinstance(llm, (ChatOpenAI, OpenAI)):
        with get_openai_callback() as cb:
            answer = llm_agent.run(message)
        return answer, cb.total_cost
    #if isinstance(llm, LlamaCpp):
     #   return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0

def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def extract_userquesion_part_only(content):
    """
    Function to extract only the user question part from the entire question
    content combining user question and pdf context.
    """
    content_split = content.split("[][][][]")
    if len(content_split) == 3:
        return content_split[1]
    return content


def set_stage():
    set_stage.has_been_set = True

set_stage.has_been_set = False

def main() -> None:
    _ = load_dotenv(find_dotenv())

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    init_page()
    model_name, temperature = select_llm()
    llm_agent, llm = init_agent(model_name, temperature)

    if not set_stage.has_been_set:
        embeddings = load_embeddings(model_name)
        texts = get_csv_file()
        chroma = build_vectore_store(texts, embeddings)

    init_messages()

    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        if chroma:
            context = [c.page_content for c in chroma.similarity_search(
                user_input, k=10)]
            user_input_w_context = PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["context", "question"]) \
                .format(
                    context=context, question=user_input)
            
        
        st.session_state.messages.append(
            HumanMessage(content=user_input_w_context))
        with st.spinner("ChatGPT is typing ..."):
            answer, cost = get_answer(llm_agent,llm, user_input)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(extract_userquesion_part_only(message.content))

    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


# streamlit run app.py
if __name__ == "__main__":
    main()
