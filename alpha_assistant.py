import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pandas as pd
from datetime import datetime
from pyspark.sql import Row
from databricks import sql
import os
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="Alpha Assistant", page_icon=":speech_balloon:")
uri = (
    "databricks://token:dapic3e9dd1a6924fd69f15dd90f6c9c35d6@dbc-eb788f31-6c73.cloud.databricks.com?"
    "http_path=/sql/1.0/warehouses/21491dc99c22a788&catalog=alpha_assistant&schema=default"
)
db = SQLDatabase.from_uri(uri)

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size, chunk_overlap):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# # create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
# def create_embaeddings(chunks):
#     embeddings = OpenAIEmbeddings()
#     vector_store = Chroma.from_documents(chunks, embeddings)
#
#     # if you want to use a specific directory for chromadb
#     # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
#     return vector_store

def insert_or_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    pinecone.init(api_key='bcd7d6fc-9460-458f-aab5-7be1265596f5', environment='gcp-starter')
    
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store

def delete_pinecone_index(index_name='all'):
    import pinecone
    pinecone.init(api_key='bcd7d6fc-9460-458f-aab5-7be1265596f5', environment='gcp-starter')

    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Ok')

def ask_and_get_answer(vector_store, q, k):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model=model, openai_api_key=api_key ,temperature=temperature)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env

    st.subheader('Alpha Assistant 🤖')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if not api_key:
          st.warning("Please input your OpenAI API key.")
        MODEL_LIST = ["AlphaGPT","AlphaJunior","AlphaSenior"]
        MODEL_LIST = ["AlphaGPT","gpt-3.5-turbo-1106","gpt-4-1106-preview"]
        MODEL = st.selectbox('Select Model :', MODEL_LIST)
        if MODEL == "AlphaJunior":
            model = "gpt-3.5-turbo-1106"
        if MODEL == "AlphaSenior":
            model = "gpt-4-1106-preview"
        if MODEL != "AlphaGPT":
        # file uploader widget
            uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

            # chunk size number widget
            chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
            chunk_overlap = st.number_input('Chunk Overlap:', min_value=100, max_value=1000, value=100, on_change=clear_history)
            temperature = st.number_input('Temperature:', min_value=0.0, max_value=1.0,value=0.7,step=0.1, on_change=clear_history)
            # k number input widget
            k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

            # add data button widget
            add_data = st.button('Add Data', on_click=clear_history)

            if uploaded_file and add_data: # if the user browsed a file
                with st.spinner('Reading, chunking and embedding file ...'):

                    # writing the file from RAM to the current directory on disk
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('./', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)

                    data_file = load_document(file_name)
                    chunks = chunk_data(data_file, chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                    st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                    tokens, embedding_cost = calculate_embedding_cost(chunks)
                    st.write(f'Embedding cost: ${embedding_cost:.4f}')
                    index_name = 'askadocument'
                    # creating the embeddings and returning the Chroma vector store
                    vector_store = insert_or_fetch_embeddings(index_name)

                    # saving the vector store in the streamlit session state (to be persistent between reruns)
                    st.session_state.vs = vector_store
                    st.success('File uploaded, chunked and embedded successfully.')
        if MODEL == "AlphaGPT":
            VARIANCE_LIST = ["SmartSaver","PerformancePlus"]
            VARIANCE = st.sidebar.selectbox('Select Variance :', VARIANCE_LIST)
            if VARIANCE =="SmartSaver":
                model_name='gpt-3.5-turbo'
            if VARIANCE =="PerformancePlus":
                model_name='gpt-4'
            temperature = st.sidebar.number_input('Temperature:', min_value=0.0, max_value=1.0,value=0.7,step=0.1, on_change=clear_history)

    # Check if 'vs' exists in session state
    if 'vs' not in st.session_state:
        st.session_state.vs = None

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
        if MODEL !='AlphaGPT':
            with st.spinner("Thinking..."):
                answer = ask_and_get_answer(st.session_state.vs, prompt, k)
            if answer is None or not answer.strip():
                st.warning("Sorry, this is out of my knowledge domain. Please shorten or rephrase the question to try again.")
        else:
            with st.spinner("Thinking..."):
                db_chain = SQLDatabaseChain.from_llm(ChatOpenAI(openai_api_key=api_key,temperature=temperature, verbose=True,model_name=model_name), db)
                answer = db_chain.run(prompt)
            if answer is None or not answer.strip():
                st.warning("Sorry, this is out of my knowledge domain. Please shorten or rephrase the question to try again.")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
        conversation_history={'datetime':current_datetime,'input':prompt,'response':answer}
        result_tuple = (conversation_history['datetime'], conversation_history['input'], conversation_history['response'])

        from databricks import sql
        import os
        
        with sql.connect(server_hostname = "dbc-eb788f31-6c73.cloud.databricks.com",
                         http_path = "/sql/1.0/warehouses/21491dc99c22a788",
                         access_token = "dapid039ed9f3529c6eaa50579c54a8d6814") as connection:
        
          with connection.cursor() as cursor:
        
            cursor.execute(f"INSERT INTO alpha_assistant.llm_chat_history.llm_model_request_history VALUES {result_tuple}")
      
    
    # run the app: streamlit run ./chat_with_documents.py

