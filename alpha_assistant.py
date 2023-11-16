import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

st.set_page_config(page_title="Alpha Assistant", page_icon=":speech_balloon:")
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
def chunk_data(data, chunk_size, chunk_overlap=20):
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

    llm = ChatOpenAI(model=MODEL, openai_api_key=api_key ,temperature=1)
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
        MODEL_LIST = ["gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4","gpt-4-1106-preview	"]
        MODEL = st.selectbox('Select Model :', MODEL_LIST)
        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        #k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')
                index_name = 'askadocument'
                # creating the embeddings and returning the Chroma vector store
                vector_store = insert_or_fetch_embeddings(index_name)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # Check if 'vs' exists in session state
    if 'vs' not in st.session_state:
        st.session_state.vs = None

    # Check if 'messages' exists in session state, if not initialize it
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

    # # Check if 'k' exists in session state, if not initialize it
    # if 'k' not in st.session_state:
    #     st.session_state.k = None

    # Display the chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User inputs a question at the bottom of the interface
    q = st.text_input('Ask a question about the content of your file:', key='user_input')

    # Submit button
    if st.button('Submit'):
        if q:  # If the user entered a question and hit submit
            # Display the user's question in the chat interface
            st.session_state.messages.append({"role": "user", "content": q})
            
            if st.session_state.vs:  # Check if the vector store exists
                vector_store = st.session_state.vs
                #st.write(f'k: {st.session_state.k}')  # Display the value of k from session state

                # Call your function to get the model's response
                answer = ask_and_get_answer(vector_store, q, 3)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Clear the user input after submitting
                #st.session_state.user_input = ""

                # Display the updated chat messages
                st.text("")  # Spacer
                for msg in st.session_state.messages:
                    st.chat_message(msg["role"]).write(msg["content"])

    # run the app: streamlit run ./chat_with_documents.py

