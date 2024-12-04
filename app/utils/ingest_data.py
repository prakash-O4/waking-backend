import os
import pickle
import pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.pinecone import Pinecone
from langchain_community.document_loaders import (PyPDFLoader,
                                                  UnstructuredFileLoader)
from langchain_openai import OpenAIEmbeddings

openai_api_key = os.getenv("OPENAI_API_KEY")

def init_pinecone(path):
    try:
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),environment="gcp-starter")
        # index = pinecone.Index(index_name="kanun")
        print("Creating a new vector store...")
        # Load the PDF document into a vector store
        loader = PyPDFLoader(file_path=path)
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
        )
        documents = loader.load_and_split(text_splitter=text_splitter)

        embeddings = OpenAIEmbeddings()
        _ = Pinecone.from_documents(documents, embeddings, index_name='wakil-g')
        print("successfully embedded")
    except Exception as e:
        print(e)



def create_or_load_vectorstore(path, vectorstore_filename):
    if not os.path.exists(vectorstore_filename):
        print("Creating a new vector store...")
        # Load the PDF document into a vector store
        loader = UnstructuredFileLoader(path, mode='paged')
        raw_documents = loader.load()

        # Split the text into documents
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
        )
        documents = text_splitter.split_documents(raw_documents)

        # Create a vector store with the documents and metadata
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Save the vector store to a file
        with open(vectorstore_filename, "wb") as f:
            pickle.dump(vectorstore, f)
    else:
        print("Loading existing vector store...")
        with open(vectorstore_filename, "rb") as f:
            vectorstore = pickle.load(f)
        
        # Load the PDF document into a vector store
        loader = UnstructuredFileLoader(path, mode='paged')
        raw_documents = loader.load()

        # Split the text into documents
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
        )
        documents = text_splitter.split_documents(raw_documents)

        # Add the documents to the existing vector store
        vectorstore.add_documents(documents)

        # Save the updated vector store to a file
        with open(vectorstore_filename, "wb") as f:
            pickle.dump(vectorstore, f)


def embed_text(text):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_texts(text,embedding=embeddings)
    FAISS.similarity_search_by_vector()
    return vector

init_pinecone('source/civil_code.pdf')

# def testPdfLoader():
#     loader = PyPDFLoader(file_path='source/Labor law.pdf')
#     text_splitter = CharacterTextSplitter(
#             separator="\n",
#             chunk_size=600,
#             chunk_overlap=100,
#             length_function=len,
#         )
#     pages = loader.load_and_split(text_splitter=text_splitter)
#     print(pages[0])

# testPdfLoader()