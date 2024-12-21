from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain and AI imports
from langchain.schema import AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

# Utility imports
import os

# Local imports
from app.utils.helpers import SupabaseHelper

# Load environment variables
load_dotenv()

# Set up environment configurations
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Wakil-G"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def create_advanced_retriever(base_retriever):
    def decompose_query(query: str) -> List[str]:
        """
        Decompose complex legal queries into sub-queries
        Limit to 3 queries to optimize performance
        """
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")  
        
        decomposition_prompt = f"""Decompose the following legal query into 3 specific, 
        more precise sub-queries that can help retrieve targeted information:

        Original Query: {query}

        Decomposed Sub-Queries (maximum 3):"""
        
        try:
            decomposed_queries = llm.invoke(decomposition_prompt).content.split('\n')
            # Ensure we get exactly 3 or fewer queries, removing any empty lines
            decomposed_queries = [q.strip() for q in decomposed_queries if q.strip()][:3]
            return decomposed_queries
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]  # Fallback to original query

    class AdvancedLegalRAGRetriever:
        def __init__(self, retriever):
            self.retriever = retriever

        def invoke(self, query: str) -> List[Document]:
            # Step 1: Query Decomposition
            decomposed_queries = decompose_query(query)
            
            # Step 2: Retrieve documents
            all_docs = []
            unique_docs = set()

            # Retrieve for original query and decomposed queries
            queries_to_search = [query] + decomposed_queries
            for search_query in queries_to_search:
                retrieved_docs = self.retriever.invoke(search_query)
                for doc in retrieved_docs:
                    if doc.page_content not in unique_docs:
                        all_docs.append(doc)
                        unique_docs.add(doc.page_content)

            # Rank and filter documents
            return self.rank_documents(all_docs)[:5]  # Top 5 most relevant documents

        def rank_documents(self, documents: List[Document]) -> List[Document]:
            """
            Custom ranking logic for retrieved documents
            """
            return sorted(
                documents, 
                key=lambda doc: len(doc.page_content), 
                reverse=True
            )

    # Return the advanced retriever
    return AdvancedLegalRAGRetriever(base_retriever)
# Pinecone initialization with advanced retrieval
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def initialize_pinecone():
    """
    Initialize a Pinecone connection and return an advanced retriever object.
    The advanced retriever will decompose queries into sub-queries and retrieve documents
    using the decomposed queries. It will then rank and filter the retrieved documents
    using a custom ranking logic.

    The function will retry up to 3 times with exponential backoff if the Pinecone
    connection fails.
    """
    logger.info("Initializing Pinecone connection")
    index_name = "wakil-g"
    try:
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Create advanced retriever
        advanced_retriever = create_advanced_retriever(base_retriever)
        
        logger.info("Pinecone connection successful with advanced retrieval")
        return advanced_retriever
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        raise

# QA prompt (kept from original implementation)
qa_system_prompt = """You are an AI assistant name of Wakil-G specialized in answering questions about Nepal's laws and constitution. Use the following context to answer the user's question precisely.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# FastAPI models
class QuestionInput(BaseModel):
    question: str

def format_sse(event: str, data: str) -> str:
    # Replace newlines in the data with '__n__'
    formatted_data = data.replace("\n", "__n__")
    # Format the SSE message
    res = f"event: {event}\ndata: {formatted_data}\n\n"
    return res

@app.post("/ask")
async def ask_question(input: QuestionInput, authorization: str = Header(None)):
    try:
        supabase_helper = SupabaseHelper()
        # Verify token and get user ID
        user_id = supabase_helper.get_user_id(authorization)

        # check if user has reached daily quota

        if supabase_helper.check_daily_quota(user_id):
            raise HTTPException(
                status_code=404,
                detail={"message": "You have reached your daily quota. Please try again tomorrow."}
            )
        
        # Get user's chat history
        chat_histories = supabase_helper.get_user_chat_history(user_id)

        # Initialize Pinecone for this request with advanced retrieval
        retriever = initialize_pinecone()
        
        
        # Retrieve relevant documents using advanced retriever
        retrieved_docs = retriever.invoke(input.question)
        
        if retrieved_docs:
            context = " ".join([doc.page_content for doc in retrieved_docs])
            source = retrieved_docs[0].metadata
        else:
            context = "No relevant information found"
            source = "Unknown"

        # Prepare chat history
        chat_history = [
            AIMessage(content=msg)
            for msg in chat_histories
        ]

        # Prepare the prompt
        prompt = qa_prompt.format(context=context, question=input.question, chat_history=chat_history)

        # Initialize ChatOpenAI with streaming
        llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True)

        async def generate_sse():
            yield format_sse("start", json.dumps({"start": True}))

            async for chunk in llm.astream(prompt):
                if chunk.content:
                    yield format_sse("data", chunk.content)
            
            yield format_sse("end", json.dumps({"end": True, "source": source}))

        return StreamingResponse(generate_sse(), media_type="text/event-stream")
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        elif "insufficient_quota" in str(e):
            raise HTTPException(
                status_code=429,
                detail="Oops! Looks like AI funds are running low."
            )
        else:
            logger.error(f"Error in /ask endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to Wakil-G!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)