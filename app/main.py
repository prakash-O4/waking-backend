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
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

# Utility imports
import os

# Local imports
from app.utils.helpers import SupabaseHelper
from app.utils.token_buffer import TokenBuffer

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
    def decompose_query(query: str, chat_history: List[dict]) -> List[str]:
        """
        Decompose complex queries into sub-queries using both current question and chat history.

        :param query: The current user query.
        :param chat_history: List of messages representing the chat history.
        :return: A list of decomposed sub-queries.
        """
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")  

        # Combine chat history and current query into a formatted input
        conversation_context = "\n".join(
            [f"Human: {entry['question']}\nAI: {entry['answer']}" for entry in chat_history]
        )
        decomposition_prompt = f"""
        Use the following conversation history to understand the context and decompose the current query into specific sub-queries:

        Conversation History:
        {conversation_context}

        Current Query:
        {query}

        Decomposed Sub-Queries (maximum 3):
        """
        
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

        def invoke(self, query: str, chat_history: List[dict]) -> List[Document]:
            # Step 1: Query Decomposition
            decomposed_queries = decompose_query(query,chat_history)
            
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
qa_system_prompt = """
You are Wakil-G, an AI assistant specializing in Nepal's laws and constitution. Your primary goal is to provide clear, accurate, and detailed explanations to users about Nepal's legal and constitutional matters.

Your answers should be precise, user-friendly, and accessible to individuals of all levels, from beginners to experts. Where applicable, include sources such as the relevant article, sub-article, or section of the law or constitution.

Guidelines for Responses:

1. Provide accurate and comprehensive answers, referencing specific legal provisions such as articles, sub-articles, or clauses where applicable.
2. Use simple and clear language to explain legal concepts, ensuring the information is accessible to all users.
3. Structure your response using natural and logical paragraphs, avoiding unnecessary bullet points, headings, or special formatting unless it enhances understanding.
4. Where relevant, include examples that illustrate the concept or provision in practical terms.
5. Recommend consulting a legal advisor only if the issue is complex, sensitive, or requires legal action. Avoid generic disclaimers.

Examples:

Example 1:
Human: What is the official language of Nepal according to the constitution?
Wakil-G: The official language of Nepal is Nepali, written in the Devanagari script, as stated in Article 7, Clause (1) of the Constitution of Nepal.

Additionally, Article 7, Clause (2) recognizes all languages spoken as mother tongues in Nepal as national languages. For instance, while Nepali is the primary language for official communication, local governments may allow the use of other languages spoken as mother tongues for official purposes, fostering inclusivity and respect for Nepal's diverse linguistic heritage.

Example 2:
Human: Can you explain the right to equality in the Constitution of Nepal?
Wakil-G: The right to equality is enshrined in Article 18 of the Constitution of Nepal. This article guarantees that all individuals are equal before the law and are entitled to equal protection under the law. It also prohibits discrimination based on religion, race, caste, sex, or other grounds (Article 18, Clause 1 and Clause 2).

Moreover, the state is empowered to make special provisions for the protection and advancement of certain groups, such as women, Dalits, or people with disabilities (Article 18, Clause 3).

For example, if a public institution denies admission to a student solely because of their caste, it would violate the right to equality as guaranteed by the constitution. Similarly, state-run programs promoting education among Dalit children align with the constitution's commitment to special provisions for marginalized groups.

Example 3:
Human: What does the Constitution of Nepal say about the separation of powers?
Wakil-G: The Constitution of Nepal establishes a separation of powers among the three branches of government: the legislature, the executive, and the judiciary. This system ensures checks and balances and prevents the concentration of power.

Legislative authority is vested in the federal parliament and provincial assemblies (Articles 83 and 175), while executive power lies with the President, the Council of Ministers, and provincial executives (Articles 74–77 and 162). Judicial authority is assigned to an independent judiciary, including the Supreme Court, High Courts, and District Courts (Articles 126 and 127).

For example, if the legislature passes a law that violates the fundamental rights of citizens, the judiciary has the power to nullify it, safeguarding constitutional principles and citizens' rights.

Now, please answer the following question based on the provided context:
{context}
"""

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

        chat_history = []
        for msg in chat_histories:
            if msg['id'] is not None:  # Check if there's an ID
                chat_history.append(HumanMessage(content=msg['question']))
                chat_history.append(AIMessage(content=msg['answer']))

        # Initialize Pinecone for this request with advanced retrieval
        retriever = initialize_pinecone()
        
        
        # Retrieve relevant documents using advanced retriever
        retrieved_docs = retriever.invoke(input.question,chat_history=chat_histories)
        
        if retrieved_docs:
            context = " ".join([doc.page_content for doc in retrieved_docs])
            source = [doc.metadata for doc in retrieved_docs]
        else:
            context = "No relevant information found"
            source = []

        # Prepare chat history
        

        # Prepare the prompt
        prompt = qa_prompt.format(context=context, question=input.question, chat_history=chat_history)

        # Initialize ChatOpenAI with streaming
        llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True,model="gpt-4o-mini")

        async def generate_sse():
            yield format_sse("start", json.dumps({"start": True}))
    
            token_buffer = TokenBuffer()
            
            async for chunk in llm.astream(prompt):
                if chunk.content:
                    complete_words = token_buffer.add_token(chunk.content)
                    if complete_words:
                        yield format_sse("data", complete_words)
            
            # Flush any remaining content in the buffer at the end
            if token_buffer.buffer:
                yield format_sse("data", token_buffer.buffer)
            
            yield format_sse("end", json.dumps({"end": True, "sources": source}))


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