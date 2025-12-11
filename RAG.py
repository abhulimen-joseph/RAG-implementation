from dotenv import load_dotenv
import os 
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from fastapi import FastAPI
from pydantic import BaseModel

# this is to load the api key from the .env file 
# load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLe_API_KEY is not found in the (.env) file")

# this is to configure the google api key  
genai.configure(api_key=API_KEY)

# this is to load the pdf 
loader = PyPDFLoader("41488-30243-PB.pdf")
docs = loader.load()

# this is for chunking 
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap= 200)
chunks = splitter.split_documents(docs)

#embeddings
embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

#Vector DB
db = Chroma.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_type ="similarity", k = 5)

#llm
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

#Rag chain
rag_chain =RetrievalQA.from_chain_type(llm=llm, retriever= retriever, chain_type = "stuff")

app = FastAPI(title = "RAG chatbot API")

#fastapi application
class Query(BaseModel):
    question: str

@app.get("/")
async def root():
    """
    Simple endpoint for API health check. 
    Returns a welcome message and API status.
    """
    return {
        "message": "Welcome to the RAG Chatbot API!",
        "status": "Operational",
        "instructions": "Use the /query endpoint with a POST request to ask questions."
    }

@app.post("/query")
async def process_query(query:Query):
    "Accepts all questions and gives back the answer"

    try:

        try:
            raw_output = rag_chain.run(query.question)
            print("RAG raw_output (from .run()):", raw_output)
            return {"question": query.question, "summary": raw_output}
            
        except Exception:
            # Fallback: use .invoke() which may return a dict
            raw_result = rag_chain.invoke({"query": query.question})
            print("RAG raw_result (dict/object):", raw_result)

            # If dict, look for common answer fields
            if isinstance(raw_result, dict):
                for key in ["result", "results", "answer", "output_text", "text"]:
                    if key in raw_result:
                        return {"question": query.question, "summary": raw_result[key]}

                # Last fallback
                return {"question": query.question, "summary": str(raw_result)}

            # If it's a string
            return {"question": query.question, "summary": str(raw_result)}
    except Exception as e:
        return{"error": str(e)}
    


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server")
    uvicorn.run("RAG:app", host= "0.0.0.0", port=8000, reload= True)

   
