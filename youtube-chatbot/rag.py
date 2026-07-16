import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableParallel,RunnableSequence
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from utils import extracting_video_id

load_dotenv()

def load_vector_store(video_id):
    index_path = f"data/indexes/{video_id}"

    if not os.path.exists(index_path):
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store

def create_vector_store(documents, video_id):
    # splitting into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    # creating embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks,embedding=embeddings)
    
    index_path = f"data/indexes/{video_id}"
    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)
    return vector_store,len(chunks)

def format_doc(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def ask_question(vector_store, question):
    # retriver
    retriever = vector_store.as_retriever(search_type = "similarity",search_kwargs={'k':4})
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.2)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        ANSWER ONLY from provided transcript context.
        If the context is insufficient, just say you don't know.
        
        {context}
        Question : {question}
        """,
        input_variables=['context','question']
    )

    retrieved_docs = retriever.invoke(question)
    context = format_doc(retrieved_docs=retrieved_docs)
    
    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })
    
    answer = llm.invoke(final_prompt)
    return answer.content, retrieved_docs