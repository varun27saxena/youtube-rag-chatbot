import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

# extracting video id




    
    

# Indexing

youtube_url = input("Paste YouTube URL: ")

video_id = extracting_video_id(youtube_url)

if not video_id:
    print("Invalid YouTube URL")
    exit()

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id,languages=["en"])

    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript)
except TranscriptsDisabled:
    print("No transcript found")
    

# splitter    
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# print(len(chunks))

# embedding

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks,embedding=embeddings)

# retriver
retriever = vector_store.as_retriever(search_type = "similarity",search_kwargs={'k':4})
# print(retriever.invoke("what is the best practices used in software engineering ?"))

# Augmentation

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

question = "is the topic of aliens discussion in this video? If yes then what was discussed"
retrieved_docs = retriever.invoke(question)

def format_doc(retrieved_docs):
    content_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return content_text
# answer = llm.invoke(final_prompt)
# print(answer.content)

# chain 
parallel_chain = RunnableParallel({
    'context':retriever|RunnableLambda(format_doc),
    'question':RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

result = main_chain.invoke("breakdown his salary like ctc")
print(result)