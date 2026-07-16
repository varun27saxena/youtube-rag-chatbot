# YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about YouTube videos using their transcripts.

The project fetches a video's transcript, splits it into meaningful chunks, converts them into embeddings, stores them in a FAISS vector database, and retrieves relevant context before generating an answer with an LLM.

> Note: The selected video must have an accessible English transcript.

## Features

- Fetches YouTube video transcripts using a video ID
- Splits long transcripts into smaller chunks
- Generates embeddings with OpenAI
- Performs semantic search with FAISS
- Answers questions using only relevant transcript context
- Uses a context-grounded prompt to reduce hallucinations

## Tech Stack

- Python
- LangChain
- OpenAI API
- FAISS
- YouTube Transcript API
- python-dotenv

## RAG Pipeline

```text
YouTube Video ID
      ↓
Fetch Transcript
      ↓
Split into Chunks
      ↓
Create Embeddings
      ↓
Store in FAISS Vector Database
      ↓
Retrieve Relevant Chunks
      ↓
LLM Generates a Context-Based Answer
```

## Installation

Clone the repository:

```bash
git clone https://github.com/varun27saxena/youtube-rag-chatbot.git
cd youtube-rag-chatbot
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install youtube-transcript-api langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu python-dotenv
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Never upload `.env` or an API key to GitHub.

## Usage

Set the YouTube video ID in `app.py`:

```python
video_id = "PHpsdIHpLUE"
```

Run the project:

```bash
python app.py
```

Ask a question through the RAG chain:

```python
result = main_chain.invoke("Can you summarize this video?")
print(result)
```

## Example Questions

```text
Can you summarize this video?
What are the main topics discussed?
What does the speaker say about software engineering?
Is the topic of aliens discussed in this video?
```

## Project Structure

```text
youtube-rag-chatbot/
├── app.py
├── .env
├── .gitignore
└── README.md
```

## Future Improvements

- Add timestamp-based citations for answers
- Build a Streamlit web interface
- Build a Chrome or Brave extension
- Cache vector indexes for previously processed videos
- Support multiple transcript languages
- Add chat history

## Author

Varun Saxena
