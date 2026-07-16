import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document

from utils import extracting_video_id, format_timestamp
from rag import create_vector_store, ask_question, load_vector_store


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None


st.title("YouTube RAG Assistant")

youtube_url = st.text_input("Paste a YouTube URL")


if st.button("Process video"):
    video_id = extracting_video_id(youtube_url)

    if not video_id:
        st.error("Please enter a valid YouTube URL.")

    else:
        # Save it for clickable timestamp links later
        st.session_state.video_id = video_id

        try:
            with st.spinner("Loading video..."):
                vector_store = load_vector_store(video_id)

                if vector_store is not None:
                    st.session_state.vector_store = vector_store
                    st.success("Loaded cached video index.")

                else:
                    transcript_list = YouTubeTranscriptApi().fetch(
                        video_id=video_id,
                        languages=["en"]
                    )

                    transcript = " ".join(
                        chunk.text for chunk in transcript_list
                    )

                    documents = [
                        Document(
                            page_content=chunk.text,
                            metadata={
                                "start": chunk.start,
                                "duration": chunk.duration
                            }
                        )
                        for chunk in transcript_list
                    ]

                    vector_store, chunk_count = create_vector_store(
                        documents,
                        video_id
                    )

                    st.session_state.vector_store = vector_store

                    st.success(
                        f"Video processed successfully — {chunk_count} chunks created."
                    )

        except TranscriptsDisabled:
            st.error("This video has disabled transcripts.")

        except Exception as error:
            st.error(f"Could not process the video: {error}")


if st.session_state.vector_store is not None:
    st.info("Video is ready. You can now ask questions.")

    st.subheader("Ask a question")

    question = st.text_input(
        "What would you like to know about this video?"
    )

    if st.button("Ask question"):
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner("Searching the transcript..."):
                answer, sources = ask_question(
                    st.session_state.vector_store,
                    question
                )

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Transcript sources"):
                for source in sources:
                    timestamp = format_timestamp(source.metadata["start"])

                    st.markdown(
                        f"**{timestamp}** — {source.page_content}"
                    )

                    start_seconds = int(source.metadata["start"])

                    youtube_link = (
                        f"https://www.youtube.com/watch?"
                        f"v={st.session_state.video_id}&t={start_seconds}s"
                    )

                    st.markdown(
                        f"[▶ Jump to {timestamp}]({youtube_link})"
                    )