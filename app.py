import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

st.set_page_config(page_title="Mini RAG Q&A", page_icon="📚")

st.title("📚 Mini RAG Q&A")
st.write("Paste a document, store it in ChromaDB, and ask questions about it.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

client = chromadb.Client()
collection = client.get_or_create_collection(name="documents")

document = st.text_area(
    "📄 Paste your document here",
    height=250,
    placeholder="Paste your notes, article, syllabus, etc."
)

if st.button("➕ Add Document"):
    if not document.strip():
        st.warning("Please enter some text.")
    else:
        chunks = [
            document[i:i + 500]
            for i in range(0, len(document), 500)
        ]

        embeddings = embedding_model.encode(chunks)

        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist()
        )

        st.success(f"Added {len(chunks)} chunk(s) to ChromaDB.")


question = st.text_input("Ask a question about your document")

if st.button("Ask AI"):
    if not question.strip():
        st.warning("Please enter a question.")

    elif collection.count() == 0:
        st.warning("Please add a document first.")

    else:
        question_embedding = embedding_model.encode([question])[0]

        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=min(3, collection.count())
        )

        retrieved_chunks = results["documents"][0]

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I don't know based on the provided document."
"""

        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            st.subheader("🤖 Answer")
            st.write(response["message"]["content"])

            with st.expander("📖 Retrieved Context"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.write(f"**Chunk {i + 1}:**")
                    st.write(chunk)

        except Exception as e:
            st.error(
                "Could not connect to Ollama. "
                "Make sure Ollama is running and run "
                "`ollama pull llama3.2` first."
            )

            st.code(str(e))


st.divider()
st.caption(
    "Python + Sentence Transformers + ChromaDB + Ollama + Streamlit"
)