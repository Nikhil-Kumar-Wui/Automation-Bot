import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import fitz
from openai import AzureOpenAI


# Azure OpenAI setup
client = AzureOpenAI(
    azure_endpoint=st.secrets["AZURE_OPENAI"]["ENDPOINT"],
    api_key=st.secrets["AZURE_OPENAI"]["API_KEY"],
    api_version="2024-12-01-preview"
)

deployment_name = st.secrets["AZURE_OPENAI"]["DEPLOYMENT"]


# Streamlit configuration
st.set_page_config(page_title="Automation Bot", page_icon="🤖", layout="centered")

col2, col1 = st.columns([10, 1])  # Emoji column smaller


with col2:
    st.title("Automation Bot👾")

with col1:
    st.image("https://em-content.zobj.net/source/microsoft-teams/363/robot_1f916.png", width=40)




if "history" not in st.session_state:
    st.session_state.history = []

# if "history" not in st.session_state:
#     st.session_state.history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, width=chunk_size)

def extract_text_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# helper function to get relevant chunks
def get_relevant_chunks(query):
    vec=st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(vec, st.session_state.chunks_vectors).flatten()
    top_indices=similarities.argsort()[-10:][::-1]
    return "\n\n".join([st.session_state.chunks[i] for i in top_indices])

def build_prompt(query):
    chat_history = "\n".join(
        [f"User: {u}\nBot: {b}" for u, b in st.session_state.history[-3:]]
    )

    if "vectorizer" in st.session_state:
        context = get_relevant_chunks(query)
        prompt = f"""
        You are a helpful assistant. Answer the user's question using the context from a PDF file.

        Context: {context}
        Chat History: {chat_history}
        User: {query}
        Bot:
        """
    else:
        prompt = f"""
        You are a helpful assistant. Answer the user's question even though no context file is uploaded.

        Chat History: {chat_history}
        User: {query}
        Bot:
        """
    
    return prompt.strip()

st.sidebar.title("Upload PDF📁")
uploaded_file=st.sidebar.file_uploader("Upload a Text or PDF file", type=["txt", "pdf"])

def process_file(uploaded_file, file_type):
    if file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")
    

    chunks = chunk_text(text)
    vectorizer=TfidfVectorizer().fit(chunks)
    chunks_vectors=vectorizer.transform(chunks)

    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.chunks_vectors = chunks_vectors

if uploaded_file:
    file_type=uploaded_file.type.split("/")[-1]
    process_file(uploaded_file, file_type)
    st.sidebar.success("File processed successfully!✅")

# Chat Interface

st.subheader("Ask anything👀:")
user_query = st.text_input("Your Question❓:")

if st.button("Ask") and user_query:
    prompt=build_prompt(user_query)
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}]
        )
        bot_reply = response.choices[0].message.content
        st.session_state.history.append((user_query, bot_reply))
        st.success("Response generated successfully!✅")
    except Exception as e:
        st.error(f"Error generating response: {e}")
if st.session_state.history:
    st.subheader("Conversation")

    for user_msg, bot_msg in reversed(st.session_state.history):
        # User message bubble (right-aligned)
        st.markdown(
            f"""
            <div style='float: right; clear: both; background-color: #262730;
                        color: white; padding: 10px 14px; border-radius: 10px;
                        margin-bottom: 10px; max-width: 75%; text-align: left;'>
                <b>🙋‍♂️ Me:</b><br>{user_msg}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Bot message bubble (left-aligned)
        st.markdown(
            f"""
            <div style='float: left; clear: both; background-color: #333;
                        color: white; padding: 10px 14px; border-radius: 10px;
                        margin-bottom: 20px; max-width: 75%; text-align: left;'>
                <b>🤖 Bot:</b><br>{bot_msg}
            </div>
            """,
            unsafe_allow_html=True
        )


