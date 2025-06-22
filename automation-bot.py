# Necessary imports
import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import fitz
from openai import AzureOpenAI # OpenAI API client
import json #JSON Tool Schema
# Importing the necessary libraries for date and time handling while using LLMs
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re



# Time by city
TIMEZONE_DATA = {
    "tokyo": "Asia/Tokyo",
    "san francisco": "America/Los_Angeles",
    "paris": "Europe/Paris"
}

def get_current_time(location):
    location_lower = location.lower()
    for key, timezone in TIMEZONE_DATA.items():
        if key in location_lower:
            current_time = datetime.now(ZoneInfo(timezone)).strftime("%I:%M %p")
            return {"location": location, "current_time": current_time}
    return {"location": location, "current_time": "unknown"}

# Weather
def get_weather_info(location):
    try:
        response = requests.get(f"https://wttr.in/{location}?format=3")
        if response.status_code == 200:
            return {"location": location, "weather": response.text}
        else:
            return {"location": location, "weather": "Unable to fetch weather"}
    except Exception as e:
        return {"location": location, "weather": str(e)}
    
# Email sending
def send_email_summary(to_email, subject, body):
    sender_email = st.secrets["EMAIL"]["ADDRESS"]
    sender_password = st.secrets["EMAIL"]["PASSWORD"]
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        st.success("Email sent successfully! ✅")
    except Exception as e:
        st.error(f"Error sending email: {e}")

def extract_email_from_text(text):
    match = re.search(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", text)
    return match.group(0) if match else None




# Function to run the conversation with tools
def run_conversation(query):
    messages = [{"role": "user", "content": query}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather_info",
                "description": "Get the current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    # First model call
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # Get the assistant message with tool_calls
    response_message = response.choices[0].message
    messages.append(response_message)

    # Prepare tool responses
    if hasattr(response_message, "tool_calls") and response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Call the corresponding Python function
            if tool_name == "get_current_time":
                result = get_current_time(**tool_args)
            elif tool_name == "get_weather_info":
                result = get_weather_info(**tool_args)
            else:
                result = {"message": "Unknown tool"}

            # Add tool response using tool_call_id
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result),
            })

        # Final assistant response after tool execution
        final_response = client.chat.completions.create(
            model=deployment_name,
            messages=messages
        )
        return final_response.choices[0].message.content

    else:
        # No tool calls needed
        return response_message.content


# Azure OpenAI setup
client = AzureOpenAI(
    api_key=st.secrets["AZURE_OPENAI"]["API_KEY"],
    api_version="2024-12-01-preview",
    azure_endpoint=st.secrets["AZURE_OPENAI"]["ENDPOINT"]
    )

deployment_name = st.secrets["AZURE_OPENAI"]["DEPLOYMENT"]


# Streamlit configuration
st.set_page_config(page_title="Automation Bot", page_icon="🤖", layout="centered")

col2, col1 = st.columns([10, 1])  # Emoji column smaller


with col2:
    st.title("Automation Bot👾")

with st.expander(" ℹ️How to use this Automation Bot?"):
    st.markdown("""
    **Instructions:**

    - 📁 **Upload a PDF or Text file** from the sidebar to load custom data.
    - 💬 **Ask questions** in the chat box below.
    - 🤖 The bot will answer using your uploaded context (if provided).
    - ⚙️ **Select automation** from the dropdown:
        - 📤 **Send to Email** — Email the bot's response to an address in your query.
        - ⚙️ **Trigger Zapier to Email** — Send to a Zapier webhook (e.g., Google Docs).
        - 💾 **Save as Text File** — Locally save response and notify Zapier.
    - 🧠 The bot can also detect if automation is implied from your query and trigger it automatically when 'None' is selected.
    - 📧 If an email is detected in your query, it will be used for sending the result.

    **Example Queries:**
    - "What's the weather in Tokyo? Email it to nikhil@example.com"
    - "Summarize this document and save it to Google Docs"
    - "Give me the key takeaways from the uploaded file"

    """)




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
    top_indices=similarities.argsort()[-5:][::-1]
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


# Dropdown for automation type
automation_type = st.selectbox(
    "Choose automation type:",
    ["❌ None", "📤 Send to Email", "⚙️ Trigger Zapier to Email", "💾 Save as Text File"],
    index=0  # Default to "None"
)

def is_email_request_via_llm(query):
    check_prompt = f"""
    Analyze the following user query and determine if it implies any automation action such as:
    - sending an email,
    - triggering a webhook,
    - saving to a file (e.g., text or Google Doc),
    - or logging/summarizing a response.

    Just answer "yes" if it implies any of the above. Otherwise, say "no".

    User query: "{query}"
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": check_prompt}]
    )
    answer = response.choices[0].message.content.strip().lower()
    return "yes" in answer

if st.button("Ask") and user_query:
    try:
        if "vectorizer" in st.session_state:
            prompt = build_prompt(user_query)
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": prompt}]
            )
            bot_reply = response.choices[0].message.content
        else:
            # Use tool logic
            bot_reply = run_conversation(user_query)

        st.session_state.history.append((user_query, bot_reply))
        st.session_state.last_bot_reply = bot_reply  # Save for safety

        # Use LLM to classify user intent
        should_email = is_email_request_via_llm(user_query) 
        


        if automation_type == "❌ None":
            

            if should_email:
                # ✅ Try sending to a personal email if found in the query
                recipient_email = extract_email_from_text(user_query)
                if recipient_email:
                    try:
                        send_email_summary(recipient_email, "Bot Response", bot_reply)
                        st.success(f"Email sent to {recipient_email} as detected from query! ✅")
                    except Exception as e:
                        st.error(f"Error sending email to {recipient_email}: {e}")
                else:
                    st.info("No email detected in the query, skipping email sending.")

                
                if re.search(r"\b(save|log|record|zapier)\b", user_query.lower()):
                    try:
                        webhook_url = st.secrets["ZAPIER"]["WEBHOOK_URL"]
                        payload = {
                            "summary": bot_reply,
                            "user_query": user_query,
                            "timestamp": datetime.now().isoformat(),
                            "filename": f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        }
                        response = requests.post(webhook_url, json=payload)
                        if response.status_code == 200:
                            st.success("Sent to Zapier to save as file ✅")
                        else:
                            st.warning(f"Zapier returned status {response.status_code}")
                    except Exception as e:
                        st.error(f"Error sending to Zapier: {e}")

            else:
                st.success("Response generated successfully (no automation needed). ✅")


        elif automation_type == "📤 Send to Email":
            try:
                # Extract recipient email from the query
                recipient_email = extract_email_from_text(user_query)

                if recipient_email:
                    send_email_summary(recipient_email, "Bot Response", bot_reply)
                else:
                    st.warning("No email found in the query. Please mention a valid email address.")
                st.success("Emailed successfully! ✅")
            except Exception as e:
                st.error(f"Email error: {e}")

        elif automation_type == "⚙️ Trigger Zapier":
            try:
                webhook_url = "https://hooks.zapier.com/hooks/catch/23478174/uoej4lv/"
                payload = {
                    "summary": bot_reply,
                    "user_query": user_query,
                    "timestamp": datetime.now().isoformat()
                }
                response = requests.post(webhook_url, json=payload)
                if response.status_code == 200:
                    st.success("Zapier webhook triggered! ✅")
                else:
                    st.warning(f"Webhook returned {response.status_code}")
            except Exception as e:
                st.error(f"Zapier webhook failed: {e}")

        elif automation_type == "💾 Save as Docs":
            try:
                file_name = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(bot_reply)
                st.success(f"Saved as {file_name} ✅")

                # Also trigger Zapier to save as Google Doc
                webhook_url = "https://hooks.zapier.com/hooks/catch/23478174/uoej4lv/"
                payload = {
                    "summary": bot_reply,
                    "user_query": user_query,
                    "timestamp": datetime.now().isoformat(),
                    "filename": file_name
                }
                response = requests.post(webhook_url, json=payload)
                if response.status_code == 200:
                    st.success("Also sent to Zapier for Google Doc ✅")
                else:
                    st.warning(f"Zapier failed with status {response.status_code}")
            except Exception as e:
                st.error(f"Error saving file or calling Zapier: {e}")

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


