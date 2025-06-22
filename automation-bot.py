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
    messages = [
                {"role": "system", "content": (
                    "You are a helpful assistant. You can answer user queries and assume external automation tools like Zapier "
                    "will handle sending emails, saving documents, or triggering webhooks. "
                    "Do not say 'I can't send email' — just respond with the answer and let the automation handle the rest."
                )},
                {"role": "user", "content": query}
            ]
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
    ### 👋 Welcome to Automation Bot!
    A smart assistant that understands your queries and can take actions like emailing responses, saving documents, or answering questions from uploaded PDFs.

    ---

    ### 📄 **Upload PDF or Text File**
    - Go to the **sidebar**.
    - Click **"Upload a Text or PDF file"**.
    - The bot will extract and understand your document.
    - You can then ask context-aware questions (e.g., *"Summarize the uploaded paper"*, *"What are the key takeaways?"*).

    ---

    ### 💬 **Ask Questions**
    - Type your question in the **chat input box**.
    - If a file is uploaded, it will use the document content to answer.
    - If no file is uploaded, the bot will still answer using general knowledge or external tools like weather/time.

    ---

    ### ⚙️ **Automation Type Options (Dropdown)**

    | Option | Description |
    |--------|-------------|
    | ❌ None | No automation — just display the response in the chat. However, if your query *implies* an automation (e.g., "Email this"), the bot may auto-trigger the correct action. |
    | 📤 Send to Email | Sends the response to the email mentioned in your query (e.g., "Send this to john@gmail.com"). |
    | ⚙️ Trigger Zapier (Email/Docs) | Triggers a connected **Zapier Webhook** for automation. It detects if you want to: <br> → **Email** someone via Zapier, or <br> → **Save** to Google Docs or another destination. |
    | 💾 Save as File / Google Docs | Bot detects intent: <br> → Saves as downloadable text file, or <br> → Sends to Google Docs via Zapier. |

    ⚠️ **Tip:** If you choose **None**, the bot will auto-detect intent based on your question!

    ---

    ### ✨ **Smart Automation Examples**
    - *"What's the weather in Paris? Email it to nikhil@gmail.com"* → Sends weather info to email.
    - *"Summarize this PDF and save to Google Docs"* → Triggers Zapier to save summary.
    - *"Download this answer to my PC"* → Gives you a download link for a text file.

    ---

    ### 🧠 Behind the Scenes
    - Uses **LLM-based intent detection** to infer what automation (if any) should be triggered.
    - If email is found in your query, it is automatically used.

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


dropdown_col, _ = st.columns([1, 1])  # Make the dropdown column narrower

with dropdown_col:
    automation_type = st.selectbox(
        "Automation Type:",
        ["❌ None", "📤 Send to Email", "⚙️ Trigger Zapier (Email/Docs)", "💾 Save as File / Google Docs"],
        index=0
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

# Keyword-based fallback (optional, can be removed if only LLM is used)
def detect_save_target_from_query(query):
    """Return 'local' or 'zapier' depending on keywords in intent."""
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["local", "pc", "my computer", "download"]):
        return "local"
    elif any(kw in query_lower for kw in ["google docs", "google drive", "docs", "save online", "cloud"]):
        return "zapier"
    return "unknown"

def detect_zapier_action_with_llm(query):
    prompt = f"""
    Determine what Zapier action is needed for the following query.

    Respond with just one word:
    - "email" if the user wants to send the response to an email address.
    - "document" if the user wants to save the response to something like Google Docs.
    - "unknown" if it's unclear.

    Query: "{query}"
    """
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip().lower()
        return result if result in ["email", "document"] else "unknown"
    except Exception as e:
        st.warning(f"Zapier action detection failed. Defaulting to 'unknown'. Error: {e}")
        return "unknown"

# LLM-based primary detection
def detect_save_target_with_llm(query):
    """Use LLM to classify if user wants local file or Google Docs."""
    prompt = f"""
    Determine the preferred save location for this user query.

    Options:
    - "local" — if the user wants to save as a file on their computer.
    - "zapier" — if the user wants to save to an online destination like Google Docs.
    - "unknown" — if unclear.

    User query: "{query}"
    Answer with only one word: local, zapier, or unknown.
    """
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer if answer in ["local", "zapier"] else "unknown"
    except Exception as e:
        st.warning(f"LLM classification failed. Using keyword fallback. Error: {e}")
        return detect_save_target_from_query(query)


if st.button("Ask") and user_query:
    try:
        # Decide if we use PDF-based prompt or tools
        if "vectorizer" in st.session_state:
            prompt = build_prompt(user_query)
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": prompt}]
            )
            bot_reply = response.choices[0].message.content
        else:
            bot_reply = run_conversation(user_query)

        # Add to history after response is generated
        st.session_state.history.append((user_query, bot_reply))
        st.session_state.last_bot_reply = bot_reply


        # Use LLM to classify user intent
        should_email = is_email_request_via_llm(user_query) 
        


        if automation_type == "❌ None":
            try:
                # Use LLM to detect automation intent
                zapier_action = detect_zapier_action_with_llm(user_query)  # "email" or "document"
                detected_target = detect_save_target_with_llm(user_query)  # "local" or "zapier"

                webhook_url = st.secrets["ZAPIER"]["WEBHOOK_URL"]
                timestamp = datetime.now().isoformat()
                filename = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                full_text = f"User Query:\n{user_query}\n\nBot Reply:\n{bot_reply}"

                # ---- Case: User wants email (via Zapier) ----
                if zapier_action == "email":
                    payload = {
                        "action_type": "email",
                        "user_query": user_query,
                        "bot_reply": bot_reply,
                        "timestamp": timestamp
                    }

                    recipient_email = extract_email_from_text(user_query)
                    if recipient_email and recipient_email.endswith("@gmail.com"):
                        payload["recipient_email"] = recipient_email
                    else:
                        st.warning("⚠️ No valid Gmail found. Zapier will proceed with default email.")

                    response = requests.post(webhook_url, json=payload)
                    if response.status_code == 200:
                        st.success("📧 Zapier triggered to send email ✅")
                    else:
                        st.warning(f"Zapier email failed with status {response.status_code}")

                # ---- Case: User wants to save to Docs via Zapier ----
                elif zapier_action == "document" or detected_target == "zapier":
                    payload = {
                        "action_type": "document",
                        "user_query": user_query,
                        "bot_reply": bot_reply,
                        "full_text": full_text,
                        "filename": filename,
                        "timestamp": timestamp
                    }

                    response = requests.post(webhook_url, json=payload)
                    if response.status_code == 200:
                        st.success("📄 Zapier triggered to save document ✅")
                    else:
                        st.warning(f"Zapier document save failed with status {response.status_code}")

                # ---- Case: Local Save Requested ----
                elif detected_target == "local":
                    st.success("💾 Detected request to save locally ✅")
                    st.download_button("⬇️ Download as Text File", full_text, filename, mime="text/plain")

                # ---- Fallback: No automation needed ----
                else:
                    st.success("✅ Response generated. No automation triggered.")

            except Exception as e:
                st.error(f"Automation detection failed: {e}")


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

        elif automation_type == "⚙️ Trigger Zapier (Email/Docs)":
            try:
                zapier_action = detect_zapier_action_with_llm(user_query)  # "email" or "document"
                webhook_url = st.secrets["ZAPIER"]["WEBHOOK_URL"]

                payload = {
                    "action_type": zapier_action,
                    "user_query": user_query,
                    "bot_reply": bot_reply,
                    "timestamp": datetime.now().isoformat()
                }

                # 👇 If the user query mentions a Gmail and action is 'email', add it to payload
                if zapier_action == "email":
                    recipient_email = extract_email_from_text(user_query)
                    if recipient_email and recipient_email.endswith("@gmail.com"):
                        payload["recipient_email"] = recipient_email
                    else:
                        st.warning("⚠️ No valid Gmail address found in query. Zapier will proceed without recipient_email.")

                elif zapier_action == "document":
                    payload["filename"] = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    payload["full_text"] = f"User Query:\n{user_query}\n\nBot Reply:\n{bot_reply}"

                # 🔁 Send to Zapier
                response = requests.post(webhook_url, json=payload)

                if response.status_code == 200:
                    if zapier_action == "email":
                        st.success("📧 Zapier triggered to send email ✅")
                    else:
                        st.success("📄 Zapier triggered to save document ✅")
                else:
                    st.warning(f"⚠️ Zapier returned status {response.status_code}")

            except Exception as e:
                st.error(f"Zapier automation failed: {e}")

        elif automation_type == "💾 Save as File / Google Docs":
            try:
                # Build content
                full_text = f"User Query:\n{user_query}\n\nBot Reply:\n{bot_reply}"
                file_name = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

                # Automatically detect target
                try:
                    detected_target = detect_save_target_with_llm(user_query)
                except Exception as e:
                    st.warning(f"LLM detection failed. Using keyword fallback. Error: {e}")
                    detected_target = detect_save_target_from_query(user_query)

                if detected_target == "local":
                    # Save file in memory for download
                    st.success("Local save option selected ✅")
                    st.download_button("⬇️ Download as Text File", full_text, file_name, mime="text/plain")

                elif detected_target == "zapier":
                    webhook_url = st.secrets["ZAPIER"]["WEBHOOK_URL"]
                    payload = {
                        "summary": bot_reply,
                        "user_query": user_query,
                        "full_text": full_text,
                        "timestamp": datetime.now().isoformat(),
                        "filename": file_name
                    }
                    response = requests.post(webhook_url, json=payload)
                    if response.status_code == 200:
                        st.success("Sent to Zapier for Google Doc ✅")
                    else:
                        st.warning(f"Zapier failed with status {response.status_code}")

                else:
                    st.info("❓ Could not determine whether to save locally or to Google Docs. Please include 'save to PC' or 'save to Google Docs' in your query.")

            except Exception as e:
                st.error(f"Error during save operation: {e}")
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

