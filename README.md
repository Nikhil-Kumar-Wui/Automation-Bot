# 🤖 Automation-Bot

An intelligent assistant that combines Azure OpenAI with PDF/text file processing, email sending, and Zapier automation — all through a clean Streamlit interface.

---

## 👋 Welcome to Automation Bot!

A smart assistant that understands your queries and can take actions like emailing responses, saving documents, or answering questions from uploaded PDFs.

---

## 📄 Upload PDF or Text File

- Use the **sidebar** to upload `.pdf` or `.txt` files.
- The bot extracts and understands the document.
- You can ask context-aware questions such as:
  - *"Summarize the uploaded paper"*
  - *"What are the key takeaways?"*

---

## 💬 Ask Questions

- Type your question in the **chat box**.
- If a file is uploaded, it uses the document content to answer.
- If no file is uploaded, it responds using external tools (e.g., weather, time) or general knowledge.

---

## ⚙️ Automation Type Options (Dropdown)

| Option                             | Description                                                                                     |
|------------------------------------|-------------------------------------------------------------------------------------------------|
| ❌ None                             | No automation — just display the response. Bot may still auto-detect intent based on your query. |
| 📤 Send to Email                    | Sends the response to the email mentioned in your query. Example: `"Send to john@gmail.com"`    |
| ⚙️ Trigger Zapier (Email/Docs)      | Triggers a connected **Zapier Webhook**:<br> → Sends email via Zapier<br> → Saves to Google Docs |
| 💾 Save as File / Google Docs       | Bot will detect and either:<br> → Give a text file to download<br> → Send to Google Docs via Zapier |

> ⚠️ **Tip:** Even when set to **None**, the bot can detect automation intents in your query and act accordingly.

---

## ✨ Smart Automation Examples

- `"What's the weather in Paris? Email it to nikhil@gmail.com"` → Sends weather info to email.
- `"Summarize this PDF and save to Google Docs"` → Triggers Zapier to save document.
- `"Download this answer to my PC"` → Offers download as `.txt` file.

---

## 🧠 Behind the Scenes

- Uses **LLM-based intent detection** to infer if automation is needed.
- If a valid email address is detected, it is auto-used for emailing.
- Handles tools like:
  - `get_current_time(location)`
  - `get_weather_info(location)`
- Supports dynamic automation based on:
  - Azure OpenAI responses
  - Intent classification via prompt
  - External actions (Zapier, Email, Local file)

---

## 🚀 Tech Stack

- [Streamlit](https://streamlit.io/) – UI
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/) – LLM backend
- [scikit-learn](https://scikit-learn.org/) – Text vectorization for PDF retrieval
- [Zapier Webhooks](https://zapier.com/) – Automation integration
- `smtplib` – Email sending
- `fitz` (PyMuPDF) – PDF parsing

---

## 🔐 Secrets Required

Store these in `.streamlit/secrets.toml`:

```toml
[AZURE_OPENAI]
API_KEY = "your-key"
ENDPOINT = "your-endpoint"
DEPLOYMENT = "your-deployment-name"

[EMAIL]
ADDRESS = "your-email@gmail.com"
PASSWORD = "your-app-password"

[ZAPIER]
WEBHOOK_URL = "your-zapier-webhook-url"
