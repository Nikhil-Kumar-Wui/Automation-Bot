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


# pip install tzdata
# pip install requests
# pip install secure-smtplib


# Initialize the Azure OpenAI client

client = AzureOpenAI(
    azure_endpoint=st.secrets["AZURE_OPENAI"]["ENDPOINT"],
    api_key=st.secrets["AZURE_OPENAI"]["API_KEY"],
    api_version="2024-12-01-preview"
)

deployment_name = st.secrets["AZURE_OPENAI"]["DEPLOYMENT"]


# Simplified timezone data
TIMEZONE_DATA = {
    "tokyo": "Asia/Tokyo",
    "san francisco": "America/Los_Angeles",
    "paris": "Europe/Paris"
}

def get_current_time(location):
    """Get the current time for a given location"""
    print(f"get_current_time called with location: {location}")  #Take the Location input from the user
    location_lower = location.lower() #Lowercase the location for case-insensitive matching
    
    for key, timezone in TIMEZONE_DATA.items():
        if key in location_lower:
            print(f"Timezone found for {key}")  #Timezone
            current_time = datetime.now(ZoneInfo(timezone)).strftime("%I:%M %p") #Get the current time in the specified timezone 
            return json.dumps({
                "location": location,
                "current_time": current_time
            })
    
    print(f"No timezone data found for {location_lower}")  
    return json.dumps({"location": location, "current_time": "unknown"})

def favorite_color():
    """Return the favorite color of NIKHIL"""
    print("favorite_color called")  # Indicate that the function was called
    return json.dumps({"color": "blue"})  # Return NIKHIL's favorite color in JSON format

def get_weather_info(location):
    """Fetches weather info using wttr.in (no API key needed)"""
    print(f"get_weather_info called with location: {location}")
    try:
        response = requests.get(f"https://wttr.in/{location}?format=3")  # Simple 1-line weather
        if response.status_code == 200:
            return json.dumps({"location": location, "weather": response.text})
        else:
            return json.dumps({"location": location, "weather": "Unable to fetch weather"})
    except Exception as e:
        return json.dumps({"location": location, "weather": str(e)})
    
# Email sending function
def send_email_summary(to_email, subject, body):
    sender_email = st.secrets["EMAIL"]["ADDRESS"]
    sender_password = st.secrets["EMAIL"]["PASSWORD"]  # NOT your Gmail password
    recipient_email = "nikhilkumarnkh12@gmail.com"  # Replace with the recipient's email

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
        print("✅ Email sent successfully!")
    except Exception as e:
        print("❌ Error sending email:", str(e))

def extract_email_from_text(text):
    """Extracts the first email found in a string."""
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None

def run_conversation():
    # Initial user message
    messages = [{"role": "user", "content": "What is the weather and time in Tokyo?"}] # Single function call
    # Parallel function call with a single tool/function defined

    # Define the function for the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                },
            }
        },
        {
            "type": "function",
            "function":{
                "name":"favorite_color",
                "description":"It tells the favorite color of NIKHIL",
                "parameters":{
                    "type": "object",
                    "properties": {
                        "color": {
                            "type": "string",
                            "description": "The favorite color of NIKHIL",
                        }
                    }
                    
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
                "location": {
                    "type": "string",
                    "description": "The city or location name, e.g. San Francisco"
                }
            },
            "required": ["location"]
        }
    }
}
    ]

    # First API call: Ask the model to use the function
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Process the model's response
    response_message = response.choices[0].message
    messages.append(response_message)

    print("Model's response:")  
    print(response_message)  

    # Handle function calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_current_time":
                function_args = json.loads(tool_call.function.arguments)
                print(f"Function arguments: {function_args}")  
                time_response = get_current_time(
                    location=function_args.get("location")
                )
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "get_current_time",
                    "content": time_response,
                })

            elif tool_call.function.name == "get_weather_info":
                function_args = json.loads(tool_call.function.arguments)
                print(f"Function arguments: {function_args}")
                weather_response = get_weather_info(
                    location=function_args.get("location")
                )
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "get_weather_info",
                    "content": weather_response,
                })

    else:
        print("No tool calls were made by the model.")  

    # Second API call: Get the final response from the model
    final_response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
    )
    # Send the summary email
    email_body = final_response.choices[0].message.content
    user_email = extract_email_from_text(messages[0]["content"])

    # Set fallback/default email
    default_email = "nikhilkumarnkh12@gmail.com"

    # Use extracted email or fallback
    send_email_summary(
        to_email=user_email if user_email else default_email,
        subject="Daily AI Summary",
        body=email_body
    )


    return final_response.choices[0].message.content

# Run the conversation and print the result
print(run_conversation())