import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

# --- Summarization ---
def summarize_thread(messages):
    prompt = f"Summarize the following email thread:\n\n{chr(10).join(messages)}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content'].strip()

# --- Classification ---
def classify_email(message):
    prompt = f"Classify the following email into one of these categories: work, personal, spam, other.\nEmail: {message}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content'].strip()

# --- Auto-Reply ---
def generate_auto_reply(message):
    prompt = f"Write a professional auto-reply to the following email:\n\n{message}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content'].strip() 