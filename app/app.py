from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
from urllib.parse import urlencode
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import openai

# Load environment variables
load_dotenv()

app = FastAPI(title="Smart Email Assistant", description="AI-powered email management system")

# OAuth2 configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# OpenAI configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

@app.get("/")
def read_root():
    """Root endpoint with basic info."""
    return {
        "message": "Smart Email Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/auth/gmail/initiate",
            "docs": "/docs",
            "emails": "/emails"
        }
    }

# --- Models ---
class EmailRequest(BaseModel):
    thread_id: str
    message: str = None

class SummarizeRequest(BaseModel):
    thread_id: str
    messages: list[str]

class ClassifyRequest(BaseModel):
    message: str

class ReplyRequest(BaseModel):
    thread_id: str
    message: str

# --- OAuth2 Endpoints ---
@app.get("/auth/gmail/initiate")
def gmail_auth_initiate():
    """Initiate Gmail OAuth2 flow."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google Client ID not configured")
    
    # Gmail API scopes (using less sensitive scopes for testing)
    scopes = [
        "https://www.googleapis.com/auth/gmail.readonly"
    ]
    
    # Build authorization URL
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "scope": " ".join(scopes),
        "response_type": "code",
        "access_type": "offline",
        "prompt": "consent"
    }
    
    auth_url_with_params = f"{auth_url}?{urlencode(params)}"
    return RedirectResponse(url=auth_url_with_params)

@app.get("/auth/gmail/callback")
def gmail_auth_callback(code: str = None, error: str = None):
    """Handle Gmail OAuth2 callback."""
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not provided")
    
    # Exchange authorization code for tokens
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": GOOGLE_REDIRECT_URI
    }
    
    try:
        response = requests.post(token_url, data=token_data)
        response.raise_for_status()
        tokens = response.json()
        
        # Store tokens (in production, use a secure database)
        # For now, we'll return them (don't do this in production!)
        return {
            "message": "Authentication successful!",
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "expires_in": tokens.get("expires_in")
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {str(e)}")

# --- Email Fetching ---
@app.get("/emails")
def fetch_emails(access_token: str = None):
    """Fetch important recent emails from Gmail Primary tab."""
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token required")
    
    try:
        # Create credentials from access token
        credentials = Credentials(access_token)
        
        # Build Gmail service
        service = build('gmail', 'v1', credentials=credentials)
        
        # Fetch only important, recent emails from Primary tab
        results = service.users().messages().list(
            userId='me',
            maxResults=10,
            labelIds=['INBOX', 'CATEGORY_PERSONAL', 'IMPORTANT'],
            q="is:important"
        ).execute()
        
        messages = results.get('messages', [])
        emails = []
        
        for message in messages:
            msg = service.users().messages().get(
                userId='me', 
                id=message['id'],
                format='metadata',
                metadataHeaders=['Subject', 'From', 'Date']
            ).execute()
            
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
            
            emails.append({
                'id': message['id'],
                'subject': subject,
                'from': sender,
                'date': date,
                'snippet': msg.get('snippet', '')
            })
        
        return {
            "emails": emails,
            "count": len(emails),
            "message": "Important recent emails from Primary tab fetched successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch emails: {str(e)}")

# --- Summarization ---
@app.post("/emails/summarize")
def summarize_email(req: SummarizeRequest):
    print("SummarizeRequest received:", req.dict())
    """Summarize an email thread using chunked/recursive Hugging Face summarization."""
    try:
        def summarize_text(text):
            api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            payload = {"inputs": text}
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict) and result.get("error"):
                return f"Hugging Face error: {result['error']}"
            return result[0]["summary_text"] if isinstance(result, list) and "summary_text" in result[0] else str(result)

        # Chunk emails (2 per chunk)
        chunk_size = 2
        messages = req.messages
        chunks = [messages[i:i+chunk_size] for i in range(0, len(messages), chunk_size)]
        chunk_summaries = []
        for chunk in chunks:
            chunk_text = "\n\n".join(chunk)
            summary = summarize_text(chunk_text)
            chunk_summaries.append(summary)

        # If more than one chunk, recursively summarize the summaries
        final_summary = "\n".join(chunk_summaries)
        if len(chunk_summaries) > 1:
            final_summary = summarize_text(final_summary)

        return {"summary": final_summary}
    except Exception as e:
        return {"summary": f"Unable to summarize email thread: {str(e)}"}

# --- Classification ---
@app.post("/emails/classify")
def classify_email(req: ClassifyRequest):
    """Classify an email using Hugging Face Inference API (zero-shot)."""
    try:
        api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        candidate_labels = ["work", "personal", "spam", "finance", "other"]
        payload = {
            "inputs": req.message,
            "parameters": {"candidate_labels": candidate_labels}
        }
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict) and result.get("error"):
            return {"category": f"Hugging Face error: {result['error']}"}
        # Pick the label with the highest score
        category = result["labels"][0] if "labels" in result and result["labels"] else "other"
        return {"category": category}
    except Exception as e:
        return {"category": "other", "message": f"Unable to classify email: {str(e)}"}

# --- Auto-Reply ---
@app.post("/emails/reply")
def auto_reply(req: ReplyRequest):
    """Generate and send an auto-reply."""
    # TODO: Use OpenAI to generate reply and send via Gmail API
    return {"reply": "This is a placeholder auto-reply."}

@app.post("/emails/summarize/recent")
def summarize_recent_emails(access_token: str = None):
    """Fetch and summarize the 10 most recent, important emails from the Primary tab."""
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token required")
    try:
        # Fetch emails (reuse logic from fetch_emails)
        credentials = Credentials(access_token)
        service = build('gmail', 'v1', credentials=credentials)
        results = service.users().messages().list(
            userId='me',
            maxResults=10,
            labelIds=['INBOX', 'CATEGORY_PERSONAL', 'IMPORTANT'],
            q="is:important"
        ).execute()
        messages = results.get('messages', [])
        email_texts = []
        for message in messages:
            msg = service.users().messages().get(
                userId='me',
                id=message['id'],
                format='metadata',
                metadataHeaders=['Subject', 'From', 'Date']
            ).execute()
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
            snippet = msg.get('snippet', '')
            email_text = f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n{snippet}"
            email_texts.append(email_text)
        # Use the same chunked summarization logic
        def summarize_text(text):
            api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            payload = {"inputs": text}
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict) and result.get("error"):
                return f"Hugging Face error: {result['error']}"
            return result[0]["summary_text"] if isinstance(result, list) and "summary_text" in result[0] else str(result)
        chunk_size = 2
        chunks = [email_texts[i:i+chunk_size] for i in range(0, len(email_texts), chunk_size)]
        chunk_summaries = []
        for chunk in chunks:
            chunk_text = "\n\n".join(chunk)
            summary = summarize_text(chunk_text)
            chunk_summaries.append(summary)
        print (chunk_summaries)
        final_summary = "\n".join(chunk_summaries)
        if len(chunk_summaries) > 1:
            final_summary = summarize_text(final_summary)
        return {"summary": final_summary}
    except Exception as e:
        return {"summary": f"Unable to summarize recent emails: {str(e)}"}

@app.post("/emails/classify/recent")
def classify_recent_emails(access_token: str = None):
    """Fetch and classify the 5 most recent, important emails from the Primary tab (with timeout and progress logging)."""
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token required")
    try:
        # Fetch emails (reuse logic from fetch_emails)
        credentials = Credentials(access_token)
        service = build('gmail', 'v1', credentials=credentials)
        results = service.users().messages().list(
            userId='me',
            maxResults=5,  # Limit to 5 for testing
            labelIds=['INBOX', 'CATEGORY_PERSONAL', 'IMPORTANT'],
            q="is:important"
        ).execute()
        messages = results.get('messages', [])
        email_results = []
        for idx, message in enumerate(messages):
            msg = service.users().messages().get(
                userId='me',
                id=message['id'],
                format='metadata',
                metadataHeaders=['Subject', 'From', 'Date']
            ).execute()
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
            snippet = msg.get('snippet', '')
            email_text = f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n{snippet}"
            # Classify using Hugging Face zero-shot classifier
            api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
            headers_hf = {"Authorization": f"Bearer {HF_API_KEY}"}
            candidate_labels = ["work", "personal", "spam", "finance", "other"]
            payload = {
                "inputs": email_text,
                "parameters": {"candidate_labels": candidate_labels}
            }
            try:
                print(f"Classifying email {idx+1}/{len(messages)}: {subject}")
                response = requests.post(api_url, headers=headers_hf, json=payload, timeout=15)
                response.raise_for_status()
                result = response.json()
                if isinstance(result, dict) and result.get("error"):
                    category = f"Hugging Face error: {result['error']}"
                else:
                    category = result["labels"][0] if "labels" in result and result["labels"] else "other"
            except Exception as e:
                print(f"Error classifying email {idx+1}: {e}")
                category = f"Unable to classify: {str(e)}"
            email_results.append({
                'id': message['id'],
                'subject': subject,
                'from': sender,
                'date': date,
                'snippet': snippet,
                'category': category
            })
        return {"emails": email_results, "count": len(email_results)}
    except Exception as e:
        return {"emails": [], "error": f"Unable to classify recent emails: {str(e)}"}
