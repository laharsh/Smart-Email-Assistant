from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import base64
from dotenv import load_dotenv
import requests
from urllib.parse import urlencode
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from core.process_emails import EmailProcessor

# Load environment variables
load_dotenv()

# Production configuration
IS_PRODUCTION = os.getenv("IS_PRODUCTION", "false").lower() == "true"
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

app = FastAPI(
    title="Smart Email Assistant", 
    description="AI-powered email management system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not IS_PRODUCTION else [BASE_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# OAuth2 configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# Set redirect URI with fallback
if IS_PRODUCTION:
    # For production, use the BASE_URL
    GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", f"{BASE_URL}/auth/gmail/callback")
else:
    # For local development
    GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/gmail/callback")

# Hugging Face configuration
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize email processor
try:
    email_processor = EmailProcessor(
        openai_api_key="dummy_key",  # Not used anymore
        hf_api_key=HF_API_KEY
    )
except Exception as e:
    print(f"Warning: Email processor initialization failed: {e}")
    email_processor = None

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Serve the main application interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint for deployment platforms."""
    return {
        "status": "healthy",
        "service": "Smart Email Assistant",
        "version": "1.0.0",
        "production": IS_PRODUCTION
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
    email_id: str
    reply_type: str = "professional"  # professional, friendly, concise
    custom_message: str = ""

# --- OAuth2 Endpoints ---
@app.get("/auth/gmail/initiate")
def gmail_auth_initiate():
    """Initiate Gmail OAuth2 flow."""
    if not GOOGLE_CLIENT_ID:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Google Client ID not configured",
                "message": "Please set GOOGLE_CLIENT_ID in your .env file",
                "setup_instructions": [
                    "1. Go to https://console.cloud.google.com/apis/credentials",
                    "2. Create a new OAuth 2.0 Client ID",
                    "3. Add http://localhost:8000/auth/gmail/callback to authorized redirect URIs",
                    "4. Copy the Client ID and Client Secret to your .env file"
                ]
            }
        )
    
    if not GOOGLE_CLIENT_SECRET:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Google Client Secret not configured",
                "message": "Please set GOOGLE_CLIENT_SECRET in your .env file"
            }
        )
    
    # Debug: Print redirect URI for troubleshooting
    print(f"DEBUG: GOOGLE_REDIRECT_URI = {GOOGLE_REDIRECT_URI}")
    print(f"DEBUG: BASE_URL = {BASE_URL}")
    
    # Gmail API scopes
    scopes = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/userinfo.email"
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
    print(f"DEBUG: Auth URL = {auth_url_with_params}")
    return RedirectResponse(url=auth_url_with_params)

@app.get("/auth/gmail/callback")
def gmail_auth_callback(code: str = None, error: str = None):
    """Handle Gmail OAuth2 callback."""
    if error:
        # Redirect back to frontend with error
        return RedirectResponse(url=f"/?error={error}")
    
    if not code:
        return RedirectResponse(url="/?error=no_code")
    
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
        
        access_token = tokens.get("access_token")
        if not access_token:
            return RedirectResponse(url="/?error=no_access_token")
        
        # Redirect back to frontend with access token
        return RedirectResponse(url=f"/?access_token={access_token}")
        
    except requests.exceptions.RequestException as e:
        return RedirectResponse(url=f"/?error=token_exchange_failed")

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
def auto_reply(req: ReplyRequest, access_token: str = None):
    """Generate and send an auto-reply to an email."""
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token required")
    
    try:
        # Get Gmail service
        credentials = Credentials(access_token)
        service = build('gmail', 'v1', credentials=credentials)
        
        # Get the original email
        message = service.users().messages().get(
            userId='me',
            id=req.email_id,
            format='full'
        ).execute()
        
        # Extract email details
        headers = message['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
        original_content = message.get('snippet', '')
        
        # Generate reply based on type
        reply_templates = {
            "professional": f"Thank you for your email regarding '{subject}'. I appreciate you reaching out and will get back to you soon with a detailed response.",
            "friendly": f"Hi! Thanks for your message about '{subject}'. I've received your email and will respond in detail shortly. Have a great day!",
            "concise": f"Received your email about '{subject}'. Will respond soon."
        }
        
        # Use custom message if provided, otherwise use template
        reply_text = req.custom_message if req.custom_message else reply_templates.get(req.reply_type, reply_templates["professional"])
        
        # Create reply message
        reply_message = {
            'raw': base64.urlsafe_b64encode(
                f'To: {sender}\r\n'
                f'Subject: Re: {subject}\r\n'
                f'Content-Type: text/plain; charset=utf-8\r\n'
                f'\r\n'
                f'{reply_text}'.encode('utf-8')
            ).decode('utf-8')
        }
        
        # Send the reply
        sent_message = service.users().messages().send(
            userId='me',
            body=reply_message
        ).execute()
        
        return {
            "success": True,
            "message": "Auto-reply sent successfully!",
            "reply_text": reply_text,
            "sent_message_id": sent_message['id']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send auto-reply: {str(e)}")



@app.post("/emails/analyze/advanced")
def analyze_emails_advanced(access_token: str = None, max_results: int = 10):
    """Advanced email analysis with comprehensive insights using the new EmailProcessor."""
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token required")
    
    try:
        # Fetch emails
        credentials = Credentials(access_token)
        service = build('gmail', 'v1', credentials=credentials)
        results = service.users().messages().list(
            userId='me',
            maxResults=max_results,
            labelIds=['INBOX', 'CATEGORY_PERSONAL', 'IMPORTANT'],
            q="is:important"
        ).execute()
        
        messages = results.get('messages', [])
        emails_data = []
        
        # Extract email data
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
            
            emails_data.append({
                'id': message['id'],
                'subject': subject,
                'from': sender,
                'date': date,
                'snippet': snippet
            })
        
        # Use advanced email processor for analysis
        if email_processor is None:
            raise HTTPException(status_code=500, detail="Email processor not initialized")
        analyzed_emails = email_processor.batch_analyze_emails(emails_data)
        
        # Generate summary statistics
        categories = [email['analysis']['category'] for email in analyzed_emails]
        priorities = [email['analysis']['priority'] for email in analyzed_emails]
        sentiments = [email['analysis']['sentiment'] for email in analyzed_emails]
        action_required = [email for email in analyzed_emails if email['analysis']['action_required']]
        
        summary_stats = {
            "total_emails": len(analyzed_emails),
            "categories": {cat: categories.count(cat) for cat in set(categories)},
            "priorities": {pri: priorities.count(pri) for pri in set(priorities)},
            "sentiments": {sent: sentiments.count(sent) for sent in set(sentiments)},
            "action_required_count": len(action_required),
            "urgent_emails": [email for email in analyzed_emails if email['analysis']['priority'] == 'urgent'],
            "high_priority_emails": [email for email in analyzed_emails if email['analysis']['priority'] in ['urgent', 'high']]
        }
        
        return {
            "emails": analyzed_emails,
            "summary_statistics": summary_stats,
            "message": f"Successfully analyzed {len(analyzed_emails)} emails with advanced AI insights"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")

@app.post("/emails/summarize/smart")
def smart_summarize_emails(access_token: str = None, category_filter: str = None):
    """Smart summarization with category-specific insights and action items."""
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token required")
    
    try:
        # Fetch emails
        credentials = Credentials(access_token)
        service = build('gmail', 'v1', credentials=credentials)
        results = service.users().messages().list(
            userId='me',
            maxResults=15,
            labelIds=['INBOX', 'CATEGORY_PERSONAL', 'IMPORTANT'],
            q="is:important"
        ).execute()
        
        messages = results.get('messages', [])
        emails_data = []
        
        # Extract email data
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
            
            emails_data.append({
                'id': message['id'],
                'subject': subject,
                'from': sender,
                'date': date,
                'snippet': snippet
            })
        
        # Analyze emails
        if email_processor is None:
            raise HTTPException(status_code=500, detail="Email processor not initialized")
        analyzed_emails = email_processor.batch_analyze_emails(emails_data)
        
        # Filter by category if specified
        if category_filter:
            analyzed_emails = [email for email in analyzed_emails if email['analysis']['category'] == category_filter]
        
        # Group by category for better organization
        categorized_summaries = {}
        for email in analyzed_emails:
            category = email['analysis']['category']
            if category not in categorized_summaries:
                categorized_summaries[category] = []
            categorized_summaries[category].append({
                'subject': email['subject'],
                'from': email['from'],
                'priority': email['analysis']['priority'],
                'summary': email['analysis']['summary'],
                'action_required': email['analysis']['action_required']
            })
        
        # Generate category-specific insights
        insights = {}
        for category, emails in categorized_summaries.items():
            high_priority = [e for e in emails if e['priority'] in ['urgent', 'high']]
            action_items = [e for e in emails if e['action_required']]
            
            insights[category] = {
                'total_emails': len(emails),
                'high_priority_count': len(high_priority),
                'action_items_count': len(action_items),
                'emails': emails
            }
        
        return {
            "categorized_summaries": categorized_summaries,
            "insights": insights,
            "total_emails_analyzed": len(analyzed_emails),
            "message": "Smart summarization completed with category-specific insights"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smart summarization failed: {str(e)}")
