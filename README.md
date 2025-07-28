# Smart Email Assistant

Automates email classification, summarization, and reply using LLMs and ML models.

## Setup

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Create a `.env` file** in the project root with:
   ```env
   GOOGLE_CLIENT_ID=your-google-client-id
   GOOGLE_CLIENT_SECRET=your-google-client-secret
   GOOGLE_REDIRECT_URI=http://localhost:8000/auth/gmail/callback
   OPENAI_API_KEY=your-openai-api-key
   ```
4. **Run the FastAPI server:**
   ```bash
   uvicorn app.app:app --reload
   ```

## Endpoints
- `GET /auth/gmail/initiate` — Start Gmail OAuth2 flow
- `GET /auth/gmail/callback` — Handle OAuth2 callback
- `GET /emails` — Fetch emails
- `POST /emails/summarize` — Summarize email thread
- `POST /emails/classify` — Classify email
- `POST /emails/reply` — Generate and send auto-reply

## Project Structure
- `app/app.py` — FastAPI app and endpoints
- `email_service/gmail_service.py` — Gmail API integration
- `llm_service/openai_service.py` — OpenAI API integration
- `core/config.py` — Environment/config loader

## Notes
- You must register your app in the Google Cloud Console to get Gmail OAuth credentials.
- You need an OpenAI API key for LLM features.
