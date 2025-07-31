import os
from typing import Optional

class Settings:
    # Gmail OAuth Configuration
    GMAIL_CLIENT_ID: str = os.getenv("GMAIL_CLIENT_ID", "your-gmail-client-id")
    GMAIL_CLIENT_SECRET: str = os.getenv("GMAIL_CLIENT_SECRET", "your-gmail-client-secret")
    GMAIL_REDIRECT_URI: str = os.getenv("GMAIL_REDIRECT_URI", "http://localhost:8000/auth/gmail/callback")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Hugging Face Configuration
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # App Configuration
    APP_NAME: str = "Smart Email Assistant"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    
    # Database (if needed in future)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # Production settings
    IS_PRODUCTION: bool = os.getenv("IS_PRODUCTION", "False").lower() == "true"
    
    @property
    def base_url(self) -> str:
        """Get the base URL for the application."""
        if self.IS_PRODUCTION:
            return os.getenv("BASE_URL", "https://your-app-name.railway.app")
        return "http://localhost:8000"

settings = Settings() 