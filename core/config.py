import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_CLIENT_ID: str = os.getenv('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET: str = os.getenv('GOOGLE_CLIENT_SECRET')
    GOOGLE_REDIRECT_URI: str = os.getenv('GOOGLE_REDIRECT_URI')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')

settings = Settings() 