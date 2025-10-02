# 🚀 Smart Email Assistant - Portfolio Project

A modern, AI-powered email management system that helps users analyze, categorize, and summarize their Gmail messages using advanced machine learning techniques.

## ✨ Features

### 🔐 Secure Authentication
- **OAuth2 Integration**: Secure Gmail authentication using Google's OAuth2
- **Production Ready**: Available to all users (not just test users)
- **HTTPS Protected**: Secure data transmission

### 🤖 AI-Powered Analysis
- **Email Classification**: Automatically categorize emails by type (work, personal, finance, etc.)
- **Priority Detection**: Identify urgent and action-required emails
- **Smart Summarization**: Generate concise summaries of email threads
- **Sentiment Analysis**: Analyze email tone and sentiment
- **Multi-Model Support**: OpenAI and Hugging Face integration

### 📊 Advanced Features
- **Real-time Processing**: Instant email analysis and insights
- **Category Visualization**: Interactive charts showing email distribution
- **Priority Filtering**: Focus on important emails first
- **Thread Summarization**: Condense long email conversations
- **Auto-Reply Generation**: AI-powered response suggestions

### 🎨 Modern UI/UX
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Beautiful Interface**: Modern UI with Tailwind CSS
- **Real-time Updates**: Live progress indicators and status updates
- **Intuitive Navigation**: Easy-to-use interface for all users

## 🛠️ Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Python 3.11**: Latest Python features and performance
- **Google Gmail API**: Secure email access and management
- **OAuth2**: Industry-standard authentication

### AI/ML
- **OpenAI API**: Advanced language model integration
- **Hugging Face**: Open-source AI model support
- **Scikit-learn**: Machine learning for email classification
- **Pandas**: Data manipulation and analysis

### Frontend
- **HTML5/CSS3**: Modern web standards
- **Tailwind CSS**: Utility-first CSS framework
- **JavaScript**: Interactive user experience
- **Font Awesome**: Beautiful icons

### Deployment
- **Railway**: Modern deployment platform
- **Render**: Alternative cloud hosting
- **Docker**: Containerization support
- **GitHub**: Version control and CI/CD

## 🚀 Quick Start

### Option 1: Deploy to Railway (Recommended)

1. **Fork this repository** to your GitHub account
2. **Sign up for Railway** at https://railway.app
3. **Connect your GitHub** repository
4. **Set environment variables** in Railway dashboard:
   ```
   GOOGLE_CLIENT_ID=your-google-client-id
   GOOGLE_CLIENT_SECRET=your-google-client-secret
   GOOGLE_REDIRECT_URI=https://your-app.railway.app/auth/gmail/callback
   OPENAI_API_KEY=your-openai-key (optional)
   HF_API_KEY=your-huggingface-key (optional)
   IS_PRODUCTION=true
   BASE_URL=https://your-app.railway.app
   ```
5. **Deploy automatically** - Railway will detect the configuration

### Option 2: Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/smart-email-assistant.git
   cd smart-email-assistant
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env_template.txt .env
   # Edit .env with your API keys
   ```

4. **Run the application**:
   ```bash
   uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Open your browser** to http://localhost:8000

## 🔧 Setup Instructions

### 1. Google Cloud Console Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Gmail API:
   - Go to "APIs & Services" > "Library"
   - Search for "Gmail API" and enable it
4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
   - Choose "Web application"
   - Add authorized redirect URIs:
     - Local: `http://localhost:8000/auth/gmail/callback`
     - Production: `https://your-app.railway.app/auth/gmail/callback`

### 2. Make App Available to All Users

1. Go to Google Cloud Console > "APIs & Services" > "OAuth consent screen"
2. Click "PUBLISH APP" to make it available to all users
3. Add app information:
   - App name: "Smart Email Assistant"
   - User support email: Your email
   - App domain: Your deployment URL
4. Add required scopes:
   - `https://www.googleapis.com/auth/gmail.readonly`
   - `https://www.googleapis.com/auth/userinfo.email`

### 3. API Keys (Optional)

For enhanced AI features, get API keys from:
- **OpenAI**: https://platform.openai.com/api-keys
- **Hugging Face**: https://huggingface.co/settings/tokens

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application interface |
| `/health` | GET | Health check endpoint |
| `/docs` | GET | Interactive API documentation |
| `/auth/gmail/initiate` | GET | Start Gmail OAuth flow |
| `/auth/gmail/callback` | GET | OAuth callback handler |
| `/emails` | GET | Fetch user emails |
| `/emails/analyze/advanced` | POST | Advanced AI analysis |
| `/emails/summarize/smart` | POST | Smart summarization |
| `/emails/classify` | POST | Classify individual email |
| `/emails/summarize` | POST | Summarize email thread |
| `/emails/reply` | POST | Generate auto-reply |

### Technical Achievements
- **Full-Stack Development**: Complete web application with modern architecture
- **AI/ML Integration**: Advanced email analysis using multiple AI models
- **Security Implementation**: OAuth2 authentication and secure data handling
- **Cloud Deployment**: Production-ready deployment with monitoring
- **API Design**: RESTful API with comprehensive documentation

### User Experience
- **Intuitive Interface**: Beautiful, responsive design
- **Real-time Processing**: Live updates and progress indicators
- **Accessibility**: Works across all devices and browsers
- **Performance**: Fast loading and efficient processing

### Scalability
- **Microservices Ready**: Modular architecture for easy scaling
- **Container Support**: Docker configuration for deployment
- **Monitoring**: Health checks and logging
- **CI/CD Ready**: Automated deployment pipeline

## 🔒 Security & Privacy

- **OAuth2 Authentication**: Industry-standard security
- **HTTPS Only**: Encrypted data transmission
- **No Data Storage**: Emails are processed in-memory only
- **User Consent**: Clear permission requests
- **Production Ready**: Secure for public use

## 🧪 Testing

Run the test suite to ensure everything works:

```bash
# Test basic functionality
python test_advanced_features.py

# Test API endpoints
python test_api_endpoints.py

# Test OAuth flow
python test_oauth_flow.py
```

## 📈 Performance

- **Fast Response Times**: Optimized for quick email analysis
- **Efficient Processing**: Smart caching and batching
- **Scalable Architecture**: Handles multiple concurrent users
- **Resource Optimized**: Minimal memory and CPU usage






