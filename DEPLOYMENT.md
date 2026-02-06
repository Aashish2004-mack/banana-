# Deployment Guide

## Option 1: Render (Recommended - Free)

1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository: `https://github.com/Aashish2004-mack/banana-.git`
4. Configure:
   - **Name**: banana-disease-classifier
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click "Create Web Service"
6. Wait 5-10 minutes for deployment

**Note**: Free tier may be slow due to large PyTorch dependencies.

## Option 2: Railway (Free)

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Python and deploys

## Option 3: Hugging Face Spaces (Best for ML)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space → Gradio/Streamlit
3. Upload your code
4. Add model files

## Option 4: Local Hosting with ngrok

```bash
# Install ngrok
pip install pyngrok

# Run your app
python app.py

# In another terminal
ngrok http 5000
```

This gives you a public URL like: `https://xxxx.ngrok.io`

## Option 5: PythonAnywhere (Free)

1. Sign up at [pythonanywhere.com](https://pythonanywhere.com)
2. Upload your code
3. Configure WSGI file
4. Set up virtual environment

## Important Notes

- GitHub Pages only hosts static HTML/CSS/JS (no Python backend)
- For production, use Render, Railway, or AWS/GCP
- Model files are large - consider using cloud storage
- Free tiers have limitations (CPU, memory, uptime)

## Quick Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Click the button above
2. Connect GitHub
3. Select repository
4. Deploy!