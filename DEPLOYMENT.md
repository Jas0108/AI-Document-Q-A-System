# Streamlit Cloud Deployment Guide

This guide will walk you through deploying your RAG application to Streamlit Cloud step-by-step.

## Prerequisites

Before you begin, make sure you have:
- ✅ A GitHub account
- ✅ A Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- ✅ Your API keys ready:
  - `GROQ_API_KEY` (required)
  - `HUGGINGFACE_API_KEY` (optional, but recommended)

---

## Step 1: Prepare Your GitHub Repository

### 1.1 Initialize Git (if not already done)

```bash
git init
```

### 1.2 Add All Files

```bash
git add .
```

### 1.3 Commit Your Code

```bash
git commit -m "Initial commit: RAG Document Q&A System"
```

### 1.4 Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Repository name: `gemma-rag-app` (or any name you prefer)
4. Description: "Dynamic RAG system for PDF document Q&A"
5. Choose **Public** (required for free Streamlit Cloud)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click **"Create repository"**

### 1.5 Push Your Code to GitHub

GitHub will show you commands. Use these (replace `YOUR_USERNAME` with your GitHub username):

```bash
git remote add origin https://github.com/YOUR_USERNAME/gemma-rag-app.git
git branch -M main
git push -u origin main
```

---

## Step 2: Sign Up for Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign up"** or **"Get started"**
3. Sign in with your **GitHub account** (recommended - easiest way)
4. Authorize Streamlit Cloud to access your GitHub repositories

---

## Step 3: Deploy Your App

### 3.1 Create New App

1. In Streamlit Cloud dashboard, click **"New app"**
2. You'll see a form to configure your app

### 3.2 Configure App Settings

Fill in the form:

- **Repository**: Select your repository (`YOUR_USERNAME/gemma-rag-app`)
- **Branch**: `main` (or `master` if you used that)
- **Main file path**: `app.py`
- **App URL**: Choose a unique name (e.g., `gemma-rag-app`)
  - This will create: `https://gemma-rag-app.streamlit.app`

### 3.3 Add Secrets (API Keys)

**IMPORTANT**: This is where you add your API keys securely.

1. Click **"Advanced settings"** → **"Secrets"**
2. You'll see a text editor. Add your secrets in TOML format:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
HUGGINGFACE_API_KEY = "your_huggingface_api_key_here"
```

**Example:**
```toml
GROQ_API_KEY = "gsk_abc123xyz..."
HUGGINGFACE_API_KEY = "hf_abc123xyz..."
```

3. Click **"Save"**

### 3.4 Deploy!

1. Click **"Deploy!"** button
2. Wait 2-3 minutes for the app to build and deploy
3. You'll see build logs in real-time

---

## Step 4: Verify Deployment

1. Once deployment completes, click **"View app"** or visit your app URL
2. Test the app:
   - Upload a PDF
   - Process documents
   - Ask a question
3. If everything works, you're done! 🎉

---

## Troubleshooting

### Build Fails

**Error: "Module not found"**
- Check `requirements.txt` includes all dependencies
- Make sure all imports in `app.py` are listed

**Error: "API key not found"**
- Double-check secrets are added correctly in Streamlit Cloud
- Verify secret names match exactly: `GROQ_API_KEY` and `HUGGINGFACE_API_KEY`
- Make sure there are no extra spaces or quotes in the secret values

### App Crashes on Load

**Check logs:**
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Click **"Manage app"** → **"Logs"**
4. Look for error messages

**Common issues:**
- Missing API keys → Add them in Secrets
- Import errors → Check `requirements.txt`
- Memory issues → Streamlit Cloud free tier has limits

### App Works But Can't Process Documents

- This is normal! Streamlit Cloud has ephemeral storage
- Users need to upload PDFs each session (this is expected behavior)
- The app is designed to work with uploaded files

---

## Updating Your App

Whenever you push changes to GitHub:

1. Streamlit Cloud will **automatically detect** the changes
2. It will show a **"Source has changed"** message
3. Click **"Always rerun"** or **"Rerun"** to deploy the update
4. Or it will auto-update if you enabled auto-rerun

---

## Pro Tips

1. **Keep your repository public** (required for free tier)
2. **Monitor your API usage** - Groq has rate limits
3. **Test locally first** before pushing to GitHub
4. **Use meaningful commit messages** - helps track changes
5. **Add a nice README** - makes your project look professional

---

## Next Steps

Once deployed:
- Share your app URL with recruiters/employers
- Add it to your portfolio/resume
- Update your GitHub README with the live link
- Consider adding a custom domain (paid feature)

---

## Support

- Streamlit Cloud Docs: [docs.streamlit.io/streamlit-community-cloud](https://docs.streamlit.io/streamlit-community-cloud)
- Streamlit Community: [discuss.streamlit.io](https://discuss.streamlit.io)

Good luck with your deployment! 🚀
