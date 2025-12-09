# 🚀 Deployment Guide for PaleoDB Anomaly Detection Dashboard

This guide covers deploying the Streamlit dashboard to various cloud platforms. **Note: Vercel is not compatible with Streamlit apps.** Use one of the platforms below instead.

## 📋 Table of Contents

- [Why Not Vercel?](#why-not-vercel)
- [Recommended Platforms](#recommended-platforms)
- [Option 1: Streamlit Cloud (Easiest - FREE)](#option-1-streamlit-cloud-easiest---free)
- [Option 2: Render (Simple - FREE)](#option-2-render-simple---free)
- [Option 3: Railway (Modern - FREE Trial)](#option-3-railway-modern---free-trial)
- [Option 4: Heroku (Classic)](#option-4-heroku-classic)
- [Option 5: Docker (Any Cloud)](#option-5-docker-any-cloud)
- [Pre-deployment Checklist](#pre-deployment-checklist)
- [Important Notes](#important-notes)

---

## Why Not Vercel?

**Vercel is NOT compatible with Streamlit** because:
- Vercel is designed for **static sites** and **serverless functions** (Next.js, React, Vue, etc.)
- Streamlit requires a **continuously running Python server**
- Vercel's serverless architecture doesn't support long-running processes

**Better alternatives**: Streamlit Cloud, Render, Railway, Heroku, or containerized deployments.

---

## Recommended Platforms

| Platform | Difficulty | Free Tier | Best For |
|----------|-----------|-----------|----------|
| **Streamlit Cloud** | ⭐ Easiest | ✅ Yes | Quick deployment, Streamlit-optimized |
| **Render** | ⭐⭐ Easy | ✅ Yes | Production apps, auto-deploy |
| **Railway** | ⭐⭐ Easy | ✅ Trial | Modern workflow, Docker support |
| **Heroku** | ⭐⭐⭐ Medium | ❌ No* | Classic PaaS, well-documented |
| **Docker** | ⭐⭐⭐⭐ Advanced | Varies | Full control, any cloud platform |

*Heroku ended free tier in November 2022

---

## Option 1: Streamlit Cloud (Easiest - FREE)

**Best choice for Streamlit apps!** Free, easy, and optimized for Streamlit.

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

### Steps

1. **Push your code to GitHub** (already done!)
   ```bash
   git push origin claude/create-model-dashboard-0145wPDJRWtiK9zkvQeSVoRL
   ```

2. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Click "Sign up" or "Sign in" with GitHub

3. **Deploy Your App**
   - Click "New app"
   - Select your repository: `hiruni-perera9/claude-code`
   - Select branch: `claude/create-model-dashboard-0145wPDJRWtiK9zkvQeSVoRL`
   - Main file path: `dashboard.py`
   - Click "Deploy!"

4. **Wait for Deployment**
   - Streamlit Cloud will install dependencies from `requirements.txt`
   - Your app will be live at: `https://your-app-name.streamlit.app`

### Important Notes for Streamlit Cloud

⚠️ **Model Checkpoint Issue**: The trained model checkpoints are NOT in the repository (too large for Git).

**Solutions**:

**A. Train Model on First Run** (Recommended for demo):
- Add a startup script to train the model if checkpoints don't exist
- Create `.streamlit/secrets.toml` for any API keys if needed

**B. Use Pre-trained Models**:
- Store model checkpoints in cloud storage (S3, Google Cloud Storage)
- Download on startup

**C. Add Startup Hook**:
Create `setup.sh` in your repo:
```bash
#!/bin/bash
# Download or train model if not exists
if [ ! -d "checkpoints" ]; then
    echo "Training model..."
    python train.py
fi
```

Then in Streamlit Cloud settings, set "Advanced settings" > "Command to run": `bash setup.sh && streamlit run dashboard.py`

### Configuration

The `.streamlit/config.toml` file is already set up with optimized settings.

---

## Option 2: Render (Simple - FREE)

Render offers a generous free tier with auto-deploy from Git.

### Steps

1. **Create Render Account**
   - Visit: https://render.com
   - Sign up with GitHub

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select `hiruni-perera9/claude-code`
   - Select branch: `claude/create-model-dashboard-0145wPDJRWtiK9zkvQeSVoRL`

3. **Configure Service**
   - **Name**: `paleodb-anomaly-dashboard`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run dashboard.py --server.port=$PORT --server.headless=true`
   - **Plan**: `Free`

4. **Advanced Settings** (Optional)
   - Set environment variables if needed
   - Health check path: `/_stcore/health`

5. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for build
   - Your app will be live at: `https://your-app-name.onrender.com`

### Configuration

The `render.yaml` file is already configured. To use it:
- In Render dashboard, use "Blueprint" option
- Select `render.yaml`
- Render will auto-configure everything

### Important Notes
- Free tier spins down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds
- Upgrade to paid tier for always-on

---

## Option 3: Railway (Modern - FREE Trial)

Railway offers $5 free credit per month and modern deployment workflow.

### Steps

1. **Create Railway Account**
   - Visit: https://railway.app
   - Sign up with GitHub

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `hiruni-perera9/claude-code`
   - Select branch: `claude/create-model-dashboard-0145wPDJRWtiK9zkvQeSVoRL`

3. **Configure Deployment**
   - Railway auto-detects Python and requirements.txt
   - Add environment variable: `PORT=8501`
   - Start command is auto-detected from `railway.json`

4. **Generate Domain**
   - Go to Settings → Networking
   - Click "Generate Domain"
   - Your app will be live at: `https://your-app-name.up.railway.app`

### Configuration

The `railway.json` file is already configured for automatic deployment.

### Important Notes
- $5/month free credit (usually enough for hobby projects)
- Usage-based pricing after free credit
- Excellent for production workloads

---

## Option 4: Heroku (Classic)

Heroku no longer has a free tier, but is still a solid choice for production.

### Prerequisites
- Heroku CLI installed: https://devcenter.heroku.com/articles/heroku-cli
- Heroku account

### Steps

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku

   # Ubuntu
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku App**
   ```bash
   cd /path/to/claude-code
   heroku create paleodb-anomaly-dashboard
   ```

4. **Deploy**
   ```bash
   git push heroku claude/create-model-dashboard-0145wPDJRWtiK9zkvQeSVoRL:main
   ```

5. **Open App**
   ```bash
   heroku open
   ```

### Configuration

The `Procfile` and `runtime.txt` files are already configured.

### Pricing
- **Eco Dynos**: $5/month (sleeps after 30 minutes)
- **Basic**: $7/month (never sleeps)
- **Standard**: $25-50/month (production features)

---

## Option 5: Docker (Any Cloud)

Deploy anywhere using Docker containers.

### Build Docker Image

```bash
# Build the image
docker build -t paleodb-anomaly-dashboard .

# Test locally
docker run -p 8501:8501 paleodb-anomaly-dashboard
```

### Deploy to Various Platforms

#### Google Cloud Run
```bash
# Tag for Google Container Registry
docker tag paleodb-anomaly-dashboard gcr.io/YOUR_PROJECT/paleodb-dashboard

# Push to GCR
docker push gcr.io/YOUR_PROJECT/paleodb-dashboard

# Deploy
gcloud run deploy paleodb-dashboard \
  --image gcr.io/YOUR_PROJECT/paleodb-dashboard \
  --platform managed \
  --allow-unauthenticated
```

#### AWS ECS/Fargate
```bash
# Tag for ECR
docker tag paleodb-anomaly-dashboard AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/paleodb-dashboard

# Push to ECR
docker push AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/paleodb-dashboard

# Deploy using AWS Console or CLI
```

#### Azure Container Instances
```bash
# Login to Azure
az login

# Create resource group
az group create --name paleodb-rg --location eastus

# Deploy container
az container create \
  --resource-group paleodb-rg \
  --name paleodb-dashboard \
  --image paleodb-anomaly-dashboard \
  --dns-name-label paleodb-dashboard \
  --ports 8501
```

#### DigitalOcean App Platform
- Use the DigitalOcean App Platform UI
- Connect GitHub repository
- Use Dockerfile for deployment
- DigitalOcean will auto-detect and build

---

## Pre-deployment Checklist

Before deploying, make sure:

- [ ] All code is committed and pushed to GitHub
- [ ] `requirements.txt` includes all dependencies
- [ ] Trained model checkpoints exist OR you have a training strategy
- [ ] Environment variables are documented (if any)
- [ ] `.gitignore` excludes large files (`checkpoints/`, `data/`)
- [ ] Dashboard runs locally without errors
- [ ] Port configuration is flexible (uses `$PORT` environment variable)

---

## Important Notes

### Model Checkpoints

⚠️ **Critical Issue**: Your trained model checkpoints are NOT in the repository.

**Options**:

1. **Train on Startup** (for demo/development):
   ```python
   # Add to dashboard.py
   import os
   if not os.path.exists('./checkpoints/best_model.pt'):
       import subprocess
       subprocess.run(['python', 'train.py'])
   ```

2. **Cloud Storage** (for production):
   - Upload checkpoints to S3/GCS/Azure Blob
   - Download on app startup
   - Cache locally during runtime

3. **Git LFS** (Large File Storage):
   ```bash
   # Install Git LFS
   git lfs install

   # Track checkpoint files
   git lfs track "*.pt"
   git add .gitattributes
   git add checkpoints/*.pt
   git commit -m "Add model checkpoints with Git LFS"
   ```

### Environment Variables

If you need API keys or secrets:

**Streamlit Cloud**: Add to app settings → Secrets
**Render/Railway/Heroku**: Add in dashboard → Environment Variables

Example `.streamlit/secrets.toml`:
```toml
[paleodb]
api_key = "your-api-key"

[model]
checkpoint_url = "https://your-cloud-storage/model.pt"
```

### Performance Optimization

For production deployments:

1. **Optimize Requirements**
   - Remove unnecessary packages
   - Use `--no-cache-dir` for pip install

2. **Enable Caching**
   ```python
   @st.cache_data
   def load_data():
       # Your data loading code
       pass
   ```

3. **Lazy Loading**
   - Load models only when needed
   - Don't load heavy dependencies on startup

4. **Memory Management**
   - Free tier platforms have limited RAM (512MB-1GB)
   - Monitor memory usage
   - Consider model quantization for smaller size

### Monitoring

Add health checks and monitoring:

```python
# In dashboard.py
import logging
logging.basicConfig(level=logging.INFO)

# Log important events
logging.info("Dashboard started")
logging.info(f"Model loaded: {config['model_type']}")
```

---

## Troubleshooting

### Build Failures

**Issue**: Deployment fails during pip install

**Solution**:
- Check `requirements.txt` for incompatible versions
- Try PyTorch CPU-only version for smaller size:
  ```
  torch>=2.0.0; platform_machine != "arm64"
  ```

### Port Binding Errors

**Issue**: App doesn't respond to requests

**Solution**: Ensure using `$PORT` environment variable
```python
# In dashboard start command
streamlit run dashboard.py --server.port=$PORT
```

### Out of Memory

**Issue**: App crashes with memory errors

**Solution**:
- Upgrade to paid tier with more RAM
- Reduce model size
- Use model quantization
- Optimize data loading

### Slow Cold Starts

**Issue**: First request takes 30+ seconds

**Solution**:
- Upgrade to paid tier (no sleep)
- Optimize startup time
- Use health check pings to keep warm

---

## Recommended Deployment Strategy

### For Demo/Testing
→ **Streamlit Cloud** (free, easy, perfect for Streamlit)

### For Production
→ **Railway** or **Render** (paid tier for always-on, auto-scaling)

### For Enterprise
→ **Docker on GCP/AWS/Azure** (full control, custom infrastructure)

---

## Next Steps

1. Choose a platform from above
2. Follow the deployment steps
3. Test your deployed dashboard
4. Set up monitoring and alerts
5. Configure custom domain (optional)

## Support

For deployment issues:
- **Streamlit Cloud**: https://discuss.streamlit.io
- **Render**: https://render.com/docs
- **Railway**: https://docs.railway.app
- **General**: Open an issue on GitHub

---

**Remember**: Vercel is NOT compatible with Streamlit. Use one of the platforms above for successful deployment! 🚀
