# 🚀 Quick Start Guide

Get up and running with the PaleoDB Anomaly Detection Dashboard in 3 simple steps!

## 🎯 The Fastest Way

```bash
# 1. Run automated setup
./setup.sh

# 2. Launch the dashboard
./run_dashboard.sh
```

That's it! The dashboard will open at `http://localhost:8501` 🎉

---

## 📝 What's Happening?

### Step 1: Setup (`./setup.sh`)

The setup script will:
- ✅ Check if a trained model exists
- ✅ Install Python dependencies (PyTorch, Transformers, etc.)
- ✅ Download PaleoDB data
- ✅ Train the anomaly detection model (~10-15 minutes)
- ✅ Evaluate the model and generate metrics

**First time?** This will take about 15 minutes. Grab a coffee! ☕

**Already have a model?** Setup completes instantly.

### Step 2: Launch (`./run_dashboard.sh`)

Starts the Streamlit dashboard with:
- 🏠 Home page with model overview
- 📊 Model performance metrics and charts
- 🔍 Interactive anomaly detection

---

## ⚠️ Troubleshooting

### "No trained model found" Error

**Solution**: Run the setup script first!
```bash
./setup.sh
```

### Setup Script Won't Run

**Issue**: Permission denied

**Solution**: Make it executable
```bash
chmod +x setup.sh
chmod +x run_dashboard.sh
```

### Dependencies Not Installed

**Solution**: Install manually
```bash
pip install -r requirements.txt
```

### Training Takes Too Long

**Solution**: Edit `train.py` to use less data
```python
# In train.py, change:
config = {
    'data_limit': 1000,  # Reduced from 10000
    # ... other settings
}
```

### Python Version Issues

**Requirement**: Python 3.8 or higher

**Check version**:
```bash
python --version
```

**Upgrade if needed**: Use pyenv, conda, or your system's package manager

---

## 🎓 Manual Setup (If You Prefer)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train.py
```
⏱️ Takes 10-15 minutes

### Step 3: Evaluate the Model
```bash
python evaluate.py
```
⏱️ Takes 2-3 minutes

### Step 4: Launch Dashboard
```bash
streamlit run dashboard.py
```

---

## 🌐 Deploying to the Cloud?

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for detailed instructions on deploying to:
- Streamlit Cloud (Free, easiest)
- Render (Free tier available)
- Railway ($5/month credit)
- Docker (Any cloud platform)

⚠️ **Important**: Vercel is NOT compatible with Streamlit apps!

---

## 📖 More Information

- **[README.md](README.md)** - Complete documentation
- **[DASHBOARD.md](DASHBOARD.md)** - Dashboard features and usage
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Cloud deployment guide
- **[MODEL_SELECTION.md](MODEL_SELECTION.md)** - Model architecture details

---

## 🆘 Still Having Issues?

1. Check that you have Python 3.8+
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Try running setup.sh with verbose output: `bash -x setup.sh`
4. Check the terminal for error messages
5. Open an issue on GitHub with the error details

---

## ⚡ Quick Command Reference

```bash
# First time setup
./setup.sh

# Launch dashboard
./run_dashboard.sh

# Manual training
python train.py

# Manual evaluation
python evaluate.py

# Run inference
python inference.py --limit 1000

# Check Python version
python --version

# Install dependencies
pip install -r requirements.txt
```

---

**Ready?** Run `./setup.sh` to get started! 🚀
