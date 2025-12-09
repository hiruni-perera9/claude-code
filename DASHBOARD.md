# 🦕 PaleoDB Anomaly Detection Dashboard

Interactive web dashboard for visualizing model performance and performing real-time anomaly detection on fossil occurrence data.

## Overview

The dashboard is built with **Streamlit** and **Plotly** to provide an intuitive interface for:
- Viewing model performance metrics and evaluation charts
- Running interactive anomaly detection on PaleoDB data
- Visualizing detection results with rich, interactive plots

## Quick Start

### Prerequisites

1. Train a model first:
```bash
python train.py
```

2. Run evaluation to generate metrics:
```bash
python evaluate.py
```

### Launch Dashboard

```bash
# Using the convenience script
./run_dashboard.sh

# Or directly with streamlit
streamlit run dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## Dashboard Pages

### 🏠 Home Page

The home page provides:
- Overview of the anomaly detection system
- Quick model information summary
- Key performance metrics at a glance
- Navigation guide

**Key Metrics Displayed**:
- Model type and architecture
- Input dimension and training configuration
- ROC-AUC, PR-AUC, F1-Score, and Accuracy

### 📊 Model Performance

Comprehensive evaluation of the trained model with interactive visualizations.

#### Metrics Section

Three panels showing:

1. **Classification Metrics**
   - ROC-AUC: Area under ROC curve
   - PR-AUC: Area under Precision-Recall curve
   - F1-Score: Harmonic mean of precision and recall

2. **Detection Quality**
   - Accuracy: Overall classification accuracy
   - Precision: True positives / (True positives + False positives)
   - Recall: True positives / (True positives + False negatives)

3. **Configuration**
   - Threshold: Optimal decision threshold
   - Specificity: True negatives / (True negatives + False positives)
   - Model type

#### Interactive Visualizations

1. **Confusion Matrix**
   - Heatmap showing True Negatives, False Positives, False Negatives, True Positives
   - Annotated with actual counts

2. **ROC Curve** (Generated on demand)
   - True Positive Rate vs False Positive Rate
   - Includes AUC score
   - Interactive hover for detailed values

3. **Precision-Recall Curve** (Generated on demand)
   - Precision vs Recall trade-off
   - Includes PR-AUC score
   - Useful for imbalanced datasets

4. **Score Distribution**
   - Histogram of anomaly scores
   - Separate distributions for normal and anomaly classes
   - Threshold line showing decision boundary

#### Configuration Details

Two tables showing:
- **Training Configuration**: Data limit, batch size, validation split, anomaly ratio, model type
- **Model Metadata**: Input dimensions, feature counts

#### Regenerate Evaluation

Click "🔄 Regenerate Evaluation Plots" to:
- Re-run evaluation on the validation set
- Generate fresh interactive plots
- Update all visualizations

### 🔍 Anomaly Detection

Interactive interface for detecting anomalies in new data.

#### Data Sources

**1. Fetch from PaleoDB**
- Directly download fossil occurrence records from the Paleobiology Database
- Configure number of records (10 to 10,000)
- Optional taxonomic filtering (e.g., "Dinosauria", "Mammalia")
- Click "🔽 Fetch Data" to download

**2. Upload CSV File**
- Upload your own CSV file with fossil occurrence data
- Must match the format used during training
- Drag-and-drop or click to browse

#### Data Preview

- View first 10 rows of loaded data
- Check data format and column names
- Verify record count

#### Run Detection

1. Select number of top anomalies to display (5-100)
2. Click "🚀 Detect Anomalies" to run inference
3. Wait for processing (progress indicator shown)

#### Results Display

**Summary Metrics**:
- Total samples processed
- Number of anomalies detected
- Anomaly rate (percentage)
- Maximum anomaly score

**Visualizations**:

1. **Anomaly Score Distribution**
   - Histogram of scores colored by anomaly status
   - Compare normal vs anomalous distributions
   - Interactive histogram with hover details

2. **Top Anomalies Table**
   - Displays top K most anomalous samples
   - Anomalies highlighted in red
   - Shows anomaly score, probability, and original data
   - Scrollable for easy browsing

3. **Anomaly Score Scatter Plot**
   - All samples plotted by index and score
   - Color-coded by anomaly status (normal/anomaly)
   - Threshold line showing decision boundary
   - Hover to see detailed information

#### Download Results

- Click "📥 Download Results as CSV" to export
- CSV includes all samples with:
  - Original data columns
  - `anomaly_score`: Reconstruction error
  - `is_anomaly`: Binary prediction (0=normal, 1=anomaly)
  - `anomaly_probability`: Probability-like score [0, 1]

## Features

### Interactive Charts

All visualizations are built with Plotly, providing:
- **Zoom**: Click and drag to zoom into regions
- **Pan**: Shift + drag to pan around
- **Hover**: See detailed values on hover
- **Toggle**: Click legend items to show/hide
- **Download**: Save plots as PNG images

### Real-time Processing

- Data fetching happens in real-time
- Progress indicators for long operations
- Asynchronous processing for smooth UX

### Session State

- Loaded data persists across interactions
- No need to re-fetch or re-upload
- Results cached for exploration

### Responsive Design

- Wide layout for maximum screen usage
- Responsive columns adapt to screen size
- Mobile-friendly (basic support)

## Configuration

### Port Configuration

Default port: `8501`

To change the port:
```bash
streamlit run dashboard.py --server.port 8080
```

### Theme

The dashboard uses Streamlit's default theme with custom CSS for:
- Metric cards with shadows
- Header styling
- Color-coded tables

To customize, edit the CSS in `dashboard.py`:
```python
st.markdown("""
<style>
    /* Your custom CSS here */
</style>
""", unsafe_allow_html=True)
```

### Data Limits

Maximum records to fetch from PaleoDB: 10,000 (configurable in code)

To increase:
```python
# In dashboard.py, find:
limit = st.number_input("Number of records to fetch",
                        min_value=10,
                        max_value=10000,  # Change this
                        value=500,
                        step=100)
```

## Troubleshooting

### Dashboard Won't Start

**Problem**: `streamlit: command not found`

**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### No Model Found

**Problem**: "No trained model found" warning

**Solution**: Train a model first
```bash
python train.py
python evaluate.py
```

### Port Already in Use

**Problem**: Port 8501 is already in use

**Solution**: Use a different port
```bash
streamlit run dashboard.py --server.port 8502
```

### Slow Performance

**Problem**: Dashboard is slow with large datasets

**Solutions**:
- Reduce number of records fetched
- Use sampling for visualization
- Close other browser tabs
- Increase system resources

### Visualization Not Showing

**Problem**: Charts don't appear

**Solutions**:
- Check browser console for errors
- Try refreshing the page
- Clear browser cache
- Update Plotly: `pip install --upgrade plotly`

### Data Format Errors

**Problem**: "Feature mismatch" warning during detection

**Cause**: Uploaded data has different features than training data

**Solution**:
- Ensure CSV has same columns as training data
- Use data from PaleoDB with same format
- Check preprocessing in `data_loader.py`

## Advanced Usage

### Custom Styling

Edit the CSS section in `dashboard.py` to customize:
- Colors and fonts
- Card layouts
- Spacing and margins
- Background colors

### Adding New Pages

To add a new page:

1. Create a new render function:
```python
def render_custom_page():
    st.markdown('<p class="main-header">Custom Page</p>', unsafe_allow_html=True)
    # Your content here
```

2. Add to navigation:
```python
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📊 Model Performance", "🔍 Anomaly Detection", "🆕 Custom"]
)

# Add routing
if page == "🆕 Custom":
    render_custom_page()
```

### Integrating Additional Models

To support multiple models:

1. Add model selection in sidebar:
```python
model_type = st.sidebar.selectbox("Select Model", ["huggingface", "transformer", "ensemble"])
```

2. Load model based on selection:
```python
detector = AnomalyDetector(
    checkpoint_dir=f'./checkpoints/{model_type}'
)
```

### Exporting Visualizations

To programmatically save charts:
```python
fig.write_html("chart.html")
fig.write_image("chart.png")
```

## Performance Tips

1. **Limit data size**: Start with smaller datasets (100-500 records)
2. **Use caching**: Streamlit automatically caches data
3. **Close unused pages**: Navigate away from resource-intensive pages
4. **Use production mode**: Deploy with `--server.headless true`

## Deployment

### Local Network Access

Allow access from other devices on your network:
```bash
streamlit run dashboard.py --server.address 0.0.0.0
```

Access from other devices: `http://YOUR_IP:8501`

### Cloud Deployment

**Streamlit Cloud** (Free):
1. Push code to GitHub
2. Go to share.streamlit.io
3. Deploy from repository

**Docker**:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "dashboard.py", "--server.port", "8501"]
```

## Technical Details

### Dependencies

- **streamlit**: Web framework
- **plotly**: Interactive visualizations
- **pandas**: Data manipulation
- **torch**: Model inference
- **sklearn**: Metrics calculation

### Architecture

```
dashboard.py
├── main() - Entry point
├── render_home_page() - Home page
├── render_performance_page() - Performance metrics
├── render_detection_page() - Anomaly detection
├── create_performance_plots() - Generate charts
└── load_checkpoint_info() - Load model info
```

### Data Flow

1. User selects data source
2. Data loaded into session state
3. Data preprocessed by `PaleoDBLoader`
4. Model inference via `AnomalyDetector`
5. Results displayed with Plotly
6. User can download results

## FAQ

**Q: Can I use the dashboard without training a model?**
A: The dashboard will load but functionality will be limited. Train a model first for full features.

**Q: How do I update the dashboard while it's running?**
A: Edit `dashboard.py` and save. Streamlit will prompt you to rerun.

**Q: Can I deploy this publicly?**
A: Yes, but ensure you have necessary data permissions and consider security implications.

**Q: Does the dashboard support multiple users?**
A: Each user gets their own session. Data is not shared between users.

**Q: Can I customize the theme?**
A: Yes, edit the CSS in `dashboard.py` or use Streamlit's theme customization.

## Support

For issues or questions:
1. Check this documentation
2. Review error messages in terminal
3. Check Streamlit logs
4. Open an issue on GitHub

## Deployment

Want to deploy your dashboard to the cloud? See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive deployment guides.

**Supported Platforms**:
- Streamlit Cloud (Recommended - Free)
- Render (Free tier available)
- Railway ($5/month credit)
- Heroku (Paid)
- Docker (Any cloud platform)

⚠️ **Important**: Vercel is NOT compatible with Streamlit. Use one of the platforms above.

## License

Same as the main project (MIT License)
