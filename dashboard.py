"""
Interactive Dashboard for PaleoDB Anomaly Detection
Displays model performance and provides anomaly detection interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import torch
from pathlib import Path

# Lazy imports - only import when needed to avoid startup failures
# from models import get_model
# from data_loader import PaleoDBLoader
# from inference import AnomalyDetector
# from evaluate import AnomalyDetectionEvaluator
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="PaleoDB Anomaly Detection Dashboard",
    page_icon="🦕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def load_checkpoint_info():
    """Load model checkpoint information"""
    # Get the absolute path to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, 'checkpoints')

    if not os.path.exists(checkpoint_dir):
        return None, None, None

    try:
        config_path = os.path.join(checkpoint_dir, 'config.json')
        metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
        model_path = os.path.join(checkpoint_dir, 'best_model.pt')

        # Check if required files exist
        if not os.path.exists(config_path) or not os.path.exists(metadata_path):
            return None, None, None

        # Also check if model file exists
        if not os.path.exists(model_path):
            return None, None, None

        with open(config_path, 'r') as f:
            config = json.load(f)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metrics_path = os.path.join(script_dir, 'evaluation_metrics.json')
        metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

        return config, metadata, metrics
    except FileNotFoundError:
        # Don't show error for missing files - this is expected before training
        return None, None, None
    except Exception as e:
        # Only show error for unexpected issues (e.g., corrupted JSON)
        st.error(f"Error loading checkpoint info: {str(e)}")
        return None, None, None


def render_home_page():
    """Render the home page"""
    st.markdown('<p class="main-header">🦕 PaleoDB Anomaly Detection Dashboard</p>', unsafe_allow_html=True)

    config, metadata, metrics = load_checkpoint_info()

    # Check if model exists - show prominent message if not
    if config is None:
        st.error("⚠️ **No Trained Model Found**")

        st.markdown("""
        ### 🚀 Quick Start Guide

        To use this dashboard, you need to train a model first. Here's how:

        **Option 1: Automatic Setup (Recommended)**
        ```bash
        ./setup.sh
        ```
        This script will:
        - Check for dependencies
        - Train the model automatically
        - Run evaluation
        - Set up everything for you

        **Option 2: Manual Training**
        ```bash
        # Install dependencies
        pip install -r requirements.txt

        # Train the model (takes 10-15 minutes)
        python train.py

        # Evaluate the model
        python evaluate.py
        ```

        **Option 3: Quick Training**
        """)

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("🚀 Train Model Now", type="primary", help="Start training (takes 10-15 minutes)"):
                with st.spinner("Training model... This will take several minutes."):
                    try:
                        import subprocess
                        import sys

                        # Get absolute path to ensure files are saved in the right place
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        train_script = os.path.join(script_dir, "train.py")

                        # Run training with explicit working directory
                        result = subprocess.run(
                            [sys.executable, train_script],
                            capture_output=True,
                            text=True,
                            cwd=script_dir,  # Set working directory
                            timeout=1800  # 30 minute timeout
                        )

                        if result.returncode == 0:
                            st.success("✅ Model trained successfully!")

                            # Verify checkpoint files were created
                            checkpoint_dir = os.path.join(script_dir, 'checkpoints')
                            model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                            config_path = os.path.join(checkpoint_dir, 'config.json')

                            if os.path.exists(model_path) and os.path.exists(config_path):
                                st.success(f"✅ Checkpoint files created at: {checkpoint_dir}")
                                st.info("Now running evaluation...")

                                # Run evaluation
                                eval_script = os.path.join(script_dir, "evaluate.py")
                                eval_result = subprocess.run(
                                    [sys.executable, eval_script],
                                    capture_output=True,
                                    text=True,
                                    cwd=script_dir,
                                    timeout=600
                                )

                                if eval_result.returncode == 0:
                                    st.success("✅ Evaluation completed!")
                                    st.balloons()
                                    st.info("🔄 Please refresh the page to see your model results!")

                                    # Add a refresh button
                                    if st.button("🔄 Refresh Page"):
                                        st.rerun()
                                else:
                                    st.warning("Model trained but evaluation failed. You can run `python evaluate.py` manually.")
                            else:
                                st.error(f"Training completed but checkpoint files not found at: {checkpoint_dir}")
                                st.info("This might be a permissions issue. Try running training manually with `python train.py`")
                        else:
                            st.error(f"❌ Training failed:\n```\n{result.stderr}\n```")

                    except subprocess.TimeoutExpired:
                        st.error("Training timed out. Please train manually using `python train.py`")
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        with col2:
            st.info("💡 Training takes 10-15 minutes. The model will be saved to `./checkpoints/` directory.")

        st.markdown("---")

        st.markdown("""
        ### 📖 About This System

        Once trained, this dashboard will provide:
        - **Model Performance Evaluation**: Interactive charts and comprehensive metrics
        - **Anomaly Detection Interface**: Real-time inference on PaleoDB data
        - **Rich Visualizations**: Plotly-powered interactive graphs

        The system uses **microsoft/deberta-v3-small** transformer architecture adapted for tabular anomaly detection.
        """)

        return

    # Normal home page content when model exists
    st.markdown("""
    ## Welcome to the PaleoDB Anomaly Detection System

    This dashboard showcases a state-of-the-art anomaly detection system for the Paleobiology Database using
    transformer-based deep learning models from Hugging Face.

    ### Features

    - **🎯 Model Performance**: Comprehensive evaluation metrics, ROC curves, precision-recall curves, and confusion matrices
    - **🔍 Anomaly Detection**: Interactive anomaly detection on PaleoDB fossil occurrence data
    - **📊 Visualizations**: Rich, interactive charts and graphs powered by Plotly
    - **🚀 Real-time Inference**: Detect anomalies in new data with instant results

    ### Model Architecture

    The system uses **microsoft/deberta-v3-small** transformer architecture adapted for tabular anomaly detection:
    - Input projection layer with LayerNorm
    - Multi-head self-attention transformer encoder
    - Compressed latent representation
    - Transformer decoder for reconstruction
    - Output projection to original feature space
    """)

    if config is not None:
        st.markdown("### 📋 Model Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Model Type", config.get('model_type', 'N/A').upper())

        with col2:
            st.metric("Input Dimension", metadata.get('input_dim', 'N/A'))

        with col3:
            st.metric("Training Samples", config.get('data_limit', 'N/A'))

        with col4:
            st.metric("Batch Size", config.get('batch_size', 'N/A'))

        if metrics:
            st.markdown("### 🎯 Quick Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

            with col2:
                st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")

            with col3:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")

            with col4:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    else:
        st.warning("⚠️ No trained model found. Please train a model first using `python train.py`")

    st.markdown("""
    ### 🚀 Get Started

    1. **Model Performance**: View detailed evaluation metrics and visualizations
    2. **Anomaly Detection**: Upload data or fetch from PaleoDB to detect anomalies

    Use the sidebar to navigate between pages.
    """)


def render_performance_page():
    """Render the model performance evaluation page"""
    # Lazy imports
    from models import get_model
    from evaluate import AnomalyDetectionEvaluator

    st.markdown('<p class="main-header">📊 Model Performance Evaluation</p>', unsafe_allow_html=True)

    config, metadata, metrics = load_checkpoint_info()

    if config is None or metrics is None:
        st.error("⚠️ No evaluation metrics found. Please run `python evaluate.py` first.")
        return

    # Load evaluation data
    st.markdown("## 📈 Performance Metrics")

    # Display metrics in cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Classification Metrics")
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")

    with col2:
        st.markdown("### Detection Quality")
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")

    with col3:
        st.markdown("### Configuration")
        st.metric("Threshold", f"{metrics.get('threshold', 0):.4f}")
        st.metric("Specificity", f"{metrics.get('specificity', 0):.4f}")
        st.metric("Model Type", config.get('model_type', 'N/A').upper())

    # Confusion Matrix
    st.markdown("## 🎯 Confusion Matrix")

    cm_data = metrics.get('confusion_matrix', {})
    tn, fp, fn, tp = cm_data.get('tn', 0), cm_data.get('fp', 0), cm_data.get('fn', 0), cm_data.get('tp', 0)

    cm_matrix = np.array([[tn, fp], [fn, tp]])

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_matrix,
        x=['Normal', 'Anomaly'],
        y=['Normal', 'Anomaly'],
        text=cm_matrix,
        texttemplate='%{text}',
        textfont={"size": 20},
        colorscale='Blues',
        showscale=True
    ))

    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500,
        font=dict(size=14)
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    # Re-generate evaluation plots
    st.markdown("## 📉 Performance Curves")

    # Check if we need to regenerate plots
    if st.button("🔄 Regenerate Evaluation Plots", help="Re-run evaluation to generate fresh plots"):
        with st.spinner("Running evaluation..."):
            try:
                # Import and run evaluation
                from train import prepare_data

                _, val_loader, _ = prepare_data(
                    limit=config['data_limit'],
                    batch_size=config['batch_size'],
                    val_split=config['val_split'],
                    anomaly_ratio=config['anomaly_ratio']
                )

                model = get_model(
                    model_type=config['model_type'],
                    input_dim=metadata['input_dim'],
                    dropout=0.1
                )

                evaluator = AnomalyDetectionEvaluator(
                    model=model,
                    checkpoint_path='./checkpoints/best_model.pt'
                )

                metrics_new, scores, labels, predictions = evaluator.evaluate(val_loader)

                # Create interactive plots
                create_performance_plots(scores, labels, predictions, metrics_new['threshold'])

                st.success("✅ Evaluation completed!")

            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
    else:
        st.info("Click the button above to generate fresh evaluation plots with interactive visualizations")

    # Model configuration details
    st.markdown("## ⚙️ Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Training Configuration")
        config_df = pd.DataFrame({
            'Parameter': ['Data Limit', 'Batch Size', 'Val Split', 'Anomaly Ratio', 'Model Type'],
            'Value': [
                config.get('data_limit', 'N/A'),
                config.get('batch_size', 'N/A'),
                config.get('val_split', 'N/A'),
                config.get('anomaly_ratio', 'N/A'),
                config.get('model_type', 'N/A')
            ]
        })
        st.dataframe(config_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Model Metadata")
        metadata_df = pd.DataFrame({
            'Parameter': ['Input Dimension', 'Feature Names Count'],
            'Value': [
                metadata.get('input_dim', 'N/A'),
                len(metadata.get('feature_names', [])) if metadata.get('feature_names') else 'N/A'
            ]
        })
        st.dataframe(metadata_df, use_container_width=True, hide_index=True)


def create_performance_plots(scores, labels, predictions, threshold):
    """Create interactive performance plots"""

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    # Create subplots
    col1, col2 = st.columns(2)

    with col1:
        # ROC Curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            showlegend=True,
            hovermode='closest'
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        # Precision-Recall Curve
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR (AUC = {pr_auc:.3f})',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig_pr.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=400,
            showlegend=True,
            hovermode='closest'
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Score Distribution
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=normal_scores,
        name='Normal',
        opacity=0.7,
        nbinsx=50,
        histnorm='probability density'
    ))
    fig_dist.add_trace(go.Histogram(
        x=anomaly_scores,
        name='Anomaly',
        opacity=0.7,
        nbinsx=50,
        histnorm='probability density'
    ))
    fig_dist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold"
    )
    fig_dist.update_layout(
        title='Anomaly Score Distribution',
        xaxis_title='Anomaly Score',
        yaxis_title='Density',
        height=400,
        barmode='overlay',
        showlegend=True,
        hovermode='closest'
    )
    st.plotly_chart(fig_dist, use_container_width=True)


def render_detection_page():
    """Render the anomaly detection page"""
    # Lazy imports
    from data_loader import PaleoDBLoader
    from inference import AnomalyDetector

    st.markdown('<p class="main-header">🔍 Anomaly Detection</p>', unsafe_allow_html=True)

    config, metadata, metrics = load_checkpoint_info()

    if config is None:
        st.error("⚠️ No trained model found. Please train a model first using `python train.py`")
        return

    st.markdown("""
    ## Detect Anomalies in PaleoDB Data

    Choose a data source and run anomaly detection on fossil occurrence records.
    """)

    # Data source selection
    data_source = st.radio(
        "Select Data Source:",
        ["Fetch from PaleoDB", "Upload CSV File"],
        horizontal=True
    )

    data = None

    if data_source == "Fetch from PaleoDB":
        st.markdown("### 📥 Fetch Data from PaleoDB")

        col1, col2 = st.columns(2)

        with col1:
            limit = st.number_input("Number of records to fetch", min_value=10, max_value=10000, value=500, step=100)

        with col2:
            base_name = st.text_input("Taxonomic group (optional)", value="", help="e.g., Dinosauria, Mammalia")

        if st.button("🔽 Fetch Data", type="primary"):
            with st.spinner(f"Fetching {limit} records from PaleoDB..."):
                try:
                    loader = PaleoDBLoader()
                    if base_name:
                        data = loader.download_paleodb_data(limit=limit, base_name=base_name)
                    else:
                        data = loader.download_paleodb_data(limit=limit)

                    st.success(f"✅ Successfully fetched {len(data)} records")
                    st.session_state['data'] = data
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")

    else:  # Upload CSV
        st.markdown("### 📤 Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(data)} records")
                st.session_state['data'] = data
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    # Display and analyze data
    if 'data' in st.session_state:
        data = st.session_state['data']

        st.markdown("### 📊 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)

        st.markdown(f"**Total Records**: {len(data)}")

        # Run anomaly detection
        st.markdown("### 🎯 Run Anomaly Detection")

        col1, col2 = st.columns(2)

        with col1:
            top_k = st.slider("Number of top anomalies to display", min_value=5, max_value=100, value=20, step=5)

        with col2:
            st.write("")  # Spacer

        if st.button("🚀 Detect Anomalies", type="primary"):
            with st.spinner("Running anomaly detection..."):
                try:
                    # Initialize detector
                    detector = AnomalyDetector(checkpoint_dir='./checkpoints')

                    # Run detection
                    results_df = detector.predict_and_explain(data, top_k=top_k)

                    st.session_state['results_df'] = results_df

                    # Display results
                    st.success("✅ Anomaly detection completed!")

                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display results
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']

        st.markdown("### 📊 Detection Results")

        # Summary metrics
        n_anomalies = results_df['is_anomaly'].sum()
        anomaly_rate = results_df['is_anomaly'].mean()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Samples", len(results_df))

        with col2:
            st.metric("Anomalies Detected", n_anomalies)

        with col3:
            st.metric("Anomaly Rate", f"{anomaly_rate*100:.2f}%")

        with col4:
            st.metric("Max Anomaly Score", f"{results_df['anomaly_score'].max():.4f}")

        # Visualizations
        st.markdown("### 📈 Anomaly Score Distribution")

        fig = px.histogram(
            results_df,
            x='anomaly_score',
            color='is_anomaly',
            nbins=50,
            title='Distribution of Anomaly Scores',
            labels={'anomaly_score': 'Anomaly Score', 'is_anomaly': 'Is Anomaly'},
            color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Top anomalies
        st.markdown(f"### 🔝 Top {top_k} Anomalies")

        top_anomalies = results_df.head(top_k)

        # Display with color coding
        def highlight_anomalies(row):
            if row['is_anomaly'] == 1:
                return ['background-color: #ffcccc'] * len(row)
            else:
                return [''] * len(row)

        st.dataframe(
            top_anomalies.style.apply(highlight_anomalies, axis=1),
            use_container_width=True,
            height=400
        )

        # Scatter plot of anomaly scores
        st.markdown("### 📍 Anomaly Score Scatter Plot")

        fig_scatter = px.scatter(
            results_df.reset_index(),
            x='index',
            y='anomaly_score',
            color='is_anomaly',
            title='Anomaly Scores Across All Samples',
            labels={'index': 'Sample Index', 'anomaly_score': 'Anomaly Score', 'is_anomaly': 'Is Anomaly'},
            color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'},
            hover_data=['anomaly_probability']
        )
        if metrics:
            fig_scatter.add_hline(
                y=metrics.get('threshold', 0.5),
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold"
            )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Download results
        st.markdown("### 💾 Download Results")

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )


def main():
    """Main application"""

    # Sidebar navigation
    st.sidebar.title("🦕 Navigation")

    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "📊 Model Performance", "🔍 Anomaly Detection"]
    )

    # Add file diagnostics in sidebar (expandable)
    with st.sidebar.expander("🔧 Diagnostics"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(script_dir, 'checkpoints')

        st.write("**Paths:**")
        st.code(f"Script dir: {script_dir}")
        st.code(f"Checkpoint dir: {checkpoint_dir}")

        st.write("**Files:**")
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            if files:
                for f in files:
                    st.text(f"✅ {f}")
            else:
                st.text("📁 Directory exists but is empty")
        else:
            st.text("❌ Checkpoint directory doesn't exist")

        # Check for model files
        model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        config_path = os.path.join(checkpoint_dir, 'config.json')
        metadata_path = os.path.join(checkpoint_dir, 'metadata.json')

        st.write("**Required files:**")
        st.text(f"{'✅' if os.path.exists(model_path) else '❌'} best_model.pt")
        st.text(f"{'✅' if os.path.exists(config_path) else '❌'} config.json")
        st.text(f"{'✅' if os.path.exists(metadata_path) else '❌'} metadata.json")

    st.sidebar.markdown("---")

    # Model info in sidebar
    config, metadata, metrics = load_checkpoint_info()

    if config is not None:
        st.sidebar.markdown("### 📋 Model Info")
        st.sidebar.markdown(f"**Model**: {config.get('model_type', 'N/A').upper()}")
        st.sidebar.markdown(f"**Input Dim**: {metadata.get('input_dim', 'N/A')}")
        if metrics:
            st.sidebar.markdown(f"**ROC-AUC**: {metrics.get('roc_auc', 0):.4f}")
    else:
        st.sidebar.warning("No model loaded")

    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    ### 📖 About

    This dashboard showcases a transformer-based anomaly detection system for PaleoDB data.

    **Features**:
    - Model performance evaluation
    - Interactive anomaly detection
    - Real-time visualizations

    **Model**: microsoft/deberta-v3-small
    """)

    # Render selected page
    if page == "🏠 Home":
        render_home_page()
    elif page == "📊 Model Performance":
        render_performance_page()
    elif page == "🔍 Anomaly Detection":
        render_detection_page()


if __name__ == "__main__":
    main()
