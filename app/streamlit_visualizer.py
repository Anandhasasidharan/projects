import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import logging
import time
import io
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import warnings

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import SHAP, with fallback if it fails
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP loaded successfully")
except ImportError as e:
    logger.warning(f"SHAP not available: {str(e)}")
    logger.info("App will run without SHAP explanations")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# App configuration
st.set_page_config(
    page_title="Cyber Threat Risk Scorer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration constants
MODEL_PATH = "models/risk_score_model.pkl"
FEATURES_PATH = "data/feature_columns.csv"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_ROWS_FOR_SHAP = 1000

class SecurityValidation:
    """Security validation utilities."""
    
    @staticmethod
    def validate_file_size(file) -> bool:
        """Check if file size is within limits."""
        if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
            return False
        return True
    
    @staticmethod
    def validate_csv_content(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate CSV content for security and format."""
        if df.empty:
            return False, "File is empty"
        
        if len(df) > 100000:  # Limit rows for performance
            return False, "File too large. Maximum 100,000 rows allowed"
        
        # Check for suspicious column names
        suspicious_patterns = ['script', 'exec', 'eval', 'import', '__']
        for col in df.columns:
            if any(pattern in str(col).lower() for pattern in suspicious_patterns):
                return False, f"Suspicious column name detected: {col}"
        
        return True, "Valid"

@st.cache_resource
def load_model_and_features() -> Tuple[Optional[Any], Optional[list], str]:
    """Load model and feature names with comprehensive error handling."""
    try:
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model file not found at {MODEL_PATH}"
            logger.error(error_msg)
            return None, None, error_msg
            
        if not os.path.exists(FEATURES_PATH):
            error_msg = f"Feature columns file not found at {FEATURES_PATH}"
            logger.error(error_msg)
            return None, None, error_msg
        
        # Load model
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Load features
        features_df = pd.read_csv(FEATURES_PATH)
        if 'feature' not in features_df.columns:
            error_msg = "Feature file must contain a 'feature' column"
            logger.error(error_msg)
            return None, None, error_msg
            
        feature_names = features_df['feature'].tolist()
        logger.info(f"Loaded {len(feature_names)} features")
        
        return model, feature_names, "Success"
        
    except Exception as e:
        error_msg = f"Error loading model or features: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg

def validate_input_data(df: pd.DataFrame, feature_names: list) -> Tuple[bool, list]:
    """Comprehensive input data validation."""
    issues = []
    
    # Security validation
    is_secure, security_msg = SecurityValidation.validate_csv_content(df)
    if not is_secure:
        issues.append(f"🚫 Security: {security_msg}")
        return False, issues
    
    # Data quality checks
    if df.empty:
        issues.append("📋 File is empty")
        
    # Check for excessive missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_percentage[missing_percentage > 50]
    if not high_missing.empty:
        issues.append(f"⚠️ Columns with >50% missing values: {', '.join(high_missing.index)}")
    
    # Check data types
    non_numeric_cols = df.select_dtypes(exclude=[np.number, 'bool']).columns
    if len(non_numeric_cols) > 0:
        logger.info(f"Non-numeric columns detected: {list(non_numeric_cols)}")
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if np.isinf(numeric_df).any().any():
        issues.append("⚠️ Infinite values detected in numeric columns")
    
    return len(issues) == 0, issues

def preprocess_data(df: pd.DataFrame, feature_names: list) -> Tuple[Optional[pd.DataFrame], bool, list]:
    """Advanced data preprocessing with validation."""
    try:
        messages = []
        
        # Handle missing values
        if df.isnull().any().any():
            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
            
            messages.append("✅ Missing values handled")
        
        # Create encoded dataframe
        input_encoded = pd.get_dummies(df, dummy_na=False)
        
        # Handle feature alignment
        missing_features = []
        extra_features = []
        
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
                missing_features.append(col)
        
        # Remove extra columns not in training features
        for col in input_encoded.columns:
            if col not in feature_names:
                extra_features.append(col)
        
        # Reorder columns to match training order
        input_encoded = input_encoded[feature_names]
        
        if missing_features:
            messages.append(f"⚠️ Added {len(missing_features)} missing features as zeros")
        
        if extra_features:
            messages.append(f"ℹ️ Removed {len(extra_features)} extra features not used in training")
        
        return input_encoded, True, messages
        
    except Exception as e:
        error_msg = f"Preprocessing error: {str(e)}"
        logger.error(error_msg)
        return None, False, [error_msg]

# Initialize app
st.title("🔐 Cyber Threat Risk Scorer")
st.markdown("### Advanced Network Anomaly Severity Prediction")
st.markdown("---")

# Load model and features
model, feature_names, load_status = load_model_and_features()

# Sidebar information
with st.sidebar:
    st.header("📊 Model Information")
    
    if model is not None:
        st.success("✅ Model Status: Ready")
        st.info(f"📈 Model Type: {type(model).__name__}")
        st.info(f"🔢 Features: {len(feature_names) if feature_names else 0}")
        
        # Display sample of expected features
        if feature_names:
            with st.expander("Show Expected Features (Sample)"):
                st.write(feature_names[:10] + ["..."] if len(feature_names) > 10 else feature_names)
    else:
        st.error(f"❌ {load_status}")
        st.info("Please check the model and feature files.")

    st.markdown("---")
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. Upload a CSV file with network traffic features
    2. Review data validation results
    3. View predicted severity scores
    4. Analyze SHAP explanations
    """)
    
    st.markdown("---")
    st.header("⚙️ Settings")
    if SHAP_AVAILABLE:
        max_shap_samples = st.slider("Max samples for SHAP", 10, 1000, 100)
    else:
        st.warning("SHAP not available")
        max_shap_samples = 100  # Default value when SHAP is not available

# Main content
if model is None or feature_names is None:
    st.error("❌ Cannot proceed without model and features. Please check the files.")
    st.stop()

uploaded_file = st.file_uploader(
    "📁 Upload CSV with traffic features", 
    type=["csv"],
    help="Upload a CSV file containing network traffic features for threat severity prediction"
)

if uploaded_file:
    try:
        # File size validation
        if not SecurityValidation.validate_file_size(uploaded_file):
            st.error(f"❌ File too large. Maximum size allowed: {MAX_FILE_SIZE / (1024*1024):.1f} MB")
            st.stop()
        
        # Progress indicator
        with st.spinner("🔄 Processing file..."):
            # Read the uploaded file
            input_df = pd.read_csv(uploaded_file)
            
        # Display file information
        st.success(f"✅ File uploaded successfully: {len(input_df)} rows, {len(input_df.columns)} columns")
        
        # Data validation
        st.subheader("🔍 Data Validation")
        is_valid, validation_issues = validate_input_data(input_df, feature_names)
        
        if validation_issues:
            for issue in validation_issues:
                if "🚫" in issue:
                    st.error(issue)
                elif "⚠️" in issue:
                    st.warning(issue)
                else:
                    st.info(issue)
        
        if not is_valid:
            st.error("❌ Data validation failed. Please fix the issues above.")
            st.stop()
        
        # Show data preview
        with st.expander("👁️ Data Preview"):
            st.dataframe(input_df.head())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(input_df))
            with col2:
                st.metric("Columns", len(input_df.columns))
            with col3:
                missing_vals = input_df.isnull().sum().sum()
                st.metric("Missing Values", missing_vals)
        
        # Data preprocessing
        st.subheader("⚙️ Data Preprocessing")
        with st.spinner("🔄 Preprocessing data..."):
            input_encoded, preprocessing_success, preprocessing_messages = preprocess_data(input_df, feature_names)
        
        # Display preprocessing results
        for message in preprocessing_messages:
            if "✅" in message:
                st.success(message)
            elif "⚠️" in message:
                st.warning(message)
            else:
                st.info(message)
        
        if not preprocessing_success or input_encoded is None:
            st.error("❌ Data preprocessing failed.")
            st.stop()
        
        # Make predictions
        st.subheader("🎯 Prediction Results")
        with st.spinner("🔄 Making predictions..."):
            try:
                preds = model.predict(input_encoded)
                
                # Add predictions to original dataframe
                result_df = input_df.copy()
                result_df['Predicted_Severity'] = preds
                
                # Display predictions summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Severity", f"{np.mean(preds):.3f}")
                with col2:
                    st.metric("Max Severity", f"{np.max(preds):.3f}")
                with col3:
                    st.metric("Min Severity", f"{np.min(preds):.3f}")
                with col4:
                    high_risk_count = np.sum(preds > np.percentile(preds, 75))
                    st.metric("High Risk Cases", high_risk_count)
                
                # Display results table
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    height=300
                )
                
                # Download button for results
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv_buffer.getvalue(),
                    file_name=f"threat_predictions_{int(time.time())}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                logger.error(f"Prediction error: {str(e)}")
                st.stop()
        
        # SHAP Explanations
        st.subheader("🔍 Model Explanations (SHAP)")
        
        if not SHAP_AVAILABLE:
            st.warning("⚠️ SHAP is not available. Explainability features are disabled.")
            st.info("To enable SHAP explanations, please install SHAP and its dependencies (numba, llvmlite).")
        else:
            # Determine number of samples for SHAP
            n_samples = min(len(input_encoded), max_shap_samples)
            
            if n_samples > 0:
                st.info(f"Computing SHAP values for {n_samples} samples...")
                
                try:
                    with st.spinner("🔄 Computing SHAP explanations..."):
                        # Use subset for SHAP to improve performance
                        shap_subset = input_encoded.iloc[:n_samples]
                        
                        explainer = shap.Explainer(model)
                        shap_values = explainer(shap_subset)
                    
                    # Display SHAP plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🐝 SHAP Beeswarm Plot**")
                        fig_beeswarm = plt.figure(figsize=(10, 6))
                        shap.plots.beeswarm(shap_values, show=False)
                        st.pyplot(fig_beeswarm, clear_figure=True)
                    
                    with col2:
                        st.markdown("**📊 SHAP Summary Plot**")
                        fig_summary = plt.figure(figsize=(10, 6))
                        shap.plots.bar(shap_values, show=False)
                        st.pyplot(fig_summary, clear_figure=True)
                    
                    # Feature importance table
                    with st.expander("📈 Feature Importance Details"):
                        feature_importance = np.abs(shap_values.values).mean(axis=0)
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Mean_SHAP_Value': feature_importance
                        }).sort_values('Mean_SHAP_Value', ascending=False)
                        
                        st.dataframe(importance_df.head(20), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ SHAP computation error: {str(e)}")
                    logger.error(f"SHAP error: {str(e)}")
                    st.info("SHAP explanations could not be computed. This may happen with certain model types or data configurations.")
            
            else:
                st.warning("⚠️ No data available for SHAP analysis.")
            
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}")

else:
    # No file uploaded - show instructions
    st.info("📁 Upload a CSV file to begin threat severity prediction.")
    
    # Show example of expected data format
    with st.expander("📋 Expected Data Format"):
        st.markdown("""
        **Your CSV file should contain network traffic features such as:**
        - Numerical features: duration, packet counts, byte counts, etc.
        - Categorical features: protocol types, service types, etc.
        
        **Requirements:**
        - File size: Maximum 50MB
        - Rows: Maximum 100,000 rows
        - Format: Valid CSV with headers
        """)
        
        if feature_names:
            st.markdown("**Expected features (sample):**")
            st.code(", ".join(feature_names[:10]) + ("..." if len(feature_names) > 10 else ""))
