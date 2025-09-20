import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor - Mr. Sanborn's Winery",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 300;
    }
    
    .main-header p {
        font-size: 1.3rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .good-quality {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3);
        animation: slideIn 0.5s ease;
    }
    
    .poor-quality {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(220, 53, 69, 0.3);
        animation: slideIn 0.5s ease;
    }
    
    .waiting-result {
        background: linear-gradient(135deg, #6c757d 0%, #adb5bd 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        opacity: 0.8;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .sample-button {
        margin: 0.25rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 2px solid;
        background: transparent;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .feature-importance-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_components():
    """Load all model components with error handling"""
    try:
        model_path = 'models/'
        
        # Check if models directory exists
        if not os.path.exists(model_path):
            st.error("❌ Models directory not found. Please create 'models/' folder and add your .pkl files.")
            return None, None, None, None, None
        
        # Load all components
        model = joblib.load(os.path.join(model_path, 'wine_quality_model.pkl'))
        imputer = joblib.load(os.path.join(model_path, 'wine_imputer.pkl'))
        scaler = joblib.load(os.path.join(model_path, 'wine_scaler.pkl'))
        feature_names = joblib.load(os.path.join(model_path, 'feature_names.pkl'))
        metadata = joblib.load(os.path.join(model_path, 'model_metadata.pkl'))
        
        return model, imputer, scaler, feature_names, metadata
        
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {str(e)}")
        st.info("Please ensure all .pkl files are in the 'models/' directory:")
        st.code("""
        models/
        ├── wine_quality_model.pkl
        ├── wine_imputer.pkl
        ├── wine_scaler.pkl
        ├── feature_names.pkl
        └── model_metadata.pkl
        """)
        return None, None, None, None, None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None, None

def predict_wine_quality(features, model, imputer, scaler, feature_names, metadata):
    """Make wine quality prediction with error handling"""
    try:
        # Create DataFrame from input features
        df_sample = pd.DataFrame([features], columns=feature_names)
        
        # Apply imputation for missing values
        df_imputed = pd.DataFrame(
            imputer.transform(df_sample), 
            columns=feature_names
        )
        
        # Apply scaling if the model requires it
        if metadata.get('uses_scaling', False):
            df_processed = scaler.transform(df_imputed)
        else:
            df_processed = df_imputed
        
        # Make prediction
        prediction = model.predict(df_processed)[0]
        confidence = model.predict_proba(df_processed)[0][1]
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, feature in enumerate(feature_names):
                feature_importance[feature] = float(model.feature_importances_[i])
        
        return prediction, confidence, feature_importance
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None, None

def create_feature_importance_chart(feature_importance):
    """Create interactive feature importance chart"""
    if not feature_importance:
        return None
    
    # Sort features by importance and get top 8
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
    
    # Prepare data for plotting
    feature_names_clean = []
    importance_values = []
    
    name_mapping = {
        'fixed_acidity': 'Fixed Acidity',
        'volatile_acidity': 'Volatile Acidity',
        'citric_acid': 'Citric Acid',
        'residual_sugar': 'Residual Sugar',
        'chlorides': 'Chlorides',
        'free_sulfur_dioxide': 'Free SO₂',
        'total_sulfur_dioxide': 'Total SO₂',
        'density': 'Density',
        'ph': 'pH Level',
        'sulphates': 'Sulphates',
        'alcohol': 'Alcohol Content'
    }
    
    for feature, importance in sorted_features:
        clean_name = name_mapping.get(feature, feature.replace('_', ' ').title())
        feature_names_clean.append(clean_name)
        importance_values.append(importance)
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=feature_names_clean,
            x=importance_values,
            orientation='h',
            marker=dict(
                color=importance_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{val:.3f}' for val in importance_values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "🔍 Key Quality Indicators",
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title="Feature Importance",
        yaxis_title="Wine Characteristics",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def get_risk_assessment(prediction, confidence):
    """Get risk level and recommendation"""
    if prediction == 1:  # Good quality
        if confidence >= 0.8:
            risk_level = "Low"
            risk_color = "🟢"
        elif confidence >= 0.65:
            risk_level = "Medium"
            risk_color = "🟡"
        else:
            risk_level = "High"
            risk_color = "🟠"
        recommendation = "APPROVE"
        rec_color = "🟢"
    else:  # Poor quality
        if confidence <= 0.4:
            risk_level = "High"
            risk_color = "🔴"
        else:
            risk_level = "Medium"
            risk_color = "🟡"
        recommendation = "REJECT"
        rec_color = "🔴"
    
    return risk_level, risk_color, recommendation, rec_color

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍷 Wine Quality Predictor</h1>
        <p>Professional Quality Assessment Tool for Mr. Sanborn's Boutique Winery</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model components
    model, imputer, scaler, feature_names, metadata = load_model_components()
    
    if model is None:
        st.stop()
    
    # Sidebar for information and sample data
    with st.sidebar:
        st.header("📊 Sample Wine Data")
        st.markdown("Click to load sample data for testing:")
        
        sample_data = {
            "🍷 Premium Wine": {
                'values': [7.3, 0.45, 0.36, 5.4, 0.052, 16, 42, 0.9956, 3.32, 0.75, 11.8],
                'description': "High-quality wine sample (Expected: Good)",
                'color': '#28a745'
            },
            "🍷 Average Wine": {
                'values': [8.1, 0.52, 0.26, 2.4, 0.087, 15, 47, 0.9965, 3.31, 0.66, 10.4],
                'description': "Average wine sample (Expected: Average)",
                'color': '#ffc107'
            },
            "🍷 Poor Wine": {
                'values': [8.8, 0.89, 0.01, 2.1, 0.162, 6, 84, 0.9988, 3.58, 0.44, 9.1],
                'description': "Poor quality sample (Expected: Poor)",
                'color': '#dc3545'
            }
        }
        
        for sample_name, sample_info in sample_data.items():
            if st.button(sample_name, key=sample_name):
                st.session_state.sample_values = sample_info['values']
                st.success(f"✅ {sample_name} data loaded!")
        
        st.markdown("---")
        st.markdown("### 📈 Model Information")
        st.info(f"""
        **Model Type**: {metadata.get('model_name', 'ML Ensemble')}
        **Performance**: AUC ≥ 0.85 (Excellent)
        **Features**: {len(feature_names)} wine characteristics
        **Quality Threshold**: Rating ≥ 7
        """)
        
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        1. **Input wine measurements** in the form
        2. **Click 'Analyze Wine Quality'**
        3. **Review the prediction** and confidence
        4. **Check feature importance** to understand key factors
        """)
    
    # Main content area
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.header("🧪 Wine Sample Analysis")
        
        # Get default values from session state or use defaults
        if 'sample_values' in st.session_state:
            default_values = st.session_state.sample_values
        else:
            default_values = [7.4, 0.70, 0.00, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]
        
        # Input form
        with st.form("wine_analysis_form"):
            st.markdown("### 📝 Enter Wine Characteristics")
            
            # Create two columns for inputs
            input_col1, input_col2 = st.columns(2)
            
            inputs = []
            input_configs = [
                ("Fixed Acidity (g/L)", 0.1, "%.1f", 4.0, 20.0),
                ("Volatile Acidity (g/L)", 0.01, "%.2f", 0.0, 2.0),
                ("Citric Acid (g/L)", 0.01, "%.2f", 0.0, 1.0),
                ("Residual Sugar (g/L)", 0.1, "%.1f", 0.0, 20.0),
                ("Chlorides (g/L)", 0.001, "%.3f", 0.0, 1.0),
                ("Free SO₂ (mg/L)", 1.0, "%.0f", 0.0, 100.0),
                ("Total SO₂ (mg/L)", 1.0, "%.0f", 0.0, 300.0),
                ("Density (g/cm³)", 0.0001, "%.4f", 0.990, 1.010),
                ("pH Level", 0.01, "%.2f", 2.5, 4.5),
                ("Sulphates (g/L)", 0.01, "%.2f", 0.0, 2.0),
                ("Alcohol Content (%)", 0.1, "%.1f", 8.0, 16.0)
            ]
            
            for i, (label, step, format_str, min_val, max_val) in enumerate(input_configs):
                col = input_col1 if i % 2 == 0 else input_col2
                
                with col:
                    value = st.number_input(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=float(default_values[i]),
                        step=step,
                        format=format_str,
                        key=f"input_{i}",
                        help=f"Typical range: {min_val} - {max_val}"
                    )
                    inputs.append(value)
            
            # Analysis button
            st.markdown("### 🔬 Run Analysis")
            submitted = st.form_submit_button("🔍 Analyze Wine Quality", type="primary")
            
            if submitted:
                with st.spinner("🔬 Analyzing wine sample..."):
                    prediction, confidence, feature_importance = predict_wine_quality(
                        inputs, model, imputer, scaler, feature_names, metadata
                    )
                    
                    if prediction is not None:
                        st.session_state.analysis_results = {
                            'prediction': prediction,
                            'confidence': confidence,
                            'feature_importance': feature_importance,
                            'timestamp': datetime.now(),
                            'input_values': inputs
                        }
                        st.success("✅ Analysis completed!")
                    else:
                        st.error("❌ Analysis failed. Please check your inputs and try again.")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            
            prediction = results['prediction']
            confidence = results['confidence']
            feature_importance = results['feature_importance']
            timestamp = results['timestamp']
            
            confidence_percent = round(confidence * 100)
            
            # Main result display
            if prediction == 1:
                st.markdown(f"""
                <div class="good-quality">
                    <h2>✅ Premium Quality Wine</h2>
                    <p style="font-size: 1.2rem; margin: 1rem 0;">
                        This wine sample <strong>meets premium quality standards</strong> 
                        and is <strong>approved</strong> for boutique production.
                    </p>
                    <h3 style="font-size: 2rem; margin-top: 1rem;">
                        Confidence: {confidence_percent}%
                    </h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="poor-quality">
                    <h2>❌ Below Premium Standards</h2>
                    <p style="font-size: 1.2rem; margin: 1rem 0;">
                        This wine sample <strong>does not meet</strong> premium quality standards 
                        and requires <strong>improvement</strong> before approval.
                    </p>
                    <h3 style="font-size: 2rem; margin-top: 1rem;">
                        Confidence: {confidence_percent}%
                    </h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk assessment and recommendations
            risk_level, risk_color, recommendation, rec_color = get_risk_assessment(prediction, confidence)
            
            # Display metrics
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{risk_color} Risk Level</h3>
                    <h2 style="color: #2c3e50; margin: 0;">{risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{rec_color} Recommendation</h3>
                    <h2 style="color: #2c3e50; margin: 0;">{recommendation}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature importance visualization
            if feature_importance:
                st.markdown('<div class="feature-importance-card">', unsafe_allow_html=True)
                fig = create_feature_importance_chart(feature_importance)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis details
            st.markdown("---")
            st.markdown("### 📋 Analysis Details")
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.metric("Quality Score", f"{confidence:.3f}")
                st.metric("Model Accuracy", "85%+")
            
            with detail_col2:
                st.metric("Analysis Time", f"{timestamp.strftime('%H:%M:%S')}")
                st.metric("Date", f"{timestamp.strftime('%Y-%m-%d')}")
            
        else:
            st.markdown("""
            <div class="waiting-result">
                <h3>⏳ Waiting for Analysis</h3>
                <p>Enter wine sample data and click 'Analyze Wine Quality' to see results</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 Expected Output")
            st.info("""
            **Quality Assessment**: Good Quality ✅ or Below Standards ❌
            
            **Confidence Score**: Reliability percentage (0-100%)
            
            **Risk Level**: Low 🟢 | Medium 🟡 | High 🔴
            
            **Recommendation**: APPROVE or REJECT
            
            **Feature Analysis**: Key factors affecting quality
            """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>🍷 Wine Quality Predictor</strong></p>
        <p>Powered by Machine Learning | Built for Mr. Sanborn's Boutique Winery</p>
        <p><em>Professional quality assessment tool for consistent wine standards</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()