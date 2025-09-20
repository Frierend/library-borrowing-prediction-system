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
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
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

    /* Responsive tweak so the metric cards wrap on narrow screens */
    @media (max-width: 900px) {
        .metric-flex { flex-direction: column; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_components():
    """Load all model components with error handling"""
    try:
        model_path = 'models/'
        if not os.path.exists(model_path):
            st.error("❌ Models directory not found. Please create 'models/' folder and add your .pkl files.")
            return None, None, None, None, None
        
        model = joblib.load(os.path.join(model_path, 'wine_quality_model.pkl'))
        imputer = joblib.load(os.path.join(model_path, 'wine_imputer.pkl'))
        scaler = joblib.load(os.path.join(model_path, 'wine_scaler.pkl'))
        feature_names = joblib.load(os.path.join(model_path, 'feature_names.pkl'))
        metadata = joblib.load(os.path.join(model_path, 'model_metadata.pkl'))
        
        return model, imputer, scaler, feature_names, metadata
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None, None

def predict_wine_quality(features, model, imputer, scaler, feature_names, metadata):
    """Make wine quality prediction with error handling"""
    try:
        df_sample = pd.DataFrame([features], columns=feature_names)
        df_imputed = pd.DataFrame(imputer.transform(df_sample), columns=feature_names)
        
        if metadata.get('uses_scaling', False):
            df_processed = scaler.transform(df_imputed)
        else:
            df_processed = df_imputed
        
        prediction = model.predict(df_processed)[0]
        # If predict_proba returns two cols, assume [prob_bad, prob_good]
        proba = model.predict_proba(df_processed)[0]
        # try to get positive class probability (if model trained with 0/1)
        if len(proba) == 2:
            confidence = proba[1]
        else:
            # fallback: max probability
            confidence = np.max(proba)
        
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
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
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
        title={'text': "🔍 Key Quality Indicators", 'x': 0.5, 'font': {'size': 20}},
        xaxis_title="Feature Importance",
        yaxis_title="Wine Characteristics",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def get_risk_assessment(prediction, confidence):
    """Get risk level and recommendation along with emoji and hex colors"""
    # confidence is expected 0..1
    if prediction == 1:
        if confidence >= 0.8:
            return ("Low", "🟢", "#28a745", "APPROVE", "🟢", "#28a745")
        elif confidence >= 0.65:
            return ("Medium", "🟡", "#ffc107", "APPROVE", "🟢", "#28a745")
        else:
            return ("High", "🟠", "#fd7e14", "APPROVE", "🟢", "#28a745")
    else:
        if confidence <= 0.4:
            return ("High", "🔴", "#dc3545", "REJECT", "🔴", "#dc3545")
        else:
            return ("Medium", "🟡", "#ffc107", "REJECT", "🔴", "#dc3545")

def render_metric_card_html(title, emoji, value, hex_color, height_px=150):
    """Return HTML for a single metric card (used inside the flexbox layout)."""
    return f"""
    <div style="
        flex: 1;
        background-color: #111827;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        border: 2px solid {hex_color};
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: {height_px}px;
        min-width: 180px;
        ">
        <div style="font-size: 18px; font-weight: 600; color: #ffffff; margin-bottom: 8px;">{title}</div>
        <div style="font-size: 24px; font-weight: 700; color: {hex_color};">{emoji if emoji else ''}</div>
        <div style="font-size: 26px; font-weight: 700; color: #2c3e50; margin-top: 8px;">{value}</div>
    </div>
    """

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🍷 Wine Quality Predictor</h1>
        <p>Professional Quality Assessment Tool for Mr. Sanborn's Boutique Winery</p>
    </div>
    """, unsafe_allow_html=True)
    
    model, imputer, scaler, feature_names, metadata = load_model_components()
    if model is None:
        st.stop()
    
    # Sidebar
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
        1. Input wine measurements in the form.
        2. Click 'Analyze Wine Quality'.
        3. Review the prediction and confidence.
        4. Check feature importance to understand key factors.
        """)
    
    # Main layout
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.header("🧪 Wine Sample Analysis")
        default_values = st.session_state.get('sample_values', [7.4, 0.70, 0.00, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4])
        with st.form("wine_analysis_form"):
            st.markdown("### 📝 Enter Wine Characteristics")
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
                        key=f"input_{i}"
                    )
                    inputs.append(value)
            submitted = st.form_submit_button("🔍 Analyze Wine Quality", type="primary")
            if submitted:
                with st.spinner("🔬 Analyzing wine sample..."):
                    prediction, confidence, feature_importance = predict_wine_quality(inputs, model, imputer, scaler, feature_names, metadata)
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
            if prediction == 1:
                st.markdown(f"""
                <div class="good-quality">
                    <h2>✅ Premium Quality Wine</h2>
                    <p>This wine sample <strong>meets premium quality standards</strong>.</p>
                    <h3>Confidence: {confidence_percent}%</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="poor-quality">
                    <h2>❌ Below Premium Standards</h2>
                    <p>This wine sample <strong>does not meet</strong> premium standards.</p>
                    <h3>Confidence: {confidence_percent}%</h3>
                </div>
                """, unsafe_allow_html=True)

            # --- NEW: aligned metric cards with dynamic colors ---
            (risk_level, risk_emoji, risk_hex,
             recommendation, rec_emoji, rec_hex) = get_risk_assessment(prediction, confidence)

            # Build HTML for both cards and render inside a flex container
            risk_card_html = render_metric_card_html("Risk Level", risk_emoji, risk_level, risk_hex)
            rec_card_html = render_metric_card_html("Recommendation", rec_emoji, recommendation, rec_hex)

            st.markdown(
                f"""
                <div class="metric-flex" style="display:flex; gap: 20px; align-items: stretch; margin-top: 20px;">
                    {risk_card_html}
                    {rec_card_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
            # --- end metric cards ---

            if feature_importance:
                st.markdown('<div class="feature-importance-card">', unsafe_allow_html=True)
                fig = create_feature_importance_chart(feature_importance)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

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
                <p>Enter wine data and click 'Analyze Wine Quality' to see results</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        <p><strong>🍷 Wine Quality Predictor</strong></p>
        <p>Powered by Machine Learning | Built for Mr. Sanborn's Boutique Winery</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
