"""
NYC Rat Risk Intelligence Platform - Streamlit Web Application

This is the main user interface for the platform, providing:
- Address-based risk assessment
- Image upload for evidence detection
- Interactive Q&A about rat prevention
- Historical data visualization
- Forecasting predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import folium
from streamlit_folium import st_folium
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.risk_scorer import RiskScorer, LocationRiskAssessor
from src.report_generator import ReportGenerator, get_report_generator

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .risk-score-container {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-low { background-color: #d4edda; color: #155724; }
    .risk-moderate { background-color: #fff3cd; color: #856404; }
    .risk-high { background-color: #f8d7da; color: #721c24; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHING AND INITIALIZATION
# =============================================================================

@st.cache_resource
def load_models():
    """Load all ML models (cached)."""
    models = {}
    
    # Skip PyTorch models in lite mode (for M1/M2 Macs with issues)
    if config.LITE_MODE:
        st.info("Running in Lite Mode (PyTorch models disabled)")
        return models
    
    # Load forecasting ensemble
    try:
        from src.forecasting_models import EnsembleForecaster
        from src.feature_engineering import FeatureEngineer
        
        fe_path = config.FORECASTING_MODELS_DIR / "feature_engineer.joblib"
        if fe_path.exists():
            models["feature_engineer"] = FeatureEngineer.load(str(fe_path))
            
        ensemble = EnsembleForecaster()
        if (config.FORECASTING_MODELS_DIR / "xgboost.joblib").exists():
            input_size = len(models.get("feature_engineer", {}).all_features) if "feature_engineer" in models else 20
            ensemble.load(str(config.FORECASTING_MODELS_DIR), input_size=input_size)
            models["forecasting"] = ensemble
    except Exception as e:
        st.warning(f"Could not load forecasting models: {e}")
    
    # Load image classifier
    try:
        from src.image_classifier import load_classifier
        classifier_path = config.CLASSIFIER_MODELS_DIR / "classifier.pt"
        if classifier_path.exists():
            models["classifier"] = load_classifier(str(classifier_path))
    except Exception as e:
        st.warning(f"Could not load image classifier: {e}")
    
    # Load RAG system
    try:
        from src.rag_system import RAGSystem
        models["rag"] = RAGSystem()
    except Exception as e:
        st.warning(f"Could not load RAG system: {e}")
    
    return models


@st.cache_data
def load_data():
    """Load historical data (cached)."""
    try:
        df = pd.read_csv(
            config.PROCESSED_DATA_DIR / "master_dataset.csv",
            parse_dates=["date"],
        )
        return df
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return None


@st.cache_resource
def get_risk_scorer():
    """Get risk scorer instance."""
    df = load_data()
    return RiskScorer(historical_data=df)


def geocode_address(address: str) -> tuple:
    """Geocode an address to coordinates."""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        
        geolocator = Nominatim(user_agent=config.GEOCODING_USER_AGENT)
        
        # Add NYC context if not present
        if "new york" not in address.lower() and "ny" not in address.lower():
            address = f"{address}, New York, NY"
            
        location = geolocator.geocode(address, timeout=10)
        
        if location:
            return location.latitude, location.longitude, location.address
        return None, None, None
        
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None, None


# =============================================================================
# UI COMPONENTS
# =============================================================================

def display_risk_score(score: float, level: str, color: str):
    """Display the risk score prominently."""
    risk_class = "risk-low" if score < 4 else "risk-moderate" if score < 7 else "risk-high"
    
    st.markdown(f"""
    <div class="risk-score-container {risk_class}">
        <h1 style="font-size: 72px; margin: 0;">{score:.1f}</h1>
        <h3 style="margin: 0;">out of 10</h3>
        <p style="font-size: 24px; margin-top: 10px;"><strong>{level} Risk</strong></p>
    </div>
    """, unsafe_allow_html=True)


def display_factor_breakdown(factors: dict):
    """Display risk factor breakdown as a chart."""
    factor_names = {
        "historical_complaints": "Historical Activity",
        "forecast_risk": "Forecast Risk",
        "restaurant_violations": "Restaurant Violations",
        "building_age": "Building Age",
        "seasonal_factor": "Seasonal Factor",
    }
    
    data = []
    for key, value in factors.items():
        if key in factor_names and value > 0:
            data.append({
                "Factor": factor_names[key],
                "Score": value,
            })
    
    if data:
        df = pd.DataFrame(data)
        fig = px.bar(
            df,
            x="Score",
            y="Factor",
            orientation="h",
            color="Score",
            color_continuous_scale=["green", "yellow", "red"],
            range_color=[0, 10],
        )
        fig.update_layout(
            showlegend=False,
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def display_historical_chart(df: pd.DataFrame, zip_code: str = None):
    """Display historical complaint trends."""
    if df is None or df.empty:
        st.info("No historical data available")
        return
    
    # Filter by location if specified
    if zip_code:
        plot_df = df[df["zip_code"] == zip_code].copy()
        title = f"Complaint History - ZIP {zip_code}"
    else:
        plot_df = df.groupby("date")["complaint_count"].sum().reset_index()
        title = "City-wide Complaint Trends"
    
    if plot_df.empty:
        st.info("No data for selected location")
        return
    
    fig = px.line(
        plot_df,
        x="date",
        y="complaint_count",
        title=title,
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Complaints",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def display_map(lat: float, lon: float, zoom: int = 15):
    """Display an interactive map."""
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )
    
    # Add marker
    folium.Marker(
        [lat, lon],
        popup="Assessment Location",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)
    
    # Add circle for area
    folium.Circle(
        [lat, lon],
        radius=200,
        color="red",
        fill=True,
        opacity=0.2,
    ).add_to(m)
    
    st_folium(m, width=None, height=300)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=NYC+Rat+Risk", width=150)
        st.title("🐀 NYC Rat Risk")
        st.markdown("---")
        
        st.markdown("### About")
        st.markdown("""
        This platform uses machine learning to assess rat activity risk 
        across NYC by analyzing:
        - Historical 311 complaints
        - Restaurant inspection violations
        - Building characteristics
        - Seasonal patterns
        """)
        
        st.markdown("---")
        st.markdown("### Resources")
        st.markdown("- [NYC 311](https://portal.311.nyc.gov/)")
        st.markdown("- [NYC Health - Rats](https://www.nyc.gov/site/doh/health/health-topics/rats.page)")
        
        st.markdown("---")
        st.markdown("*Built for CS 372 Final Project*")
    
    # Main content
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.markdown("*Predict and prevent rat activity using machine learning*")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📍 Risk Assessment",
        "📷 Image Analysis",
        "❓ Ask Questions",
        "📊 Data Explorer",
    ])
    
    # Load models and data
    models = load_models()
    data = load_data()
    risk_scorer = get_risk_scorer()
    
    # =========================================================================
    # TAB 1: Risk Assessment
    # =========================================================================
    with tab1:
        st.header("Location Risk Assessment")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            address = st.text_input(
                "Enter an address",
                placeholder="123 Main St, Brooklyn, NY",
                help="Enter any NYC address to assess rat risk",
            )
            
            # Alternative: select by ZIP
            zip_codes = data["zip_code"].unique().tolist() if data is not None else []
            selected_zip = st.selectbox(
                "Or select a ZIP code",
                [""] + sorted(zip_codes),
                help="Select a ZIP code for area-wide assessment",
            )
        
        with col2:
            assess_button = st.button("🔍 Assess Risk", type="primary", use_container_width=True)
        
        # Process button click and store results in session state
        if assess_button and (address or selected_zip):
            with st.spinner("Analyzing location..."):
                # Geocode if address provided
                if address:
                    lat, lon, full_address = geocode_address(address)
                    if lat is None:
                        st.error("Could not find address. Please try again.")
                        st.stop()
                    zip_code = selected_zip if selected_zip else None
                else:
                    lat, lon = 40.7128, -74.0060
                    zip_code = selected_zip
                    full_address = f"ZIP Code {zip_code}"
                
                # Get historical stats
                if data is not None and zip_code:
                    location_data = data[data["zip_code"] == zip_code]
                    historical_complaints = location_data["complaint_count"].sum() if not location_data.empty else 0
                    restaurant_violations = location_data["restaurant_violations_nearby"].mean() if not location_data.empty else 0
                    building_age = location_data["building_age_mean"].mean() if "building_age_mean" in location_data.columns and not location_data.empty else 50
                else:
                    historical_complaints = 0
                    restaurant_violations = 0
                    building_age = 50
                
                # Calculate risk score
                score, factors = risk_scorer.calculate_risk_score(
                    historical_complaints=historical_complaints,
                    restaurant_violations=restaurant_violations,
                    building_age=building_age,
                    month=datetime.now().month,
                    return_factors=True,
                )
                
                level = risk_scorer.get_risk_level(score)
                color = risk_scorer.get_risk_color(score)
                
                # Store in session state
                st.session_state.risk_results = {
                    'score': score,
                    'level': level,
                    'color': color,
                    'factors': factors.to_dict() if factors is not None else {},
                    'lat': lat,
                    'lon': lon,
                    'full_address': full_address,
                    'zip_code': zip_code,
                    'historical_complaints': historical_complaints,
                }
        
        # Display results from session state (OUTSIDE the button block)
        if 'risk_results' in st.session_state:
            r = st.session_state.risk_results
            
            st.markdown("---")
            st.subheader(f"📍 {r['full_address']}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                display_risk_score(r['score'], r['level'], r['color'])
                
                st.markdown("### Risk Factors")
                if r['factors']:
                    display_factor_breakdown(r['factors'])
            
            with col2:
                if r['lat'] and r['lon']:
                    st.markdown("### Location")
                    display_map(r['lat'], r['lon'])
            
            # Historical chart
            st.markdown("### Historical Trends")
            display_historical_chart(data, r['zip_code'])
    
    # =========================================================================
    # TAB 2: Image Analysis
    # =========================================================================
    with tab2:
        st.header("Image Analysis")
        st.markdown("Upload a photo to detect signs of rat activity")
        
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            help="Upload a photo of suspected rat activity",
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                with st.spinner("Analyzing image..."):
                    # Classify image
                    if "classifier" in models:
                        try:
                            pred_class, confidence, all_probs = models["classifier"].predict(image)
                            
                            st.markdown("### Classification Result")
                            
                            # Show result
                            if pred_class == "no_evidence":
                                st.success(f"✅ No rat evidence detected ({confidence:.1%} confidence)")
                            else:
                                st.warning(f"⚠️ Detected: **{pred_class.replace('_', ' ').title()}** ({confidence:.1%} confidence)")
                            
                            # Show all probabilities
                            st.markdown("### All Classes")
                            prob_df = pd.DataFrame([
                                {"Class": k.replace("_", " ").title(), "Probability": v}
                                for k, v in all_probs.items()
                            ]).sort_values("Probability", ascending=False)
                            
                            fig = px.bar(
                                prob_df,
                                x="Probability",
                                y="Class",
                                orientation="h",
                            )
                            fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Classification error: {e}")
                    else:
                        st.info("Image classifier not loaded. Train models first.")
        
        # Example images
        st.markdown("---")
        st.markdown("### What to Look For")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**🐀 Rats**")
            st.caption("Live rats or deceased rats")
        
        with col2:
            st.markdown("**💩 Droppings**")
            st.caption("Dark pellets, 1/2-3/4 inch")
        
        with col3:
            st.markdown("**🕳️ Burrows**")
            st.caption("Holes 2-3 inches wide")
        
        with col4:
            st.markdown("**🦷 Gnaw Marks**")
            st.caption("On wood, plastic, wires")
    
    # =========================================================================
    # TAB 3: Ask Questions
    # =========================================================================
    with tab3:
        st.header("Ask About Rat Prevention")
        st.markdown("Get answers based on NYC Health Department guidelines")
        
        # Example questions
        st.markdown("**Try asking:**")
        example_questions = [
            "How do I prevent rats in my apartment?",
            "What are signs of rat infestation?",
            "Who is responsible for rat control - landlord or tenant?",
            "How do I report a rat sighting?",
        ]
        
        cols = st.columns(2)
        for i, q in enumerate(example_questions):
            if cols[i % 2].button(q, key=f"example_{i}"):
                st.session_state["question"] = q
        
        # Question input
        question = st.text_input(
            "Your question",
            value=st.session_state.get("question", ""),
            placeholder="Ask anything about rat prevention and control...",
        )
        
        if question:
            with st.spinner("Finding answer..."):
                try:
                    # Get context from RAG
                    if "rag" in models:
                        context, sources = models["rag"].answer_question(question)
                    else:
                        context = ""
                        sources = []
                    
                    # Generate answer
                    report_gen = get_report_generator()
                    answer = report_gen.answer_question(question, context)
                    
                    st.markdown("### Answer")
                    st.markdown(answer)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources[:3]:
                                st.markdown(f"- {source.get('text', '')[:200]}...")
                                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # =========================================================================
    # TAB 4: Data Explorer
    # =========================================================================
    with tab4:
        st.header("Data Explorer")
        
        if data is None:
            st.warning("No data loaded. Run the data download script first.")
            st.stop()
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        with col2:
            st.metric("ZIP Codes", f"{data['zip_code'].nunique():,}")
        with col3:
            st.metric("Date Range", f"{data['date'].min().year}-{data['date'].max().year}")
        with col4:
            st.metric("Total Complaints", f"{data['complaint_count'].sum():,}")
        
        st.markdown("---")
        
        # Borough comparison
        st.subheader("Complaints by Borough")
        
        if "borough" in data.columns:
            borough_data = data.groupby("borough")["complaint_count"].sum().reset_index()
            borough_data = borough_data.sort_values("complaint_count", ascending=True)
            
            fig = px.bar(
                borough_data,
                x="complaint_count",
                y="borough",
                orientation="h",
                color="complaint_count",
                color_continuous_scale="Reds",
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series
        st.subheader("Monthly Trends")
        
        monthly_data = data.groupby("date")["complaint_count"].sum().reset_index()
        
        fig = px.line(
            monthly_data,
            x="date",
            y="complaint_count",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Complaints",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal pattern
        st.subheader("Seasonal Pattern")
        
        data["month"] = pd.to_datetime(data["date"]).dt.month
        seasonal_data = data.groupby("month")["complaint_count"].mean().reset_index()
        
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        seasonal_data["month_name"] = seasonal_data["month"].apply(lambda x: month_names[x-1])
        
        fig = px.bar(
            seasonal_data,
            x="month_name",
            y="complaint_count",
            color="complaint_count",
            color_continuous_scale="YlOrRd",
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Complaints",
            height=300,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw data
        st.subheader("Raw Data")
        st.dataframe(data.head(100), use_container_width=True)


if __name__ == "__main__":
    main()