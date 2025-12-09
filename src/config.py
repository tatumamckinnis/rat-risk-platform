"""
Configuration settings for the NYC Rat Risk Intelligence Platform.

This module contains all configurable parameters for data processing,
model training, and application behavior.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
FORECASTING_MODELS_DIR = MODELS_DIR / "forecasting"
CLASSIFIER_MODELS_DIR = MODELS_DIR / "classifier"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, 
                 FORECASTING_MODELS_DIR, CLASSIFIER_MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Ollama (local LLM - recommended, no API key needed)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # or "mistral", "phi3"

# Anthropic (optional)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

# OpenAI (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# NYC Open Data
NYC_OPEN_DATA_TOKEN = os.getenv("NYC_OPEN_DATA_TOKEN")
NYC_OPEN_DATA_DOMAIN = "data.cityofnewyork.us"

# Demo mode (use cached responses)
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# Lite mode (skip PyTorch models - use if getting crashes)
LITE_MODE = os.getenv("LITE_MODE", "false").lower() == "true"

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# NYC Open Data dataset identifiers
DATASETS = {
    "rat_sightings": "3q43-55fe",
    "restaurant_inspections": "43nn-pn8j",
    "pluto": None,  # Downloaded separately
}

# Maximum records to load (-1 for all)
MAX_RECORDS = int(os.getenv("MAX_RECORDS", -1))

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Time-based features
LAG_PERIODS = [1, 2, 3, 4, 12]  # Months
ROLLING_WINDOWS = [3, 6, 12]    # Months

# Spatial aggregation
SPATIAL_RESOLUTION = "zip_code"  # Options: zip_code, community_board, nta

# Feature columns
CATEGORICAL_FEATURES = ["borough", "month", "day_of_week", "season"]
NUMERICAL_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
    "rolling_std_3", "rolling_std_6",
    "restaurant_violations_nearby",
    "building_age_mean",
    "construction_permits_nearby",
    "year",
]

# =============================================================================
# MODEL CONFIGURATION - FORECASTING
# =============================================================================

# Train/validation/test split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# XGBoost hyperparameters (tuned via cross-validation)
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
}

# LSTM hyperparameters
LSTM_PARAMS = {
    "input_size": len(NUMERICAL_FEATURES) + 10,  # +10 for encoded categoricals
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "sequence_length": 12,  # Months of history
}

# Training parameters
TRAINING_PARAMS = {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.5,
}

# Prophet hyperparameters
PROPHET_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
}

# Ensemble weights (determined by validation performance)
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.45,
    "lstm": 0.35,
    "prophet": 0.20,
}

# =============================================================================
# MODEL CONFIGURATION - IMAGE CLASSIFICATION
# =============================================================================

# Model architecture
CLASSIFIER_ARCHITECTURE = "resnet18"  # Options: resnet18, resnet50, efficientnet_b0

# Class labels
IMAGE_CLASSES = [
    "rat",
    "droppings", 
    "burrow",
    "gnaw_marks",
    "no_evidence",
]

# Image preprocessing
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Data augmentation (training only)
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.1,
    "rotation_limit": 30,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "blur_limit": 3,
}

# Classifier training
CLASSIFIER_TRAINING = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "early_stopping_patience": 7,
    "freeze_backbone_epochs": 5,  # Fine-tune only head initially
}

# =============================================================================
# RAG CONFIGURATION
# =============================================================================

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ChromaDB settings
CHROMA_COLLECTION_NAME = "rat_complaints"
CHROMA_PERSIST_DIR = str(EMBEDDINGS_DIR / "chroma_db")

# Retrieval settings
RAG_TOP_K = 5
RAG_SIMILARITY_THRESHOLD = 0.3

# Chunking for documents
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# =============================================================================
# RISK SCORING CONFIGURATION
# =============================================================================

# Component weights for final risk score
RISK_WEIGHTS = {
    "historical_complaints": 0.30,
    "forecast_risk": 0.25,
    "restaurant_violations": 0.20,
    "building_age": 0.15,
    "nearby_construction": 0.10,
}

# Risk level thresholds (1-10 scale)
RISK_THRESHOLDS = {
    "low": 3,
    "medium": 6,
    "high": 8,
}

# =============================================================================
# REPORT GENERATION CONFIGURATION
# =============================================================================

# System prompt for LLM
REPORT_SYSTEM_PROMPT = """You are an expert public health analyst specializing in urban rodent control. 
You provide clear, actionable advice based on data analysis. Your tone is professional but accessible.
You always cite specific data when making claims and provide practical recommendations."""

# Report sections
REPORT_SECTIONS = [
    "risk_summary",
    "historical_analysis", 
    "forecast_outlook",
    "contributing_factors",
    "recommendations",
]

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

# Streamlit settings
APP_TITLE = "NYC Rat Risk Intelligence Platform"
APP_ICON = "🐀"
APP_LAYOUT = "wide"

# Map settings
DEFAULT_MAP_CENTER = [40.7128, -74.0060]  # NYC
DEFAULT_MAP_ZOOM = 11

# Geocoding
GEOCODING_USER_AGENT = "rat-risk-platform"

# Cache settings
CACHE_TTL = 3600  # seconds

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
