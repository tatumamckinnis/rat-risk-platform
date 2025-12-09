"""
Model utilities for NYC Rat Risk Intelligence Platform.

Provides convenient functions for loading and using trained models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config

logger = logging.getLogger(__name__)


def load_all_models() -> Dict:
    """
    Load all trained models.
    
    Returns:
        Dictionary containing all loaded models
    """
    models = {}
    
    # Load feature engineer
    fe_path = config.FORECASTING_MODELS_DIR / "feature_engineer.joblib"
    if fe_path.exists():
        from src.feature_engineering import FeatureEngineer
        models["feature_engineer"] = FeatureEngineer.load(str(fe_path))
        logger.info("Loaded feature engineer")
    
    # Load forecasting ensemble
    xgb_path = config.FORECASTING_MODELS_DIR / "xgboost.joblib"
    if xgb_path.exists():
        from src.forecasting_models import EnsembleForecaster
        ensemble = EnsembleForecaster()
        input_size = len(models["feature_engineer"].all_features) if "feature_engineer" in models else 20
        ensemble.load(str(config.FORECASTING_MODELS_DIR), input_size=input_size)
        models["ensemble"] = ensemble
        logger.info("Loaded forecasting ensemble")
    
    # Load image classifier
    classifier_path = config.CLASSIFIER_MODELS_DIR / "classifier.pt"
    if classifier_path.exists():
        from src.image_classifier import load_classifier
        models["classifier"] = load_classifier(str(classifier_path))
        logger.info("Loaded image classifier")
    
    # Load RAG system
    try:
        from src.rag_system import RAGSystem
        models["rag"] = RAGSystem()
        logger.info("Loaded RAG system")
    except Exception as e:
        logger.warning(f"Could not load RAG system: {e}")
    
    return models


def predict_complaints(
    zip_code: str,
    historical_data: pd.DataFrame,
    models: Dict,
    periods_ahead: int = 3,
) -> np.ndarray:
    """
    Predict future complaint counts for a ZIP code.
    
    Args:
        zip_code: ZIP code to predict for
        historical_data: Historical complaint data
        models: Dictionary of loaded models
        periods_ahead: Number of months to predict
        
    Returns:
        Array of predicted complaint counts
    """
    if "ensemble" not in models or "feature_engineer" not in models:
        logger.warning("Models not loaded")
        return np.array([0] * periods_ahead)
    
    fe = models["feature_engineer"]
    ensemble = models["ensemble"]
    
    # Filter data for ZIP code
    zip_data = historical_data[historical_data["zip_code"] == zip_code].copy()
    
    if len(zip_data) == 0:
        return np.array([0] * periods_ahead)
    
    # Transform features
    df_transformed, _ = fe.transform(zip_data)
    X = fe.get_feature_matrix(df_transformed)
    
    # Predict using XGBoost (simpler for point predictions)
    if "xgboost" in ensemble.models:
        predictions = ensemble.models["xgboost"].predict(X[-periods_ahead:])
    else:
        predictions = np.array([0] * periods_ahead)
    
    return np.maximum(predictions, 0)  # Ensure non-negative


def classify_image(image, models: Dict) -> Dict:
    """
    Classify an image for rat evidence.
    
    Args:
        image: PIL Image
        models: Dictionary of loaded models
        
    Returns:
        Classification results
    """
    if "classifier" not in models:
        return {
            "predicted_class": "unknown",
            "confidence": 0.0,
            "probabilities": {},
            "is_rat_evidence": False,
        }
    
    classifier = models["classifier"]
    pred_class, confidence, all_probs = classifier.predict(image)
    
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": all_probs,
        "is_rat_evidence": pred_class != "no_evidence",
    }


def get_rag_context(
    query: str,
    zip_code: str = None,
    models: Dict = None,
) -> Tuple[str, list]:
    """
    Get RAG context for a query.
    
    Args:
        query: User query
        zip_code: Optional location context
        models: Dictionary of loaded models
        
    Returns:
        Tuple of (context string, sources list)
    """
    if models is None or "rag" not in models:
        return "", []
    
    rag = models["rag"]
    return rag.answer_question(query, zip_code)


if __name__ == "__main__":
    # Test model loading
    logging.basicConfig(level=logging.INFO)
    
    models = load_all_models()
    print(f"Loaded models: {list(models.keys())}")
