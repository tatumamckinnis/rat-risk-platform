"""
Model training script for NYC Rat Risk Intelligence Platform.

This script trains all ML components:
- Time-series forecasting models (XGBoost, LSTM, Prophet, Ensemble)
- Image classification model
- RAG index building

Usage:
    python src/train_models.py                    # Train all
    python src/train_models.py --component forecasting
    python src/train_models.py --component classifier
    python src/train_models.py --component rag
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.data_preprocessing import (
    load_master_dataset,
    create_master_dataset,
    get_train_val_test_split,
)
from src.feature_engineering import (
    FeatureEngineer,
    create_sequences,
)
from src.forecasting_models import (
    XGBoostForecaster,
    LSTMForecaster,
    ProphetForecaster,
    EnsembleForecaster,
    evaluate_model,
    compare_models,
)
from src.image_classifier import (
    RatEvidenceClassifier,
    ImageClassifierTrainer,
)
from src.rag_system import build_rag_index
from src.risk_scorer import RiskScorer

# Set up logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.PROJECT_ROOT / "training.log"),
    ],
)
logger = logging.getLogger(__name__)


def train_forecasting_models(
    df: pd.DataFrame = None,
    save_dir: Path = None,
) -> dict:
    """
    Train all forecasting models.
    
    Args:
        df: Master dataset (loads if not provided)
        save_dir: Directory to save models
        
    Returns:
        Dictionary of trained models and metrics
    """
    logger.info("=" * 50)
    logger.info("TRAINING FORECASTING MODELS")
    logger.info("=" * 50)
    
    save_dir = save_dir or config.FORECASTING_MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data if not provided
    if df is None:
        df = load_master_dataset()
        
    # Feature engineering
    logger.info("Performing feature engineering...")
    fe = FeatureEngineer()
    df_transformed, y = fe.fit_transform(df)
    
    # Save feature engineer
    fe.save(str(save_dir / "feature_engineer.joblib"))
    
    # Get feature matrix
    X = fe.get_feature_matrix(df_transformed)
    
    # Train/val/test split (time-based)
    n = len(X)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    results = {}
    
    # =========================================================================
    # Train XGBoost
    # =========================================================================
    logger.info("\n--- Training XGBoost ---")
    
    xgb_model = XGBoostForecaster()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    xgb_pred_val = xgb_model.predict(X_val)
    xgb_pred_test = xgb_model.predict(X_test)
    
    xgb_metrics = evaluate_model(y_test, xgb_pred_test)
    logger.info(f"XGBoost Test Metrics: {xgb_metrics}")
    
    # Save
    xgb_model.save(str(save_dir / "xgboost.joblib"))
    
    # Feature importance
    importance = xgb_model.get_feature_importance()
    importance_df = pd.DataFrame({
        "feature": fe.all_features,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(save_dir / "xgboost_feature_importance.csv", index=False)
    
    results["xgboost"] = {
        "model": xgb_model,
        "metrics": xgb_metrics,
        "predictions": xgb_pred_test,
    }
    
    # =========================================================================
    # Train LSTM
    # =========================================================================
    logger.info("\n--- Training LSTM ---")
    
    # Create sequences
    seq_len = config.LSTM_PARAMS.get("sequence_length", 12)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)
    
    logger.info(f"Sequence shapes - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}")
    
    lstm_model = LSTMForecaster(input_size=X.shape[1], sequence_length=seq_len)
    lstm_model.fit(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=config.TRAINING_PARAMS["epochs"],
    )
    
    # Evaluate
    lstm_pred_test = lstm_model.predict(X_test_seq)
    lstm_metrics = evaluate_model(y_test_seq, lstm_pred_test)
    logger.info(f"LSTM Test Metrics: {lstm_metrics}")
    
    # Save
    lstm_model.save(str(save_dir / "lstm.pt"))
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(lstm_model.training_history["train_loss"], label="Train")
    plt.plot(lstm_model.training_history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training History")
    plt.legend()
    plt.savefig(save_dir / "lstm_training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    results["lstm"] = {
        "model": lstm_model,
        "metrics": lstm_metrics,
        "predictions": lstm_pred_test,
    }
    
    # =========================================================================
    # Train Prophet
    # =========================================================================
    logger.info("\n--- Training Prophet ---")
    
    try:
        # Prophet needs DataFrame with ds and y columns
        df_prophet = df_transformed[["date", "zip_code", "complaint_count"]].copy()
        
        # Split for Prophet
        train_df = df_prophet.iloc[:train_end]
        test_df = df_prophet.iloc[val_end:]
        
        prophet_model = ProphetForecaster()
        prophet_model.fit(train_df, date_col="date", target_col="complaint_count")
        
        # Evaluate
        prophet_pred_test = prophet_model.predict(test_df)
        prophet_metrics = evaluate_model(
            test_df["complaint_count"].values,
            prophet_pred_test,
        )
        logger.info(f"Prophet Test Metrics: {prophet_metrics}")
        
        # Save
        prophet_model.save(str(save_dir / "prophet.joblib"))
        
        results["prophet"] = {
            "model": prophet_model,
            "metrics": prophet_metrics,
            "predictions": prophet_pred_test,
        }
    except Exception as e:
        logger.warning(f"Prophet training failed: {e}")
        results["prophet"] = None
    
    # =========================================================================
    # Create Ensemble
    # =========================================================================
    logger.info("\n--- Creating Ensemble ---")
    
    ensemble = EnsembleForecaster()
    ensemble.add_model("xgboost", xgb_model, config.ENSEMBLE_WEIGHTS.get("xgboost", 0.45))
    ensemble.add_model("lstm", lstm_model, config.ENSEMBLE_WEIGHTS.get("lstm", 0.35))
    
    if results.get("prophet"):
        ensemble.add_model("prophet", prophet_model, config.ENSEMBLE_WEIGHTS.get("prophet", 0.20))
    
    # Save ensemble
    ensemble.save(str(save_dir))
    
    # Evaluate ensemble (using XGBoost/LSTM aligned test set)
    # Note: For proper ensemble eval, we'd need aligned predictions
    logger.info("Ensemble created and saved")
    
    results["ensemble"] = ensemble
    
    # =========================================================================
    # Generate comparison report
    # =========================================================================
    logger.info("\n--- Model Comparison ---")
    
    comparison_data = []
    for name in ["xgboost", "lstm"]:
        if name in results and results[name]:
            metrics = results[name]["metrics"]
            comparison_data.append({
                "Model": name.upper(),
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "R²": metrics["r2"],
                "MAPE": metrics.get("mape", np.nan),
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(save_dir / "model_comparison.csv", index=False)
    logger.info(f"\n{comparison_df.to_string()}")
    
    # Save test predictions for analysis
    np.save(save_dir / "y_test.npy", y_test)
    np.save(save_dir / "xgboost_predictions.npy", xgb_pred_test)
    
    logger.info("\nForecasting model training complete!")
    
    return results


def train_image_classifier(
    data_dir: Path = None,
    save_dir: Path = None,
) -> ImageClassifierTrainer:
    """
    Train the image classification model.
    
    Note: This requires training images to be organized in folders by class.
    If no training data is available, creates a demo model.
    
    Args:
        data_dir: Directory containing training images
        save_dir: Directory to save model
        
    Returns:
        Trained ImageClassifierTrainer
    """
    logger.info("=" * 50)
    logger.info("TRAINING IMAGE CLASSIFIER")
    logger.info("=" * 50)
    
    save_dir = save_dir or config.CLASSIFIER_MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = data_dir or config.DATA_DIR / "images"
    
    # Check if training data exists
    if not data_dir.exists() or not any(data_dir.iterdir()):
        logger.warning("No training images found. Creating demo model...")
        
        # Create a model with pretrained weights (no fine-tuning)
        model = RatEvidenceClassifier(
            num_classes=len(config.IMAGE_CLASSES),
            architecture=config.CLASSIFIER_ARCHITECTURE,
            pretrained=True,
        )
        
        trainer = ImageClassifierTrainer(model, class_names=config.IMAGE_CLASSES)
        trainer.save(str(save_dir / "classifier.pt"))
        
        logger.info("Demo classifier saved (pretrained weights only)")
        return trainer
    
    # Load training data
    logger.info(f"Loading training images from {data_dir}...")
    
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    
    for class_idx, class_name in enumerate(config.IMAGE_CLASSES):
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue
            
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        # Split 80/20 train/val
        n_train = int(len(images) * 0.8)
        
        train_paths.extend(images[:n_train])
        train_labels.extend([class_idx] * n_train)
        
        val_paths.extend(images[n_train:])
        val_labels.extend([class_idx] * (len(images) - n_train))
        
        logger.info(f"  {class_name}: {len(images)} images")
    
    if not train_paths:
        logger.error("No training images found!")
        return None
    
    logger.info(f"Total: {len(train_paths)} train, {len(val_paths)} val")
    
    # Create and train model
    model = RatEvidenceClassifier(
        num_classes=len(config.IMAGE_CLASSES),
        architecture=config.CLASSIFIER_ARCHITECTURE,
        pretrained=True,
    )
    
    trainer = ImageClassifierTrainer(model, class_names=config.IMAGE_CLASSES)
    
    trainer.train(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
    )
    
    # Save
    trainer.save(str(save_dir / "classifier.pt"))
    
    # Plot training history
    if trainer.training_history["train_loss"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(trainer.training_history["train_loss"], label="Train")
        axes[0].plot(trainer.training_history["val_loss"], label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        
        axes[1].plot(trainer.training_history["train_acc"], label="Train")
        axes[1].plot(trainer.training_history["val_acc"], label="Val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Training Accuracy")
        axes[1].legend()
        
        plt.savefig(save_dir / "classifier_training_history.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    logger.info("Image classifier training complete!")
    
    return trainer


def build_rag_indexes():
    """Build RAG indexes for complaints and guidelines."""
    logger.info("=" * 50)
    logger.info("BUILDING RAG INDEXES")
    logger.info("=" * 50)
    
    # Load complaints data
    complaints_path = config.RAW_DATA_DIR / "rat_sightings.csv"
    
    if complaints_path.exists():
        complaints_df = pd.read_csv(complaints_path, low_memory=False)
        logger.info(f"Loaded {len(complaints_df)} complaints")
    else:
        complaints_df = None
        logger.warning("No complaints data found")
    
    # Build indexes
    rag = build_rag_index(complaints_df)
    
    logger.info("RAG index building complete!")
    
    return rag


def train_all():
    """Train all components."""
    logger.info("=" * 60)
    logger.info("NYC RAT RISK INTELLIGENCE PLATFORM - FULL TRAINING")
    logger.info("=" * 60)
    
    # Ensure data exists
    processed_path = config.PROCESSED_DATA_DIR / "master_dataset.csv"
    
    if not processed_path.exists():
        logger.info("Creating master dataset...")
        try:
            create_master_dataset()
        except Exception as e:
            logger.warning(f"Could not create master dataset: {e}")
            logger.info("Creating synthetic data for demonstration...")
            create_synthetic_data()
    
    # Load data
    try:
        df = load_master_dataset()
    except Exception as e:
        logger.warning(f"Could not load data: {e}")
        logger.info("Using synthetic data...")
        df = create_synthetic_data()
    
    # Train components
    results = {}
    
    # 1. Forecasting models
    try:
        results["forecasting"] = train_forecasting_models(df)
    except Exception as e:
        logger.error(f"Forecasting training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Image classifier
    try:
        results["classifier"] = train_image_classifier()
    except Exception as e:
        logger.error(f"Classifier training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. RAG indexes
    try:
        results["rag"] = build_rag_indexes()
    except Exception as e:
        logger.error(f"RAG building failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return results


def create_synthetic_data() -> pd.DataFrame:
    """Create synthetic data for demonstration/testing."""
    logger.info("Creating synthetic demonstration data...")
    
    np.random.seed(config.RANDOM_SEED)
    
    # Generate date range
    dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq="M")
    
    # NYC ZIP codes (sample)
    zip_codes = [
        "10001", "10002", "10003", "10004", "10005",
        "11201", "11211", "11215", "11217", "11238",
        "10451", "10452", "10453", "10454", "10455",
        "11101", "11102", "11103", "11104", "11105",
        "10301", "10302", "10303", "10304", "10305",
    ]
    
    boroughs = {
        "100": "Manhattan",
        "112": "Brooklyn",
        "104": "Bronx",
        "111": "Queens",
        "103": "Staten Island",
    }
    
    records = []
    
    for zip_code in zip_codes:
        # Base rate varies by area
        base_rate = np.random.uniform(5, 25)
        
        for date in dates:
            # Seasonal variation (higher in summer)
            month = date.month
            seasonal = 1 + 0.5 * np.sin(2 * np.pi * (month - 3) / 12)
            
            # Random variation
            noise = np.random.normal(0, 3)
            
            # Trend (slight increase over time)
            trend = 0.01 * (date.year - 2018)
            
            # Complaint count
            count = max(0, int(base_rate * seasonal * (1 + trend) + noise))
            
            # Borough
            borough = boroughs.get(zip_code[:3], "Manhattan")
            
            records.append({
                "date": date,
                "zip_code": zip_code,
                "borough": borough,
                "complaint_count": count,
                "restaurant_violations_nearby": np.random.poisson(3),
                "building_age_mean": np.random.uniform(30, 100),
                "old_building_pct": np.random.uniform(0.3, 0.8),
            })
    
    df = pd.DataFrame(records)
    
    # Save
    df.to_csv(config.PROCESSED_DATA_DIR / "master_dataset.csv", index=False)
    logger.info(f"Created synthetic dataset with {len(df)} records")
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train NYC Rat Risk Intelligence Platform models"
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["all", "forecasting", "classifier", "rag"],
        default="all",
        help="Component to train",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for demonstration",
    )
    
    args = parser.parse_args()
    
    # Create synthetic data if requested or if no real data exists
    if args.synthetic:
        create_synthetic_data()
    
    # Train requested component(s)
    if args.component == "all":
        train_all()
    elif args.component == "forecasting":
        df = load_master_dataset() if not args.synthetic else create_synthetic_data()
        train_forecasting_models(df)
    elif args.component == "classifier":
        train_image_classifier()
    elif args.component == "rag":
        build_rag_indexes()


if __name__ == "__main__":
    main()
