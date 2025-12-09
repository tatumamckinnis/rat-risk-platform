"""
Forecasting models for NYC Rat Risk Intelligence Platform.

This module implements multiple time-series forecasting models:
- XGBoost (gradient boosting)
- LSTM (deep learning)
- Prophet (Facebook's time-series library)
- Ensemble (weighted combination)
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

from . import config
from .feature_engineering import FeatureEngineer, create_sequences

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class BaseForecaster(ABC):
    """Abstract base class for forecasting models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the model."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load the model."""
        pass


class XGBoostForecaster(BaseForecaster):
    """XGBoost-based forecasting model."""
    
    def __init__(self, params: Dict = None):
        """
        Initialize XGBoost forecaster.
        
        Args:
            params: XGBoost hyperparameters
        """
        self.params = params or config.XGBOOST_PARAMS
        self.model = None
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ):
        """
        Fit the XGBoost model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        logger.info("Training XGBoost model...")
        
        self.model = xgb.XGBRegressor(**self.params)
        
        eval_set = [(X, y)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        
        logger.info("XGBoost training complete")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_
    
    def save(self, path: str):
        """Save the model."""
        joblib.dump(self.model, path)
        logger.info(f"Saved XGBoost model to {path}")
        
    def load(self, path: str):
        """Load the model."""
        self.model = joblib.load(path)
        logger.info(f"Loaded XGBoost model from {path}")


class LSTMModel(nn.Module):
    """LSTM neural network for time-series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * self.num_directions)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, 1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step
        last_out = lstm_out[:, -1, :]
        
        # Batch normalization
        last_out = self.batch_norm(last_out)
        
        # Fully connected
        output = self.fc(last_out)
        
        return output.squeeze(-1)


class LSTMForecaster(BaseForecaster):
    """LSTM-based forecasting model with training loop."""
    
    def __init__(
        self,
        input_size: int,
        sequence_length: int = 12,
        params: Dict = None,
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            input_size: Number of input features
            sequence_length: Length of input sequences
            params: Model hyperparameters
        """
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.params = params or config.LSTM_PARAMS
        
        # Override input size
        self.params["input_size"] = input_size
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.params.get("hidden_size", 64),
            num_layers=self.params.get("num_layers", 2),
            dropout=self.params.get("dropout", 0.3),
            bidirectional=self.params.get("bidirectional", True),
        ).to(DEVICE)
        
        self.training_history = {"train_loss": [], "val_loss": []}
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        early_stopping_patience: int = None,
        **kwargs,
    ):
        """
        Train the LSTM model.
        
        Args:
            X: Training sequences (n_samples, seq_len, features)
            y: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
        """
        logger.info("Training LSTM model...")
        
        # Get training parameters
        epochs = epochs or config.TRAINING_PARAMS["epochs"]
        batch_size = batch_size or config.TRAINING_PARAMS["batch_size"]
        learning_rate = learning_rate or config.TRAINING_PARAMS["learning_rate"]
        patience = early_stopping_patience or config.TRAINING_PARAMS["early_stopping_patience"]
        
        # Create data loaders
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        y_tensor = torch.FloatTensor(y).to(DEVICE)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
            y_val_tensor = torch.FloatTensor(y_val).to(DEVICE)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
            
        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.TRAINING_PARAMS["lr_scheduler_patience"],
            factor=config.TRAINING_PARAMS["lr_scheduler_factor"],
        )
        
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
                
            avg_train_loss = np.mean(train_losses)
            self.training_history["train_loss"].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_losses.append(loss.item())
                        
                avg_val_loss = np.mean(val_losses)
                self.training_history["val_loss"].append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                    
            if (epoch + 1) % 10 == 0:
                val_str = f", val_loss: {avg_val_loss:.4f}" if val_loader else ""
                logger.info(f"Epoch {epoch + 1}/{epochs}, train_loss: {avg_train_loss:.4f}{val_str}")
                
        # Load best model state
        if hasattr(self, "best_state"):
            self.model.load_state_dict(self.best_state)
            
        logger.info("LSTM training complete")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()
    
    def save(self, path: str):
        """Save the model."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "params": self.params,
            "training_history": self.training_history,
        }, path)
        logger.info(f"Saved LSTM model to {path}")
        
    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=DEVICE)
        
        self.input_size = checkpoint["input_size"]
        self.sequence_length = checkpoint["sequence_length"]
        self.params = checkpoint["params"]
        self.training_history = checkpoint["training_history"]
        
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.params.get("hidden_size", 64),
            num_layers=self.params.get("num_layers", 2),
            dropout=self.params.get("dropout", 0.3),
            bidirectional=self.params.get("bidirectional", True),
        ).to(DEVICE)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded LSTM model from {path}")


class ProphetForecaster(BaseForecaster):
    """Prophet-based forecasting model."""
    
    def __init__(self, params: Dict = None):
        """
        Initialize Prophet forecaster.
        
        Args:
            params: Prophet hyperparameters
        """
        self.params = params or config.PROPHET_PARAMS
        self.models = {}  # One model per location
        
    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "complaint_count",
        group_col: str = "zip_code",
        **kwargs,
    ):
        """
        Fit Prophet models (one per location).
        
        Args:
            df: DataFrame with date and target columns
            date_col: Name of date column
            target_col: Name of target column
            group_col: Column to group by (train separate models)
        """
        from prophet import Prophet
        
        logger.info("Training Prophet models...")
        
        # Train a model for each location
        locations = df[group_col].unique()
        
        for location in locations:
            location_df = df[df[group_col] == location][[date_col, target_col]].copy()
            location_df.columns = ["ds", "y"]
            
            model = Prophet(**self.params)
            model.fit(location_df)
            
            self.models[location] = model
            
        logger.info(f"Trained {len(self.models)} Prophet models")
        
    def predict(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        group_col: str = "zip_code",
        periods: int = 1,
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            df: DataFrame with dates and locations
            date_col: Name of date column
            group_col: Column identifying location
            periods: Number of periods to forecast ahead
            
        Returns:
            Array of predictions
        """
        predictions = []
        
        for _, row in df.iterrows():
            location = row[group_col]
            date = row[date_col]
            
            if location in self.models:
                future = pd.DataFrame({"ds": [date]})
                forecast = self.models[location].predict(future)
                pred = forecast["yhat"].values[0]
            else:
                pred = 0  # Default for unknown locations
                
            predictions.append(max(0, pred))  # Ensure non-negative
            
        return np.array(predictions)
    
    def save(self, path: str):
        """Save all Prophet models."""
        joblib.dump(self.models, path)
        logger.info(f"Saved Prophet models to {path}")
        
    def load(self, path: str):
        """Load Prophet models."""
        self.models = joblib.load(path)
        logger.info(f"Loaded {len(self.models)} Prophet models from {path}")


class EnsembleForecaster:
    """Ensemble of multiple forecasting models."""
    
    def __init__(
        self,
        models: Dict[str, BaseForecaster] = None,
        weights: Dict[str, float] = None,
    ):
        """
        Initialize ensemble forecaster.
        
        Args:
            models: Dictionary of model name -> model instance
            weights: Dictionary of model name -> weight
        """
        self.models = models or {}
        self.weights = weights or config.ENSEMBLE_WEIGHTS
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
    def add_model(self, name: str, model: BaseForecaster, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
        
        # Re-normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
    def predict(self, X: np.ndarray, X_df: pd.DataFrame = None) -> np.ndarray:
        """
        Generate weighted ensemble predictions.
        
        Args:
            X: Feature matrix for XGBoost/LSTM
            X_df: DataFrame for Prophet (needs date column)
            
        Returns:
            Weighted average predictions
        """
        predictions = {}
        
        for name, model in self.models.items():
            if name == "prophet" and X_df is not None:
                preds = model.predict(X_df)
            else:
                preds = model.predict(X)
                
            predictions[name] = preds
            
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for name, preds in predictions.items():
            if name in self.weights:
                ensemble_pred += self.weights[name] * preds
                
        return ensemble_pred
    
    def save(self, directory: str):
        """Save all models in the ensemble."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            if name == "xgboost":
                model.save(directory / "xgboost.joblib")
            elif name == "lstm":
                model.save(directory / "lstm.pt")
            elif name == "prophet":
                model.save(directory / "prophet.joblib")
                
        # Save weights
        joblib.dump(self.weights, directory / "ensemble_weights.joblib")
        
        logger.info(f"Saved ensemble to {directory}")
        
    def load(self, directory: str, input_size: int = None):
        """Load all models in the ensemble."""
        directory = Path(directory)
        
        # Load XGBoost
        xgb_path = directory / "xgboost.joblib"
        if xgb_path.exists():
            self.models["xgboost"] = XGBoostForecaster()
            self.models["xgboost"].load(str(xgb_path))
            
        # Load LSTM
        lstm_path = directory / "lstm.pt"
        if lstm_path.exists() and input_size is not None:
            self.models["lstm"] = LSTMForecaster(input_size=input_size)
            self.models["lstm"].load(str(lstm_path))
            
        # Load Prophet
        prophet_path = directory / "prophet.joblib"
        if prophet_path.exists():
            self.models["prophet"] = ProphetForecaster()
            self.models["prophet"].load(str(prophet_path))
            
        # Load weights
        weights_path = directory / "ensemble_weights.joblib"
        if weights_path.exists():
            self.weights = joblib.load(weights_path)
            
        logger.info(f"Loaded ensemble from {directory}")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric name -> value
    """
    # Ensure non-negative predictions
    y_pred = np.maximum(y_pred, 0)
    
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }
    
    # MAPE (handling zeros)
    mask = y_true != 0
    if mask.sum() > 0:
        metrics["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics["mape"] = np.nan
        
    return metrics


def compare_models(
    models: Dict[str, BaseForecaster],
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compare multiple models on test data.
    
    Args:
        models: Dictionary of model name -> model
        X_test: Test features
        y_test: Test targets
        X_test_df: DataFrame for Prophet
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for name, model in models.items():
        if name == "prophet" and X_test_df is not None:
            y_pred = model.predict(X_test_df)
        else:
            y_pred = model.predict(X_test)
            
        metrics = evaluate_model(y_test, y_pred)
        metrics["model"] = name
        results.append(metrics)
        
    return pd.DataFrame(results).set_index("model")


if __name__ == "__main__":
    # Test forecasting models
    logger.info("Testing forecasting models...")
    
    # Generate synthetic data
    np.random.seed(config.RANDOM_SEED)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.5
    
    # Split
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Test XGBoost
    xgb_model = XGBoostForecaster()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    print(f"XGBoost: {evaluate_model(y_test, xgb_pred)}")
    
    # Test LSTM
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length=12)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length=12)
    
    lstm_model = LSTMForecaster(input_size=n_features)
    lstm_model.fit(X_train_seq, y_train_seq, epochs=10)
    lstm_pred = lstm_model.predict(X_test_seq)
    print(f"LSTM: {evaluate_model(y_test_seq, lstm_pred)}")
