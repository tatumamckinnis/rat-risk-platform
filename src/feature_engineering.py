"""
Feature engineering module for NYC Rat Risk Intelligence Platform.

This module creates features for time-series forecasting models,
including lag features, rolling statistics, and seasonal encodings.
"""

import logging
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from . import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for rat complaint forecasting.
    
    Creates lag features, rolling statistics, seasonal encodings,
    and handles categorical variables.
    """
    
    def __init__(
        self,
        lag_periods: List[int] = None,
        rolling_windows: List[int] = None,
    ):
        """
        Initialize the feature engineer.
        
        Args:
            lag_periods: List of lag periods (in months)
            rolling_windows: List of rolling window sizes (in months)
        """
        self.lag_periods = lag_periods or config.LAG_PERIODS
        self.rolling_windows = rolling_windows or config.ROLLING_WINDOWS
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        
        # Store feature names
        self.numerical_features = []
        self.categorical_features = []
        self.all_features = []
        
    def _create_lag_features(
        self, 
        df: pd.DataFrame,
        target_col: str = "complaint_count",
        group_col: str = "zip_code",
    ) -> pd.DataFrame:
        """Create lag features for the target variable."""
        df = df.copy()
        
        for lag in self.lag_periods:
            col_name = f"lag_{lag}"
            df[col_name] = df.groupby(group_col)[target_col].shift(lag)
            
        return df
    
    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = "complaint_count",
        group_col: str = "zip_code",
    ) -> pd.DataFrame:
        """Create rolling statistics features."""
        df = df.copy()
        
        for window in self.rolling_windows:
            # Rolling mean
            mean_col = f"rolling_mean_{window}"
            df[mean_col] = (
                df.groupby(group_col)[target_col]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
            
            # Rolling std
            std_col = f"rolling_std_{window}"
            df[std_col] = (
                df.groupby(group_col)[target_col]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std())
            )
            
            # Rolling max
            max_col = f"rolling_max_{window}"
            df[max_col] = (
                df.groupby(group_col)[target_col]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).max())
            )
            
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df = df.copy()
        
        # Extract from date
        if "date" in df.columns:
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["quarter"] = df["date"].dt.quarter
            
            # Cyclical encoding for month
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            
            # Season
            df["season"] = df["month"].map({
                12: "winter", 1: "winter", 2: "winter",
                3: "spring", 4: "spring", 5: "spring",
                6: "summer", 7: "summer", 8: "summer",
                9: "fall", 10: "fall", 11: "fall",
            })
            
            # Is summer (peak rat season)
            df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
            
        return df
    
    def _create_trend_features(
        self,
        df: pd.DataFrame,
        target_col: str = "complaint_count",
        group_col: str = "zip_code",
    ) -> pd.DataFrame:
        """Create trend-based features."""
        df = df.copy()
        
        # Month-over-month change
        df["mom_change"] = df.groupby(group_col)[target_col].diff()
        
        # Year-over-year change (12-month diff)
        df["yoy_change"] = df.groupby(group_col)[target_col].diff(12)
        
        # Percent change
        df["mom_pct_change"] = df.groupby(group_col)[target_col].pct_change()
        df["mom_pct_change"] = df["mom_pct_change"].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _encode_categoricals(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        categorical_cols = ["borough", "season"]
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    # Handle unseen categories
                    df[col] = df[col].fillna("Unknown")
                    df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[col] = df[col].fillna("Unknown")
                    # Handle unseen categories during transform
                    known_categories = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_categories else "Unknown"
                    )
                    if "Unknown" not in self.label_encoders[col].classes_:
                        # Add unknown class
                        self.label_encoders[col].classes_ = np.append(
                            self.label_encoders[col].classes_, "Unknown"
                        )
                    df[f"{col}_encoded"] = self.label_encoders[col].transform(df[col])
                    
        return df
    
    def _get_feature_names(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Get lists of numerical and categorical feature names."""
        
        numerical = []
        categorical = []
        
        # Lag features
        for lag in self.lag_periods:
            numerical.append(f"lag_{lag}")
            
        # Rolling features
        for window in self.rolling_windows:
            numerical.extend([
                f"rolling_mean_{window}",
                f"rolling_std_{window}",
                f"rolling_max_{window}",
            ])
            
        # Time features
        numerical.extend([
            "year", "month_sin", "month_cos", "is_summer",
        ])
        
        # Trend features
        numerical.extend([
            "mom_change", "yoy_change", "mom_pct_change",
        ])
        
        # External features
        external_numerical = [
            "restaurant_violations_nearby",
            "building_age_mean",
            "old_building_pct",
        ]
        for col in external_numerical:
            if col in df.columns:
                numerical.append(col)
                
        # Categorical (encoded)
        for col in ["borough", "season"]:
            if f"{col}_encoded" in df.columns:
                categorical.append(f"{col}_encoded")
                
        return numerical, categorical
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "complaint_count",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit the feature engineer and transform the data.
        
        Args:
            df: Input DataFrame with raw data
            target_col: Name of the target variable
            
        Returns:
            Tuple of (transformed DataFrame, target array)
        """
        logger.info("Fitting and transforming features...")
        
        df = df.copy()
        
        # Create all features
        df = self._create_lag_features(df, target_col)
        df = self._create_rolling_features(df, target_col)
        df = self._create_time_features(df)
        df = self._create_trend_features(df, target_col)
        df = self._encode_categoricals(df, fit=True)
        
        # Get feature names
        self.numerical_features, self.categorical_features = self._get_feature_names(df)
        self.all_features = self.numerical_features + self.categorical_features
        
        # Drop rows with NaN in features (due to lags)
        max_lag = max(self.lag_periods)
        df = df.iloc[max_lag:]
        
        # Handle remaining NaNs
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                
        # Fit scaler on numerical features
        numerical_data = df[self.numerical_features].values
        self.scaler.fit(numerical_data)
        
        # Transform
        df[self.numerical_features] = self.scaler.transform(numerical_data)
        
        self.is_fitted = True
        
        # Get feature matrix and target
        X = df[self.all_features].values
        y = df[target_col].values
        
        logger.info(f"Created {len(self.all_features)} features")
        
        return df, y
    
    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = "complaint_count",
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Transform new data using fitted feature engineer.
        
        Args:
            df: Input DataFrame
            target_col: Name of target variable (if available)
            
        Returns:
            Tuple of (transformed DataFrame, target array or None)
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
            
        logger.info("Transforming features...")
        
        df = df.copy()
        
        # Create all features
        df = self._create_lag_features(df, target_col)
        df = self._create_rolling_features(df, target_col)
        df = self._create_time_features(df)
        df = self._create_trend_features(df, target_col)
        df = self._encode_categoricals(df, fit=False)
        
        # Drop rows with NaN in lag features
        max_lag = max(self.lag_periods)
        df = df.iloc[max_lag:]
        
        # Handle remaining NaNs
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                
        # Transform numerical features
        df[self.numerical_features] = self.scaler.transform(
            df[self.numerical_features].values
        )
        
        # Get target if available
        y = df[target_col].values if target_col in df.columns else None
        
        return df, y
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from transformed DataFrame."""
        return df[self.all_features].values
    
    def save(self, path: str):
        """Save the fitted feature engineer."""
        joblib.dump({
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "all_features": self.all_features,
            "lag_periods": self.lag_periods,
            "rolling_windows": self.rolling_windows,
            "is_fitted": self.is_fitted,
        }, path)
        logger.info(f"Saved feature engineer to {path}")
        
    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        """Load a fitted feature engineer."""
        data = joblib.load(path)
        
        fe = cls(
            lag_periods=data["lag_periods"],
            rolling_windows=data["rolling_windows"],
        )
        fe.scaler = data["scaler"]
        fe.label_encoders = data["label_encoders"]
        fe.numerical_features = data["numerical_features"]
        fe.categorical_features = data["categorical_features"]
        fe.all_features = data["all_features"]
        fe.is_fitted = data["is_fitted"]
        
        logger.info(f"Loaded feature engineer from {path}")
        
        return fe


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array (n_samples,)
        sequence_length: Number of time steps in each sequence
        
    Returns:
        Tuple of (X_seq, y_seq) with shapes
        (n_sequences, sequence_length, n_features) and (n_sequences,)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
        
    return np.array(X_seq), np.array(y_seq)


def get_feature_importance_names(feature_engineer: FeatureEngineer) -> List[str]:
    """Get human-readable feature names for importance plots."""
    return feature_engineer.all_features


if __name__ == "__main__":
    # Test feature engineering
    from .data_preprocessing import load_master_dataset
    
    df = load_master_dataset()
    
    fe = FeatureEngineer()
    df_transformed, y = fe.fit_transform(df)
    
    print(f"Features: {fe.all_features}")
    print(f"Shape: {df_transformed[fe.all_features].shape}")
    
    # Save
    fe.save(config.MODELS_DIR / "feature_engineer.joblib")
