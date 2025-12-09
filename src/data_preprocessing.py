"""
Data preprocessing module for NYC Rat Risk Intelligence Platform.

This module handles cleaning, transforming, and merging datasets
for use in model training and inference.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from . import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def clean_rat_sightings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess rat sighting data.
    
    Args:
        df: Raw rat sightings DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning rat sightings data ({len(df)} records)...")
    
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Parse dates
    date_columns = ["created_date", "closed_date", "due_date"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Extract date components
    if "created_date" in df.columns:
        df["year"] = df["created_date"].dt.year
        df["month"] = df["created_date"].dt.month
        df["day"] = df["created_date"].dt.day
        df["day_of_week"] = df["created_date"].dt.dayofweek
        df["week_of_year"] = df["created_date"].dt.isocalendar().week
        df["quarter"] = df["created_date"].dt.quarter
        
        # Season
        df["season"] = df["month"].map({
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall",
        })
    
    # Clean borough names
    if "borough" in df.columns:
        borough_mapping = {
            "MANHATTAN": "Manhattan",
            "BROOKLYN": "Brooklyn",
            "BRONX": "Bronx",
            "QUEENS": "Queens",
            "STATEN ISLAND": "Staten Island",
        }
        df["borough"] = df["borough"].str.upper().map(borough_mapping)
        df = df.dropna(subset=["borough"])
    
    # Extract coordinates
    for coord_col in ["latitude", "longitude"]:
        if coord_col in df.columns:
            df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce")
    
    # Clean zip codes
    if "incident_zip" in df.columns:
        df["zip_code"] = df["incident_zip"].astype(str).str[:5]
        df = df[df["zip_code"].str.match(r"^\d{5}$", na=False)]
    
    # Calculate resolution time (if closed)
    if "created_date" in df.columns and "closed_date" in df.columns:
        df["resolution_days"] = (
            df["closed_date"] - df["created_date"]
        ).dt.total_seconds() / (24 * 3600)
        df["resolution_days"] = df["resolution_days"].clip(lower=0)
    
    # Filter to valid NYC coordinates
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df[
            (df["latitude"].between(40.4, 41.0)) &
            (df["longitude"].between(-74.3, -73.6))
        ]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["unique_key"] if "unique_key" in df.columns else None)
    
    logger.info(f"Cleaned data has {len(df)} records")
    
    return df


def clean_restaurant_inspections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess restaurant inspection data.
    
    Args:
        df: Raw restaurant inspection DataFrame
        
    Returns:
        Cleaned DataFrame with rodent violations
    """
    logger.info(f"Cleaning restaurant inspection data ({len(df)} records)...")
    
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Parse dates
    if "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
        df["inspection_year"] = df["inspection_date"].dt.year
        df["inspection_month"] = df["inspection_date"].dt.month
    
    # Filter for rodent-related violations
    rodent_keywords = ["rodent", "rat", "mice", "mouse", "vermin"]
    if "violation_description" in df.columns:
        mask = df["violation_description"].str.lower().str.contains(
            "|".join(rodent_keywords), 
            na=False
        )
        df = df[mask]
    
    # Clean zip codes
    if "zipcode" in df.columns:
        df["zip_code"] = df["zipcode"].astype(str).str[:5]
    elif "zip" in df.columns:
        df["zip_code"] = df["zip"].astype(str).str[:5]
    
    # Clean borough
    if "boro" in df.columns:
        borough_mapping = {
            "1": "Manhattan",
            "2": "Bronx", 
            "3": "Brooklyn",
            "4": "Queens",
            "5": "Staten Island",
            "Manhattan": "Manhattan",
            "Bronx": "Bronx",
            "Brooklyn": "Brooklyn",
            "Queens": "Queens",
            "Staten Island": "Staten Island",
        }
        df["borough"] = df["boro"].astype(str).map(borough_mapping)
    
    # Extract coordinates if available
    for coord_col in ["latitude", "longitude"]:
        if coord_col in df.columns:
            df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce")
    
    logger.info(f"Filtered to {len(df)} rodent violation records")
    
    return df


def clean_pluto_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess PLUTO building data.
    
    Args:
        df: Raw PLUTO DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning PLUTO data ({len(df)} records)...")
    
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Extract year built
    year_col = None
    for col in ["yearbuilt", "year_built", "cnstrct_yr", "yearconstructed"]:
        if col in df.columns:
            year_col = col
            break
    
    if year_col:
        df["year_built"] = pd.to_numeric(df[year_col], errors="coerce")
        # Filter to reasonable years
        current_year = datetime.now().year
        df = df[df["year_built"].between(1800, current_year) | df["year_built"].isna()]
        # Calculate building age
        df["building_age"] = current_year - df["year_built"]
    
    # Clean zip codes
    for col in ["zipcode", "zip_code", "zip"]:
        if col in df.columns:
            df["zip_code"] = df[col].astype(str).str[:5]
            break
    
    logger.info(f"Cleaned PLUTO data has {len(df)} records")
    
    return df


def aggregate_by_location_time(
    df: pd.DataFrame,
    time_column: str = "created_date",
    location_column: str = "zip_code",
    freq: str = "M",
) -> pd.DataFrame:
    """
    Aggregate complaint data by location and time period.
    
    Args:
        df: Cleaned complaints DataFrame
        time_column: Column containing timestamps
        location_column: Column for spatial aggregation
        freq: Pandas frequency string (M=monthly, W=weekly)
        
    Returns:
        Aggregated DataFrame with complaint counts
    """
    logger.info(f"Aggregating by {location_column} and {freq} frequency...")
    
    df = df.copy()
    
    # Create period column
    df["period"] = df[time_column].dt.to_period(freq)
    
    # Aggregate
    agg_df = df.groupby([location_column, "period"]).agg(
        complaint_count=("unique_key" if "unique_key" in df.columns else time_column, "count"),
        avg_resolution_days=("resolution_days", "mean") if "resolution_days" in df.columns else (time_column, "count"),
    ).reset_index()
    
    # Convert period back to datetime
    agg_df["date"] = agg_df["period"].dt.to_timestamp()
    
    # Fill missing periods with zeros
    all_locations = df[location_column].unique()
    all_periods = pd.period_range(
        df["period"].min(),
        df["period"].max(),
        freq=freq
    )
    
    full_index = pd.MultiIndex.from_product(
        [all_locations, all_periods],
        names=[location_column, "period"]
    )
    
    agg_df = agg_df.set_index([location_column, "period"]).reindex(full_index, fill_value=0).reset_index()
    agg_df["date"] = agg_df["period"].dt.to_timestamp()
    
    logger.info(f"Created {len(agg_df)} location-time records")
    
    return agg_df


def merge_restaurant_violations(
    rat_df: pd.DataFrame,
    restaurant_df: pd.DataFrame,
    on: str = "zip_code",
    time_window_months: int = 6,
) -> pd.DataFrame:
    """
    Merge restaurant rodent violation counts with rat sighting data.
    
    Args:
        rat_df: Aggregated rat sighting DataFrame
        restaurant_df: Cleaned restaurant inspection DataFrame
        on: Column to join on
        time_window_months: Lookback window for counting violations
        
    Returns:
        Merged DataFrame with violation counts
    """
    logger.info("Merging restaurant violation data...")
    
    # Aggregate restaurant violations by location and month
    if "inspection_date" in restaurant_df.columns:
        restaurant_df["period"] = restaurant_df["inspection_date"].dt.to_period("M")
        
        restaurant_agg = restaurant_df.groupby([on, "period"]).size().reset_index(name="violation_count")
        restaurant_agg["date"] = restaurant_agg["period"].dt.to_timestamp()
        
        # Calculate rolling sum of violations
        restaurant_agg = restaurant_agg.sort_values(["zip_code", "date"])
        restaurant_agg["restaurant_violations_nearby"] = (
            restaurant_agg.groupby(on)["violation_count"]
            .rolling(window=time_window_months, min_periods=1)
            .sum()
            .reset_index(drop=True)
        )
        
        # Merge with rat data
        rat_df = rat_df.merge(
            restaurant_agg[[on, "date", "restaurant_violations_nearby"]],
            on=[on, "date"],
            how="left"
        )
        rat_df["restaurant_violations_nearby"] = rat_df["restaurant_violations_nearby"].fillna(0)
    else:
        rat_df["restaurant_violations_nearby"] = 0
    
    logger.info("Restaurant violations merged")
    
    return rat_df


def merge_building_data(
    rat_df: pd.DataFrame,
    pluto_df: pd.DataFrame,
    on: str = "zip_code",
) -> pd.DataFrame:
    """
    Merge building age statistics with rat sighting data.
    
    Args:
        rat_df: Aggregated rat sighting DataFrame
        pluto_df: Cleaned PLUTO DataFrame
        on: Column to join on
        
    Returns:
        Merged DataFrame with building statistics
    """
    logger.info("Merging building data...")
    
    if "building_age" in pluto_df.columns:
        # Aggregate building stats by location
        building_stats = pluto_df.groupby(on).agg(
            building_age_mean=("building_age", "mean"),
            building_age_median=("building_age", "median"),
            building_count=("building_age", "count"),
            old_building_pct=("building_age", lambda x: (x > 50).mean()),
        ).reset_index()
        
        # Merge with rat data
        rat_df = rat_df.merge(building_stats, on=on, how="left")
        
        # Fill missing values with median
        for col in ["building_age_mean", "building_age_median", "old_building_pct"]:
            if col in rat_df.columns:
                rat_df[col] = rat_df[col].fillna(rat_df[col].median())
    else:
        rat_df["building_age_mean"] = 50  # Default assumption
        rat_df["old_building_pct"] = 0.5
    
    logger.info("Building data merged")
    
    return rat_df


def create_master_dataset(
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create the master dataset by loading, cleaning, and merging all data sources.
    
    Args:
        save_path: Path to save the processed dataset
        
    Returns:
        Master DataFrame ready for feature engineering
    """
    logger.info("Creating master dataset...")
    
    # Load raw data
    rat_df = pd.read_csv(config.RAW_DATA_DIR / "rat_sightings.csv", low_memory=False)
    
    # Try to load restaurant data
    restaurant_path = config.RAW_DATA_DIR / "restaurant_inspections.csv"
    if restaurant_path.exists():
        restaurant_df = pd.read_csv(restaurant_path, low_memory=False)
    else:
        restaurant_df = pd.DataFrame()
        logger.warning("Restaurant inspection data not found")
    
    # Try to load PLUTO data
    pluto_path = config.RAW_DATA_DIR / "pluto_simplified.csv"
    if pluto_path.exists():
        pluto_df = pd.read_csv(pluto_path, low_memory=False)
    else:
        pluto_df = pd.DataFrame()
        logger.warning("PLUTO data not found")
    
    # Clean each dataset
    rat_df = clean_rat_sightings(rat_df)
    if not restaurant_df.empty:
        restaurant_df = clean_restaurant_inspections(restaurant_df)
    if not pluto_df.empty:
        pluto_df = clean_pluto_data(pluto_df)
    
    # Aggregate rat sightings by location and time
    master_df = aggregate_by_location_time(
        rat_df,
        time_column="created_date",
        location_column="zip_code",
        freq="M"
    )
    
    # Add borough information
    zip_borough = rat_df.groupby("zip_code")["borough"].first().reset_index()
    master_df = master_df.merge(zip_borough, on="zip_code", how="left")
    
    # Merge restaurant violations
    if not restaurant_df.empty:
        master_df = merge_restaurant_violations(master_df, restaurant_df)
    else:
        master_df["restaurant_violations_nearby"] = 0
    
    # Merge building data
    if not pluto_df.empty:
        master_df = merge_building_data(master_df, pluto_df)
    else:
        master_df["building_age_mean"] = 50
        master_df["old_building_pct"] = 0.5
    
    # Sort by location and time
    master_df = master_df.sort_values(["zip_code", "date"]).reset_index(drop=True)
    
    # Save processed dataset
    save_path = save_path or config.PROCESSED_DATA_DIR / "master_dataset.csv"
    master_df.to_csv(save_path, index=False)
    logger.info(f"Master dataset saved to {save_path} ({len(master_df)} records)")
    
    return master_df


def load_master_dataset() -> pd.DataFrame:
    """
    Load the processed master dataset.
    
    Returns:
        Master DataFrame
    """
    path = config.PROCESSED_DATA_DIR / "master_dataset.csv"
    
    if not path.exists():
        logger.info("Master dataset not found, creating...")
        return create_master_dataset()
    
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info(f"Loaded master dataset with {len(df)} records")
    
    return df


def get_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    time_column: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation, and test sets.
    
    Args:
        df: Master DataFrame
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        time_column: Column to sort by
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.sort_values(time_column)
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    create_master_dataset()
