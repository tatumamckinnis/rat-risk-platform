#!/usr/bin/env python3
"""
Data download script for NYC Rat Risk Intelligence Platform.

This script downloads all required datasets from NYC Open Data
and other public sources.

Usage:
    python data/download_data.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_acquisition import download_all_data
from src.data_preprocessing import create_master_dataset

if __name__ == "__main__":
    print("=" * 60)
    print("NYC Rat Risk Intelligence Platform - Data Download")
    print("=" * 60)
    
    # Download all datasets
    print("\n1. Downloading raw datasets...")
    try:
        datasets = download_all_data()
        print(f"   Downloaded {len(datasets)} datasets")
    except Exception as e:
        print(f"   Warning: Some downloads failed: {e}")
        print("   Continuing with available data...")
    
    # Create master dataset
    print("\n2. Creating master dataset...")
    try:
        master_df = create_master_dataset()
        print(f"   Created master dataset with {len(master_df)} records")
    except Exception as e:
        print(f"   Error creating master dataset: {e}")
        print("   You may need to run with synthetic data.")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train models: python src/train_models.py")
    print("  2. Run app: streamlit run src/app.py")
