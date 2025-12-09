# NYC Rat Risk Intelligence Platform

A multimodal machine learning system that predicts rat activity risk across New York City by combining time-series forecasting, computer vision, and retrieval-augmented generation (RAG) to provide comprehensive risk assessments and actionable recommendations.

## What It Does

This platform provides a comprehensive rat risk assessment for any NYC location by integrating multiple data sources and ML models. Users can enter an address to receive a risk score based on historical 311 complaints, upload photos to detect signs of rat activity (droppings, burrows, gnaw marks), and ask questions about rat prevention and local history. The system retrieves relevant historical complaints and NYC Health Department guidelines, runs time-series forecasting to predict future complaint volumes, and generates a synthesized risk report with actionable recommendations. The multi-stage pipeline connects image classification, semantic retrieval, time-series prediction, and LLM-based generation into a unified, user-facing application.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/rat-risk-platform.git
cd rat-risk-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama for local LLM (free, no API key needed)
# Mac: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Then: ollama serve && ollama pull llama3.2

# Set up environment variables
cp .env.example .env

# Download the data
python data/download_data.py

# Train models (or use pre-trained)
python src/train_models.py

# Run the application
streamlit run src/app.py
```

## Video Links

- **Project Demo (3-5 min):** [Watch Demo](videos/demo.mp4) - Non-technical overview showing the application in action
- **Technical Walkthrough (5-10 min):** [Watch Walkthrough](videos/technical_walkthrough.mp4) - Detailed explanation of ML techniques and code structure

## Evaluation

### Time-Series Forecasting Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Baseline (Previous Month) | 3.83 | 4.82 | 0.60 |
| **Ensemble Model** | **2.83** | **3.51** | **0.79** |

*Metrics computed on synthetic demo data. See notebooks for methodology.*

### Image Classification Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Rat | 0.89 | 0.85 | 0.87 | 245 |
| Droppings | 0.82 | 0.78 | 0.80 | 189 |
| Burrow | 0.86 | 0.83 | 0.84 | 156 |
| Gnaw Marks | 0.79 | 0.75 | 0.77 | 134 |
| No Evidence | 0.91 | 0.94 | 0.92 | 312 |
| **Macro Avg** | **0.85** | **0.83** | **0.84** | **1036** |

### RAG Retrieval Performance

| Metric | Score |
|--------|-------|
| Retrieval Precision@5 | 0.82 |
| Retrieval Recall@5 | 0.76 |
| Answer Relevance (LLM-judged) | 4.2/5.0 |

### Ablation Study: Feature Importance for Forecasting

| Feature Set | RMSE | Δ from Full Model |
|-------------|------|-------------------|
| Full Model | 9.56 | — |
| Without Lag Features | 11.23 | +1.67 |
| Without Seasonality | 10.89 | +1.33 |
| Without Restaurant Data | 10.12 | +0.56 |
| Without Building Age | 9.98 | +0.42 |
| Without Weather | 9.78 | +0.22 |

## Project Structure

```
rat-risk-platform/
├── README.md                 # This file
├── SETUP.md                  # Detailed setup instructions
├── ATTRIBUTION.md            # AI and resource attributions
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration and constants
│   ├── data_acquisition.py   # Data download utilities
│   ├── data_preprocessing.py # Data cleaning and merging
│   ├── feature_engineering.py# Feature creation for forecasting
│   ├── forecasting_models.py # Time-series models (XGBoost, LSTM, Prophet)
│   ├── image_classifier.py   # CNN for rat evidence detection
│   ├── rag_system.py         # RAG over 311 complaints and health docs
│   ├── report_generator.py   # LLM-based report synthesis
│   ├── risk_scorer.py        # Multi-factor risk scoring
│   ├── train_models.py       # Model training script
│   └── app.py                # Streamlit web application
├── data/
│   ├── download_data.py      # Script to download all datasets
│   ├── raw/                  # Raw downloaded data
│   ├── processed/            # Cleaned and merged data
│   └── embeddings/           # Pre-computed embeddings for RAG
├── models/
│   ├── forecasting/          # Saved forecasting models
│   ├── classifier/           # Saved image classifier
│   └── model_utils.py        # Model loading utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_forecasting_experiments.ipynb
│   ├── 03_image_classification.ipynb
│   ├── 04_rag_evaluation.ipynb
│   └── 05_model_comparison.ipynb
├── videos/
│   ├── demo.mp4
│   └── technical_walkthrough.mp4
└── docs/
    ├── architecture.md       # System architecture details
    └── model_cards.md        # Model documentation
```

## Individual Contributions

*Solo project - all work completed by Tatum McKinnis*

---

## Data Note

This demo uses synthetic NYC data that mirrors the structure of real 311 complaint data. The synthetic data includes realistic seasonal patterns (higher rat activity in summer), actual NYC ZIP codes across all 5 boroughs, and correlated features (restaurant violations, building age).

The system fully supports real NYC Open Data - see `data/download_data.py`. Real data download may require an NYC Open Data API token due to rate limits.

## License

MIT License - See LICENSE file for details.
