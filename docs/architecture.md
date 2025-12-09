# System Architecture

This document describes the architecture of the NYC Rat Risk Intelligence Platform.

## Overview

The platform is a multimodal machine learning system that combines multiple data sources and ML models to provide comprehensive rat risk assessments for any NYC location.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE (Streamlit)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Address Input  │  Image Upload  │  Question Input  │  Results Display      │
└────────┬────────┴───────┬────────┴────────┬─────────┴───────────────────────┘
         │                │                 │
         ▼                ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING LAYER                                     │
├───────────────┬─────────────────┬──────────────────┬────────────────────────┤
│   Geocoding   │ Image Classifier│   RAG System     │  Report Generator      │
│   (geopy)     │ (ResNet-18)     │  (ChromaDB)      │  (Claude API)          │
└───────┬───────┴────────┬────────┴─────────┬────────┴───────────┬────────────┘
        │                │                  │                    │
        ▼                ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML MODELS LAYER                                    │
├─────────────────┬────────────────────┬──────────────────────────────────────┤
│   Forecasting   │  Risk Scorer       │   Embedding Model                    │
│   (Ensemble)    │  (Multi-factor)    │   (all-MiniLM)                       │
└────────┬────────┴─────────┬──────────┴──────────────┬───────────────────────┘
         │                  │                         │
         ▼                  ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
├────────────────┬─────────────────┬──────────────────┬───────────────────────┤
│  311 Complaints│ Restaurant Data │  Building Data   │  Health Guidelines    │
│  (NYC Open Data│ (NYC Open Data) │  (PLUTO)         │  (NYC DOH)            │
└────────────────┴─────────────────┴──────────────────┴───────────────────────┘
```

## Component Details

### 1. Data Acquisition Layer

**Source:** `src/data_acquisition.py`

Responsible for downloading and caching data from:
- NYC Open Data API (311 complaints, restaurant inspections)
- NYC Department of City Planning (PLUTO building data)
- NYC Department of Health (guidelines PDFs)

Key Classes:
- `NYCOpenDataClient`: Handles API authentication and pagination
- Download functions for each dataset type

### 2. Data Preprocessing Layer

**Source:** `src/data_preprocessing.py`

Handles:
- Data cleaning and standardization
- Date parsing and feature extraction
- Spatial aggregation by ZIP code
- Dataset merging (311 + restaurants + buildings)

Key Functions:
- `clean_rat_sightings()`: Process 311 complaint data
- `aggregate_by_location_time()`: Create time-series features
- `create_master_dataset()`: Merge all data sources

### 3. Feature Engineering Layer

**Source:** `src/feature_engineering.py`

Creates ML-ready features:
- Lag features (1, 2, 3, 4, 12 months)
- Rolling statistics (mean, std, max over 3, 6, 12 months)
- Cyclical time encodings (month sin/cos)
- Categorical encodings (borough, season)

Key Class:
- `FeatureEngineer`: Scikit-learn style fit/transform interface

### 4. Forecasting Models Layer

**Source:** `src/forecasting_models.py`

Implements multiple forecasting approaches:

| Model | Architecture | Use Case |
|-------|--------------|----------|
| XGBoost | Gradient boosting | Tabular features |
| LSTM | Bidirectional LSTM | Sequential patterns |
| Prophet | Additive model | Trend + seasonality |
| Ensemble | Weighted average | Combined prediction |

Key Classes:
- `XGBoostForecaster`: Wrapper around XGBRegressor
- `LSTMForecaster`: PyTorch LSTM with training loop
- `ProphetForecaster`: Per-location Prophet models
- `EnsembleForecaster`: Weighted model combination

### 5. Image Classification Layer

**Source:** `src/image_classifier.py`

CNN-based classifier for rat evidence detection:

**Architecture:**
```
ResNet-18 (pretrained)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Dense (512 → 256)
    ↓
ReLU + BatchNorm
    ↓
Dropout (0.25)
    ↓
Dense (256 → 5 classes)
```

**Classes:** rat, droppings, burrow, gnaw_marks, no_evidence

Key Classes:
- `RatEvidenceClassifier`: PyTorch model
- `ImageClassifierTrainer`: Training and inference wrapper

### 6. RAG System Layer

**Source:** `src/rag_system.py`

Retrieval-Augmented Generation for semantic search:

**Components:**
1. **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
2. **Vector Store**: ChromaDB with cosine similarity
3. **Document Collections**:
   - 311 complaints (with metadata)
   - Health guidelines (chunked)

Key Classes:
- `EmbeddingModel`: Sentence transformer wrapper
- `ComplaintRAG`: Search over complaint history
- `GuidelinesRAG`: Search over health guidelines
- `RAGSystem`: Combined retrieval interface

### 7. Risk Scoring Layer

**Source:** `src/risk_scorer.py`

Multi-factor risk calculation:

**Factors and Weights:**
| Factor | Weight | Description |
|--------|--------|-------------|
| Historical complaints | 30% | Past complaint density |
| Forecast risk | 25% | Predicted future activity |
| Restaurant violations | 20% | Nearby food establishment issues |
| Building age | 15% | Older buildings = higher risk |
| Construction | 10% | Nearby activity |

Key Class:
- `RiskScorer`: Calculates 1-10 risk score from inputs

### 8. Report Generation Layer

**Source:** `src/report_generator.py`

LLM-based report synthesis:

**Flow:**
1. Collect inputs (risk score, history, forecast, RAG context)
2. Construct structured prompt
3. Call Claude API (or demo mode)
4. Return formatted markdown report

Key Class:
- `ReportGenerator`: Anthropic/OpenAI API wrapper

### 9. Web Application Layer

**Source:** `src/app.py`

Streamlit-based user interface:

**Pages/Sections:**
1. Address lookup with geocoding
2. Image upload and classification
3. Question answering with RAG
4. Risk report display
5. Interactive map visualization

## Data Flow

### Risk Assessment Flow

```
User enters address
        │
        ▼
    Geocoding
        │
        ▼
┌───────┴───────┐
│               │
▼               ▼
Query       Load
Historical  Building
Data        Data
│               │
▼               ▼
Feature     Calculate
Engineering Risk Factors
│               │
└───────┬───────┘
        │
        ▼
  Forecasting
  (XGB/LSTM/Prophet)
        │
        ▼
  Risk Scoring
  (weighted combination)
        │
        ▼
  RAG Retrieval
  (complaints + guidelines)
        │
        ▼
  Report Generation
  (Claude API)
        │
        ▼
  Display Results
```

### Image Analysis Flow

```
User uploads image
        │
        ▼
Image Preprocessing
(resize, normalize)
        │
        ▼
CNN Classification
(ResNet-18)
        │
        ▼
Class Probabilities
        │
        ▼
Update Risk Score
(if evidence found)
        │
        ▼
Include in Report
```

## Deployment

### Local Development
```bash
streamlit run src/app.py
```

### Production Considerations

1. **API Rate Limits**: Cache NYC Open Data responses
2. **Model Loading**: Pre-load models at startup
3. **GPU**: Optional for image classification
4. **LLM Costs**: Consider response caching

## Dependencies

Core ML:
- PyTorch, scikit-learn, XGBoost, Prophet

NLP/Embeddings:
- sentence-transformers, LangChain, ChromaDB

APIs:
- anthropic, openai, sodapy (NYC Open Data)

Web:
- Streamlit, Folium (maps)

## Future Enhancements

1. **Real-time 311 integration**: Stream new complaints
2. **Model retraining pipeline**: Automated retraining
3. **Mobile app**: React Native frontend
4. **Multi-city expansion**: Other urban areas
