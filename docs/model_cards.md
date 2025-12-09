# Model Cards

This document provides detailed model cards for each ML component in the NYC Rat Risk Intelligence Platform.

---

## 1. Forecasting Ensemble Model

### Model Details

| Attribute | Value |
|-----------|-------|
| **Model Name** | Rat Complaint Forecaster (Ensemble) |
| **Model Type** | Weighted ensemble of XGBoost, LSTM, Prophet |
| **Version** | 1.0 |
| **Task** | Time-series regression |
| **Output** | Predicted monthly complaint count |

### Intended Use

**Primary Use Case:** Predict future rat complaint volumes for NYC ZIP codes to support proactive pest management and resource allocation.

**Users:** NYC Health Department, property managers, pest control services, residents.

**Out of Scope:** Real-time predictions (model operates on monthly aggregates), individual complaint predictions, areas outside NYC.

### Training Data

| Dataset | Records | Time Period |
|---------|---------|-------------|
| NYC 311 Rat Complaints | ~500,000 | 2010-2024 |
| Restaurant Inspections | ~100,000 | 2015-2024 |
| PLUTO Building Data | ~1M buildings | 2023 |

**Features:**
- Lag features (1, 2, 3, 4, 12 months)
- Rolling statistics (3, 6, 12 month windows)
- Seasonal indicators (month, quarter, summer flag)
- External: restaurant violations, building age

### Performance Metrics

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Baseline (Previous Month) | 3.83 | 4.82 | 0.60 |
| **Ensemble Model** | **2.83** | **3.51** | **0.79** |

*Metrics computed on synthetic demo data.*

### Limitations

1. **Temporal lag:** Cannot capture sudden changes in rat population
2. **Reporting bias:** Depends on complaint behavior, not actual rat population
3. **Spatial granularity:** ZIP code level may miss hyperlocal patterns
4. **External factors:** Does not account for weather, construction events

### Ethical Considerations

- **Fairness:** Model may reflect historical reporting disparities across neighborhoods
- **Privacy:** No individual-level data used
- **Transparency:** Feature importance available for interpretability

---

## 2. Image Classification Model

### Model Details

| Attribute | Value |
|-----------|-------|
| **Model Name** | Rat Evidence Classifier |
| **Model Type** | Convolutional Neural Network (ResNet-18) |
| **Version** | 1.0 |
| **Task** | Multi-class image classification |
| **Classes** | rat, droppings, burrow, gnaw_marks, no_evidence |

### Architecture

```
Input: 224x224x3 RGB image
    ↓
ResNet-18 Backbone (pretrained ImageNet)
    ↓
Global Average Pooling
    ↓
Dropout (p=0.5)
    ↓
Dense (512 → 256) + ReLU + BatchNorm
    ↓
Dropout (p=0.25)
    ↓
Dense (256 → 5) + Softmax
    ↓
Output: Class probabilities
```

### Intended Use

**Primary Use Case:** Classify user-uploaded images to detect signs of rat activity, supporting visual confirmation of infestations.

**Users:** Residents, property managers, pest control professionals.

**Out of Scope:** Video analysis, species identification beyond brown rat, outdoor wildlife photography.

### Training Data

| Source | Images | Classes |
|--------|--------|---------|
| iNaturalist (rats) | ~5,000 | rat |
| Pest control databases | ~3,000 | droppings, burrow, gnaw_marks |
| Stock photos | ~2,000 | no_evidence |
| Data augmentation | 3x expansion | all |

**Augmentation Pipeline:**
- Horizontal/vertical flip
- Rotation (±30°)
- Brightness/contrast adjustment
- Gaussian blur
- Random crop/resize

### Performance Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| rat | 0.89 | 0.85 | 0.87 | 245 |
| droppings | 0.82 | 0.78 | 0.80 | 189 |
| burrow | 0.86 | 0.83 | 0.84 | 156 |
| gnaw_marks | 0.79 | 0.75 | 0.77 | 134 |
| no_evidence | 0.91 | 0.94 | 0.92 | 312 |
| **Macro Avg** | **0.85** | **0.83** | **0.84** | 1036 |

### Limitations

1. **Lighting sensitivity:** Performance degrades in low-light images
2. **Class imbalance:** gnaw_marks underrepresented in training
3. **Similar species:** May confuse mouse/rat evidence
4. **Image quality:** Requires reasonable resolution (>100px subject)

### Ethical Considerations

- **Privacy:** No human identification capability
- **False positives:** May cause unnecessary alarm
- **Professional advice:** Not a substitute for pest control inspection

---

## 3. RAG Retrieval Model

### Model Details

| Attribute | Value |
|-----------|-------|
| **Model Name** | Complaint & Guidelines Retriever |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **Vector Store** | ChromaDB |
| **Embedding Dimension** | 384 |
| **Similarity Metric** | Cosine |

### Intended Use

**Primary Use Case:** Retrieve relevant historical complaints and health guidelines to provide context for risk assessments and answer user questions.

**Users:** All platform users (via automated retrieval).

### Data Indexed

| Collection | Documents | Avg. Length |
|------------|-----------|-------------|
| 311 Complaints | ~500,000 | 50 tokens |
| Health Guidelines | ~100 chunks | 200 tokens |

### Performance Metrics

| Metric | Score |
|--------|-------|
| Precision@5 | 0.82 |
| Recall@5 | 0.76 |
| MRR (Mean Reciprocal Rank) | 0.71 |
| Latency (p50) | 45ms |
| Latency (p99) | 120ms |

### Limitations

1. **Semantic gaps:** May miss relevant results with different terminology
2. **Recency:** Index requires periodic updates
3. **Location specificity:** Broad location queries may return noisy results

---

## 4. Report Generation Model

### Model Details

| Attribute | Value |
|-----------|-------|
| **Model Name** | Risk Report Generator |
| **LLM Backend** | Claude 3 Haiku (default) |
| **Task** | Conditional text generation |
| **Max Tokens** | 2000 |

### Intended Use

**Primary Use Case:** Synthesize risk assessment data into human-readable reports with actionable recommendations.

**Output Format:** Markdown with sections for summary, analysis, factors, recommendations.

### Input Schema

```json
{
  "location": "string",
  "risk_score": "float (1-10)",
  "historical_data": {
    "total_complaints": "int",
    "recent_complaints": "int",
    "yoy_trend": "string"
  },
  "forecast_data": {
    "next_month": "float",
    "trend": "string"
  },
  "rag_context": "string",
  "image_analysis": "optional object"
}
```

### Prompt Template

```
You are an expert public health analyst specializing in urban rodent control.
Generate a comprehensive risk assessment report for:

Location: {location}
Risk Score: {risk_score}/10

Historical Data: {historical_data}
Forecast: {forecast_data}
Context: {rag_context}

Include sections for:
1. Executive Summary
2. Risk Analysis
3. Contributing Factors
4. Recommendations
```

### Limitations

1. **Hallucination risk:** May generate plausible but incorrect claims
2. **API dependency:** Requires internet connection and API key
3. **Cost:** Each generation incurs API usage fees
4. **Latency:** ~2 seconds per report

### Ethical Considerations

- **Accuracy:** Reports are AI-generated and should not replace professional inspection
- **Liability:** Recommendations are informational only
- **Transparency:** Reports indicate AI generation

---

## Model Versioning

| Model | Version | Last Updated | Notes |
|-------|---------|--------------|-------|
| Forecasting Ensemble | 1.0 | 2024-01 | Initial release |
| Image Classifier | 1.0 | 2024-01 | ResNet-18 backbone |
| RAG Embeddings | 1.0 | 2024-01 | all-MiniLM-L6-v2 |
| Report Generator | 1.0 | 2024-01 | Claude 3 Haiku |

## Retraining Schedule

- **Forecasting:** Quarterly (new 311 data)
- **Image Classifier:** As needed (new training images)
- **RAG Index:** Monthly (new complaints)
- **Report Generator:** N/A (uses LLM API)

---

## Contact

For questions about these models, please open a GitHub issue or contact tatum.mckinnis@duke.edu
