# Attribution

This document provides detailed attribution for all external resources, AI-generated code, libraries, and datasets used in this project.

## AI-Generated Code Attribution

Portions of this codebase were developed with assistance from Claude (Anthropic). Specific attributions:

### Substantial AI Assistance

| File | Lines | Description |
|------|-------|-------------|
| `src/app.py` | 1-350 | Initial Streamlit application structure and UI components |
| `src/rag_system.py` | 50-120 | ChromaDB integration and retrieval logic |
| `src/report_generator.py` | 1-150 | LLM prompt engineering and response parsing |
| `src/forecasting_models.py` | 200-280 | LSTM architecture definition |
| `notebooks/*.ipynb` | Various | Notebook structure and visualization code |

### Partial AI Assistance

| File | Lines | Description |
|------|-------|-------------|
| `src/data_preprocessing.py` | 80-120 | Data cleaning functions |
| `src/feature_engineering.py` | 1-100 | Feature creation utilities |
| `src/image_classifier.py` | 50-100 | Training loop structure |

### Human-Written (No AI)

| File | Description |
|------|-------------|
| `src/config.py` | Configuration values determined through experimentation |
| `data/download_data.py` | API integration based on NYC Open Data documentation |
| Model hyperparameters | Determined through manual hyperparameter search |

## Datasets

### NYC 311 Rat Sightings
- **Source:** NYC Open Data
- **URL:** https://data.cityofnewyork.us/Social-Services/Rat-Sightings/3q43-55fe
- **License:** Public Domain (NYC Open Data Terms of Use)
- **Citation:** NYC Department of Health and Mental Hygiene

### NYC Restaurant Inspection Results
- **Source:** NYC Open Data
- **URL:** https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j
- **License:** Public Domain
- **Citation:** NYC Department of Health and Mental Hygiene

### PLUTO (Primary Land Use Tax Lot Output)
- **Source:** NYC Department of City Planning
- **URL:** https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page
- **License:** Public Domain
- **Citation:** NYC Department of City Planning

### Rat Evidence Images (Training Data)
- **Source:** Multiple sources combined
- **iNaturalist:** https://www.inaturalist.org/ (CC BY-NC license for individual images)
- **Pest Control Educational Resources:** Various public domain images
- **Original photographs:** Taken by project author

### NYC Health Department Guidelines
- **Source:** NYC Department of Health
- **URL:** https://www.nyc.gov/site/doh/health/health-topics/rats.page
- **License:** Public Domain (US Government work)

## Libraries and Frameworks

### Machine Learning

| Library | Version | License | Use |
|---------|---------|---------|-----|
| PyTorch | 2.0+ | BSD-3-Clause | Deep learning framework |
| scikit-learn | 1.3+ | BSD-3-Clause | Classical ML models |
| XGBoost | 2.0+ | Apache 2.0 | Gradient boosting |
| Prophet | 1.1+ | MIT | Time-series forecasting |
| Transformers | 4.35+ | Apache 2.0 | Pre-trained models |
| sentence-transformers | 2.2+ | Apache 2.0 | Text embeddings |

### NLP and RAG

| Library | Version | License | Use |
|---------|---------|---------|-----|
| LangChain | 0.1+ | MIT | RAG orchestration |
| ChromaDB | 0.4+ | Apache 2.0 | Vector database |
| tiktoken | 0.5+ | MIT | Token counting |

### Data Processing

| Library | Version | License | Use |
|---------|---------|---------|-----|
| pandas | 2.0+ | BSD-3-Clause | Data manipulation |
| NumPy | 1.24+ | BSD-3-Clause | Numerical computing |
| Pillow | 10.0+ | HPND | Image processing |
| albumentations | 1.3+ | MIT | Image augmentation |

### Visualization

| Library | Version | License | Use |
|---------|---------|---------|-----|
| Matplotlib | 3.7+ | PSF | Plotting |
| Seaborn | 0.12+ | BSD-3-Clause | Statistical visualization |
| Plotly | 5.15+ | MIT | Interactive plots |
| Folium | 0.14+ | MIT | Map visualization |

### Web Application

| Library | Version | License | Use |
|---------|---------|---------|-----|
| Streamlit | 1.28+ | Apache 2.0 | Web framework |
| streamlit-folium | 0.15+ | MIT | Map integration |

### APIs

| Service | Use | Terms |
|---------|-----|-------|
| Anthropic Claude API | Report generation | https://www.anthropic.com/terms |
| NYC Open Data API | Data retrieval | https://opendata.cityofnewyork.us/overview/#termsofuse |

## Pre-trained Models

### ResNet-18 (Image Classification Base)
- **Source:** torchvision
- **Original Paper:** He et al., "Deep Residual Learning for Image Recognition" (2016)
- **License:** BSD-3-Clause
- **Modification:** Fine-tuned final layers on rat evidence dataset

### all-MiniLM-L6-v2 (Text Embeddings)
- **Source:** Sentence Transformers / Hugging Face
- **Original:** Microsoft
- **License:** Apache 2.0
- **Use:** Embedding 311 complaints for semantic search

## Code References and Inspiration

### Time-Series Forecasting
- Prophet documentation examples: https://facebook.github.io/prophet/
- "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos (online textbook)

### RAG Implementation
- LangChain documentation: https://python.langchain.com/docs/
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)

### Image Classification
- PyTorch Transfer Learning Tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## Acknowledgments

- NYC Open Data team for maintaining accessible public datasets
- Course instructors and TAs for guidance on project direction
- Open source community for the excellent ML ecosystem
