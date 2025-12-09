# Setup Instructions

This document provides detailed instructions for setting up and running the NYC Rat Risk Intelligence Platform.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Git
- At least 8GB RAM (16GB recommended for model training)
- CUDA-compatible GPU (optional, but recommended for faster training)

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/rat-risk-platform.git
cd rat-risk-platform
```

## Step 2: Create Virtual Environment

We recommend using a virtual environment to avoid dependency conflicts.

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Troubleshooting Dependencies

If you encounter issues with PyTorch, install it separately first:
```bash
# CPU only
pip install torch torchvision

# With CUDA (check your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If Prophet fails to install:
```bash
pip install pystan==2.19.1.1
pip install prophet
```

## Step 4: Set Up Local LLM (Ollama) - Recommended

The easiest way to run this project is with Ollama, a free local LLM that requires **no API keys**.

### Install Ollama

**Mac:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai/download

### Start Ollama and Download Model

```bash
# Start the Ollama server (runs in background)
ollama serve

# In a new terminal, download the model (~2GB)
ollama pull llama3.2
```

That's it! The app will automatically detect Ollama when it's running.

### Alternative Models (optional)

If you have more RAM/GPU, you can use larger models:
```bash
ollama pull llama3.1      # 8B params, better quality
ollama pull mistral       # 7B params, fast
ollama pull phi3          # 3.8B params, very fast
```

Set your preferred model in `.env`:
```
OLLAMA_MODEL=llama3.1
```

## Step 5: (Optional) Cloud API Keys

If you prefer cloud LLMs instead of Ollama, you can use Anthropic or OpenAI:

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your preferred text editor:

```bash
# If using Ollama (recommended - no key needed):
OLLAMA_MODEL=llama3.2

# If using Anthropic instead:
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# If using OpenAI instead:
OPENAI_API_KEY=your_openai_api_key_here

# Optional: NYC Open Data token (increases rate limits)
NYC_OPEN_DATA_TOKEN=your_token_here
```

### Getting API Keys (only if not using Ollama)

1. **Anthropic API Key**:
   - Visit https://console.anthropic.com/
   - Create an account and generate an API key

2. **OpenAI API Key**:
   - Visit https://platform.openai.com/
   - Create an account and generate an API key

3. **NYC Open Data Token** (Optional but recommended):
   - Visit https://data.cityofnewyork.us/
   - Create an account
   - Go to Developer Settings to create an app token
   - This increases API rate limits from 1000 to 50000 requests/hour

## Step 6: Download Data

Run the data download script to fetch all required datasets:

```bash
python data/download_data.py
```

This will download:
- NYC 311 Rat Sightings (2010-present)
- Restaurant Inspection Results
- PLUTO Building Data (simplified)
- NYC Health Department Guidelines (PDFs)

**Note:** The full download may take 10-30 minutes depending on your internet connection. Data files total approximately 2GB.

### Manual Data Download (if script fails)

If the automated download fails, you can manually download from:

1. **311 Rat Sightings:**
   https://data.cityofnewyork.us/Social-Services/Rat-Sightings/3q43-55fe
   - Export as CSV, save to `data/raw/rat_sightings.csv`

2. **Restaurant Inspections:**
   https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j
   - Export as CSV, save to `data/raw/restaurant_inspections.csv`

3. **PLUTO Data:**
   https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page
   - Download the CSV version, save to `data/raw/pluto.csv`

## Step 7: Train Models

Train all models with default configurations:

```bash
python src/train_models.py
```

Or train individual components:

```bash
# Train only forecasting models
python src/train_models.py --component forecasting

# Train only image classifier
python src/train_models.py --component classifier

# Build only RAG index
python src/train_models.py --component rag
```

**Training Times (approximate):**
- Forecasting models: 10-20 minutes
- Image classifier: 30-60 minutes (GPU), 2-4 hours (CPU)
- RAG index: 5-10 minutes

### Using Pre-trained Models

If you want to skip training, pre-trained model weights are available:

```bash
# Download pre-trained models
python scripts/download_pretrained.py
```

## Step 8: Run the Application

Start the Streamlit web application:

```bash
streamlit run src/app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Step 9: Testing the System

To verify everything is working:

1. **Test Address Lookup:**
   - Enter "123 Main St, Brooklyn, NY" in the address field
   - Click "Assess Risk"
   - You should see a risk score and historical data

2. **Test Image Upload:**
   - Upload any image (or use samples in `data/test_images/`)
   - The system should classify it

3. **Test Question Answering:**
   - Ask "How do I prevent rats in my building?"
   - You should receive a relevant response

## Configuration Options

The application can be customized via `src/config.py`:

```python
# Model selection
FORECASTING_MODEL = "ensemble"  # Options: xgboost, lstm, prophet, ensemble
CLASSIFIER_MODEL = "resnet18"   # Options: resnet18, resnet50, efficientnet

# RAG settings
RAG_TOP_K = 5                   # Number of documents to retrieve
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Risk scoring weights
WEIGHTS = {
    "historical_complaints": 0.3,
    "forecast_risk": 0.25,
    "restaurant_violations": 0.2,
    "building_age": 0.15,
    "nearby_construction": 0.1
}
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Common Issues

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `src/config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### Issue: "Rate limit exceeded" during data download
**Solution:** Add NYC Open Data token to `.env` or wait and retry.

### Issue: Streamlit not finding modules
**Solution:** Run from project root directory:
```bash
cd rat-risk-platform
streamlit run src/app.py
```

### Issue: Prophet installation fails on M1 Mac
**Solution:**
```bash
brew install cmake
pip install prophet
```

## For Graders

To test this system with full LLM functionality (no API keys needed):

1. Install Ollama from https://ollama.ai
2. Run `ollama serve` in one terminal
3. Run `ollama pull llama3.2` in another terminal
4. The app will automatically use Ollama for report generation

Alternative (without any LLM):
1. Set `DEMO_MODE=true` in `.env`
2. The system will use pre-written placeholder reports
3. All other ML models still run locally

## Contact

For issues or questions, please open a GitHub issue or contact [your email].
