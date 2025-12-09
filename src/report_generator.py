"""
Report generation module for NYC Rat Risk Intelligence Platform.

This module uses LLMs (Ollama local, Claude, or GPT) to synthesize risk 
assessment findings into comprehensive, actionable reports.

Ollama is the default and runs completely locally with no API keys required.
"""

import logging
from typing import Dict, List, Optional
import json

from . import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates risk assessment reports using LLMs."""
    
    def __init__(
        self,
        provider: str = None,
        model: str = None,
    ):
        """
        Initialize the report generator.
        
        Args:
            provider: LLM provider ("ollama", "anthropic", or "openai")
            model: Model name (uses config default if not specified)
        """
        # Auto-detect provider if not specified
        if provider is None:
            provider = self._auto_detect_provider()
            
        self.provider = provider
        
        if provider == "ollama":
            self.model = model or config.OLLAMA_MODEL
            self._init_ollama()
        elif provider == "anthropic":
            self.model = model or config.ANTHROPIC_MODEL
            self._init_anthropic()
        elif provider == "openai":
            self.model = model or config.OPENAI_MODEL
            self._init_openai()
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
        self.system_prompt = config.REPORT_SYSTEM_PROMPT
        
    def _auto_detect_provider(self) -> str:
        """Auto-detect the best available provider."""
        # First, try Ollama (local, free)
        if self._check_ollama_available():
            logger.info("Auto-detected Ollama (local LLM)")
            return "ollama"
        # Then try Anthropic
        if config.ANTHROPIC_API_KEY:
            logger.info("Auto-detected Anthropic API key")
            return "anthropic"
        # Then try OpenAI
        if config.OPENAI_API_KEY:
            logger.info("Auto-detected OpenAI API key")
            return "openai"
        # Default to Ollama (will show setup instructions if not running)
        logger.info("No LLM detected, defaulting to Ollama")
        return "ollama"
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        self.client = None
        self.ollama_available = self._check_ollama_available()
        
        if self.ollama_available:
            logger.info(f"Ollama available, using model: {self.model}")
            # Check if model is downloaded
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags")
                models = [m["name"] for m in response.json().get("models", [])]
                if not any(self.model in m for m in models):
                    logger.warning(f"Model {self.model} not found. Run: ollama pull {self.model}")
            except:
                pass
        else:
            logger.warning("Ollama not running. Start with: ollama serve")
            logger.info("Or install from: https://ollama.ai")
        
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
            
            if not config.ANTHROPIC_API_KEY:
                logger.warning("ANTHROPIC_API_KEY not set, using demo mode")
                self.client = None
            else:
                self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
                logger.info(f"Initialized Anthropic client with model: {self.model}")
        except ImportError:
            logger.warning("anthropic package not installed")
            self.client = None
            
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            if not config.OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY not set, using demo mode")
                self.client = None
            else:
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                logger.info(f"Initialized OpenAI client with model: {self.model}")
        except ImportError:
            logger.warning("openai package not installed")
            self.client = None
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama local API."""
        if not self.ollama_available:
            return self._demo_response(prompt)
        
        try:
            import requests
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{self.system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2000,
                    }
                },
                timeout=120,  # Ollama can be slow on first run
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return self._demo_response(prompt)
                
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return self._demo_response(prompt)
            
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        if self.client is None:
            return self._demo_response(prompt)
            
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        return response.content[0].text
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        if self.client is None:
            return self._demo_response(prompt)
            
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        
        return response.choices[0].message.content
    
    def _demo_response(self, prompt: str) -> str:
        """Generate a demo response when no API key is available."""
        return """
## Risk Assessment Summary

Based on the available data, this location shows **moderate rat activity risk**.

### Key Findings

1. **Historical Activity**: The area has seen periodic rat sighting reports over the past year, with seasonal peaks during summer months.

2. **Contributing Factors**: 
   - Nearby food establishments with recent rodent violations
   - Older building stock (average age 50+ years)
   - Urban density creating multiple harborage opportunities

3. **Forecast Outlook**: Our model predicts stable to slightly elevated risk over the next 3 months, with potential increases during warmer weather.

### Recommendations

1. **Immediate Actions**:
   - Inspect and seal any gaps larger than 1/4 inch around building perimeter
   - Ensure garbage is stored in sealed containers
   - Remove any standing water sources

2. **Ongoing Prevention**:
   - Schedule regular pest control inspections
   - Maintain vegetation and remove debris
   - Report any sightings to 311 promptly

3. **For Property Owners**:
   - Review NYC Health Code requirements for rodent control
   - Consider professional pest management services
   - Address any structural issues that could provide entry points

*Note: This is a demonstration response. Connect an API key for personalized analysis.*
"""
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if config.DEMO_MODE:
            return self._demo_response(prompt)
        
        if self.provider == "ollama":
            return self._call_ollama(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            return self._demo_response(prompt)
            
    def generate_risk_report(
        self,
        location: str,
        risk_score: float,
        historical_data: Dict,
        forecast_data: Dict,
        image_analysis: Dict = None,
        rag_context: str = None,
    ) -> str:
        """
        Generate a comprehensive risk assessment report.
        
        Args:
            location: Address or location description
            risk_score: Calculated risk score (1-10)
            historical_data: Historical complaint data
            forecast_data: Forecasting model predictions
            image_analysis: Image classification results (if available)
            rag_context: Retrieved context from RAG system
            
        Returns:
            Generated report as markdown string
        """
        # Build the prompt
        prompt_parts = [
            f"Generate a comprehensive rat risk assessment report for the following location:\n",
            f"**Location:** {location}\n",
            f"**Overall Risk Score:** {risk_score:.1f}/10\n\n",
        ]
        
        # Add historical data
        prompt_parts.append("## Historical Data\n")
        prompt_parts.append(f"- Total complaints in area: {historical_data.get('total_complaints', 'N/A')}\n")
        prompt_parts.append(f"- Recent complaints (last 6 months): {historical_data.get('recent_complaints', 'N/A')}\n")
        prompt_parts.append(f"- Year-over-year trend: {historical_data.get('yoy_trend', 'N/A')}\n")
        
        if historical_data.get('sample_descriptions'):
            prompt_parts.append("- Sample complaint descriptions:\n")
            for desc in historical_data['sample_descriptions'][:3]:
                prompt_parts.append(f"  - {desc}\n")
                
        # Add forecast data
        prompt_parts.append("\n## Forecast Data\n")
        prompt_parts.append(f"- Predicted complaints (next month): {forecast_data.get('next_month', 'N/A')}\n")
        prompt_parts.append(f"- Predicted complaints (next 3 months): {forecast_data.get('next_3_months', 'N/A')}\n")
        prompt_parts.append(f"- Forecast confidence: {forecast_data.get('confidence', 'N/A')}\n")
        prompt_parts.append(f"- Trend direction: {forecast_data.get('trend', 'N/A')}\n")
        
        # Add image analysis if available
        if image_analysis:
            prompt_parts.append("\n## Image Analysis\n")
            prompt_parts.append(f"- Detected evidence: {image_analysis.get('predicted_class', 'N/A')}\n")
            prompt_parts.append(f"- Confidence: {image_analysis.get('confidence', 0):.1%}\n")
            
            if image_analysis.get('is_rat_evidence'):
                prompt_parts.append("- **WARNING**: Rat evidence detected in uploaded image\n")
                
        # Add RAG context
        if rag_context:
            prompt_parts.append("\n## Retrieved Context\n")
            prompt_parts.append(rag_context)
            prompt_parts.append("\n")
            
        # Add instructions
        prompt_parts.append("""
## Instructions

Based on the above data, generate a comprehensive risk assessment report with the following sections:

1. **Executive Summary** (2-3 sentences summarizing the key findings)
2. **Risk Analysis** (explain the risk score and contributing factors)
3. **Historical Patterns** (discuss what the complaint history reveals)
4. **Forecast Outlook** (interpret the model predictions)
5. **Contributing Factors** (list specific factors increasing/decreasing risk)
6. **Recommendations** (provide actionable steps for the user)
7. **Resources** (mention relevant NYC resources like 311, Health Department)

Use markdown formatting. Be specific and cite the data provided. Maintain a professional but accessible tone.
""")
        
        prompt = "".join(prompt_parts)
        
        return self.generate(prompt)
    
    def generate_quick_summary(
        self,
        risk_score: float,
        key_factors: List[str],
    ) -> str:
        """
        Generate a quick one-paragraph summary.
        
        Args:
            risk_score: Risk score (1-10)
            key_factors: List of key contributing factors
            
        Returns:
            Short summary string
        """
        risk_level = "low" if risk_score < 4 else "moderate" if risk_score < 7 else "high"
        
        prompt = f"""
Generate a single paragraph (3-4 sentences) summarizing a rat risk assessment with:
- Risk level: {risk_level} ({risk_score:.1f}/10)
- Key factors: {', '.join(key_factors[:3])}

Be concise and actionable. Start with the risk level, explain why, and give one key recommendation.
"""
        
        return self.generate(prompt)
    
    def answer_question(
        self,
        question: str,
        context: str,
        location: str = None,
    ) -> str:
        """
        Answer a user question using RAG context.
        
        Args:
            question: User's question
            context: Retrieved context from RAG
            location: Location context (optional)
            
        Returns:
            Generated answer
        """
        prompt = f"""
Answer the following question about rat control and prevention in NYC.

Question: {question}

{"Location context: " + location if location else ""}

Relevant information from our knowledge base:
{context}

Provide a helpful, accurate answer based on the above information. If the context doesn't contain relevant information, provide general guidance based on NYC Health Department recommendations. Be specific and actionable.
"""
        
        return self.generate(prompt)
    
    def format_risk_score_explanation(self, risk_score: float) -> str:
        """
        Get an explanation for a risk score.
        
        Args:
            risk_score: Risk score (1-10)
            
        Returns:
            Explanation string
        """
        if risk_score < 3:
            return "🟢 **Low Risk** - Minimal rat activity indicators. Maintain standard prevention practices."
        elif risk_score < 5:
            return "🟡 **Low-Moderate Risk** - Some activity indicators present. Consider enhanced monitoring."
        elif risk_score < 7:
            return "🟠 **Moderate Risk** - Notable activity indicators. Active prevention measures recommended."
        elif risk_score < 8.5:
            return "🔴 **High Risk** - Significant activity indicators. Immediate action recommended."
        else:
            return "⚫ **Very High Risk** - Critical activity levels. Professional intervention strongly recommended."


def get_report_generator() -> ReportGenerator:
    """
    Get a configured report generator.
    
    Priority: Ollama (local) > Anthropic > OpenAI > Demo mode
    
    Returns:
        ReportGenerator instance
    """
    # Auto-detect will handle the priority
    return ReportGenerator()


if __name__ == "__main__":
    # Test report generation
    logger.info("Testing report generator...")
    
    generator = get_report_generator()
    
    # Test quick summary
    summary = generator.generate_quick_summary(
        risk_score=6.5,
        key_factors=["nearby restaurant violations", "older buildings", "summer season"],
    )
    print("Quick Summary:")
    print(summary)
    print("\n" + "="*50 + "\n")
    
    # Test full report
    report = generator.generate_risk_report(
        location="123 Main St, Brooklyn, NY 11201",
        risk_score=6.5,
        historical_data={
            "total_complaints": 45,
            "recent_complaints": 12,
            "yoy_trend": "+15%",
            "sample_descriptions": [
                "Rat seen near garbage cans",
                "Rodent droppings in alley",
                "Burrow holes near building foundation",
            ],
        },
        forecast_data={
            "next_month": 4,
            "next_3_months": 14,
            "confidence": "medium",
            "trend": "stable",
        },
    )
    print("Full Report:")
    print(report)
