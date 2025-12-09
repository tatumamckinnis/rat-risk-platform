"""
Risk scoring module for NYC Rat Risk Intelligence Platform.

This module combines multiple factors (historical complaints, forecasts,
restaurant violations, building age, etc.) into a unified risk score.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from . import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class RiskFactors:
    """Container for risk factor values."""
    historical_complaints: float = 0.0
    forecast_risk: float = 0.0
    restaurant_violations: float = 0.0
    building_age: float = 0.0
    nearby_construction: float = 0.0
    seasonal_factor: float = 0.0
    image_evidence: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "historical_complaints": self.historical_complaints,
            "forecast_risk": self.forecast_risk,
            "restaurant_violations": self.restaurant_violations,
            "building_age": self.building_age,
            "nearby_construction": self.nearby_construction,
            "seasonal_factor": self.seasonal_factor,
            "image_evidence": self.image_evidence,
        }


class RiskScorer:
    """
    Multi-factor risk scoring system.
    
    Combines multiple risk indicators into a single score (1-10).
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
        historical_data: pd.DataFrame = None,
    ):
        """
        Initialize the risk scorer.
        
        Args:
            weights: Dictionary of factor weights
            historical_data: Historical complaint data for normalization
        """
        self.weights = weights or config.RISK_WEIGHTS
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Statistics for normalization (will be computed from data)
        self.stats = {
            "complaints_mean": 10,
            "complaints_std": 5,
            "complaints_max": 50,
            "violations_mean": 5,
            "violations_std": 3,
            "building_age_mean": 50,
            "building_age_std": 30,
        }
        
        if historical_data is not None:
            self._compute_stats(historical_data)
            
    def _compute_stats(self, df: pd.DataFrame):
        """Compute normalization statistics from historical data."""
        if "complaint_count" in df.columns:
            self.stats["complaints_mean"] = df["complaint_count"].mean()
            self.stats["complaints_std"] = df["complaint_count"].std()
            self.stats["complaints_max"] = df["complaint_count"].max()
            
        if "restaurant_violations_nearby" in df.columns:
            self.stats["violations_mean"] = df["restaurant_violations_nearby"].mean()
            self.stats["violations_std"] = df["restaurant_violations_nearby"].std()
            
        if "building_age_mean" in df.columns:
            self.stats["building_age_mean"] = df["building_age_mean"].mean()
            self.stats["building_age_std"] = df["building_age_mean"].std()
            
        logger.info(f"Computed normalization stats: {self.stats}")
        
    def _normalize_to_10(self, value: float, mean: float, std: float) -> float:
        """
        Normalize a value to 0-10 scale using z-score.
        
        Args:
            value: Raw value
            mean: Population mean
            std: Population standard deviation
            
        Returns:
            Normalized score (0-10)
        """
        if std == 0:
            return 5.0
            
        # Z-score
        z = (value - mean) / std
        
        # Convert to 0-10 scale (roughly -2 to +2 std maps to 0-10)
        score = 5 + (z * 2.5)
        
        # Clip to valid range
        return np.clip(score, 0, 10)
    
    def _score_historical_complaints(
        self,
        complaint_count: float,
        time_period_months: int = 12,
    ) -> float:
        """
        Score based on historical complaint volume.
        
        Args:
            complaint_count: Number of complaints in the period
            time_period_months: Time period for the count
            
        Returns:
            Risk score (0-10)
        """
        # Normalize by time period
        monthly_rate = complaint_count / time_period_months
        
        return self._normalize_to_10(
            monthly_rate,
            self.stats["complaints_mean"] / 12,
            self.stats["complaints_std"] / 12,
        )
    
    def _score_forecast(
        self,
        predicted_complaints: float,
        historical_average: float = None,
    ) -> float:
        """
        Score based on forecasted complaint volume.
        
        Args:
            predicted_complaints: Predicted complaints for next period
            historical_average: Historical average for comparison
            
        Returns:
            Risk score (0-10)
        """
        historical_average = historical_average or self.stats["complaints_mean"]
        
        # Score based on how much above/below average
        ratio = predicted_complaints / max(historical_average, 1)
        
        # Map ratio to 0-10 (1.0 = average = 5)
        score = 5 * ratio
        
        return np.clip(score, 0, 10)
    
    def _score_restaurant_violations(
        self,
        violation_count: float,
    ) -> float:
        """
        Score based on nearby restaurant rodent violations.
        
        Args:
            violation_count: Number of violations in area
            
        Returns:
            Risk score (0-10)
        """
        return self._normalize_to_10(
            violation_count,
            self.stats["violations_mean"],
            self.stats["violations_std"],
        )
    
    def _score_building_age(
        self,
        average_age: float,
    ) -> float:
        """
        Score based on building age (older = higher risk).
        
        Args:
            average_age: Average building age in years
            
        Returns:
            Risk score (0-10)
        """
        # Older buildings have higher risk
        # Map age to score: <20 years = low, >80 years = high
        if average_age < 20:
            return 2.0
        elif average_age < 40:
            return 4.0
        elif average_age < 60:
            return 6.0
        elif average_age < 80:
            return 7.5
        else:
            return 9.0
    
    def _score_seasonal(self, month: int) -> float:
        """
        Score based on seasonal factors.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Risk score (0-10)
        """
        # Rats are most active in summer/early fall
        seasonal_scores = {
            1: 4.0,   # January
            2: 3.5,   # February
            3: 4.0,   # March
            4: 5.0,   # April
            5: 6.0,   # May
            6: 7.5,   # June
            7: 8.5,   # July
            8: 9.0,   # August
            9: 8.0,   # September
            10: 6.5,  # October
            11: 5.0,  # November
            12: 4.0,  # December
        }
        return seasonal_scores.get(month, 5.0)
    
    def _score_image_evidence(
        self,
        classification_result: Dict,
    ) -> float:
        """
        Score based on image classification results.
        
        Args:
            classification_result: Results from image classifier
            
        Returns:
            Risk score (0-10)
        """
        if not classification_result:
            return 0.0  # No image provided
            
        predicted_class = classification_result.get("predicted_class", "no_evidence")
        confidence = classification_result.get("confidence", 0.0)
        
        # Score based on what was detected
        class_scores = {
            "rat": 10.0,
            "droppings": 8.0,
            "burrow": 7.5,
            "gnaw_marks": 7.0,
            "no_evidence": 0.0,
        }
        
        base_score = class_scores.get(predicted_class, 0.0)
        
        # Weight by confidence
        return base_score * confidence
    
    def calculate_risk_score(
        self,
        historical_complaints: float = 0,
        predicted_complaints: float = None,
        restaurant_violations: float = 0,
        building_age: float = 50,
        month: int = None,
        image_result: Dict = None,
        return_factors: bool = False,
    ) -> Tuple[float, Optional[RiskFactors]]:
        """
        Calculate overall risk score.
        
        Args:
            historical_complaints: Historical complaint count
            predicted_complaints: Forecasted complaints
            restaurant_violations: Nearby restaurant violations
            building_age: Average building age in area
            month: Current month (1-12)
            image_result: Image classification result
            return_factors: Whether to return individual factor scores
            
        Returns:
            Risk score (1-10), optionally with factor breakdown
        """
        import datetime
        
        # Default month to current
        if month is None:
            month = datetime.datetime.now().month
            
        # Calculate individual factor scores
        factors = RiskFactors()
        
        factors.historical_complaints = self._score_historical_complaints(
            historical_complaints
        )
        
        if predicted_complaints is not None:
            factors.forecast_risk = self._score_forecast(predicted_complaints)
        else:
            factors.forecast_risk = factors.historical_complaints  # Fallback
            
        factors.restaurant_violations = self._score_restaurant_violations(
            restaurant_violations
        )
        
        factors.building_age = self._score_building_age(building_age)
        factors.seasonal_factor = self._score_seasonal(month)
        
        if image_result:
            factors.image_evidence = self._score_image_evidence(image_result)
            
        # Calculate weighted score
        factor_dict = factors.to_dict()
        
        score = 0.0
        total_weight = 0.0
        
        for factor_name, weight in self.weights.items():
            if factor_name in factor_dict:
                factor_value = factor_dict[factor_name]
                score += weight * factor_value
                total_weight += weight
                
        # Add image evidence as bonus if present (not in standard weights)
        if factors.image_evidence > 0:
            # Image evidence can increase score by up to 2 points
            score = min(10, score + factors.image_evidence * 0.2)
            
        # Ensure score is in valid range (1-10, not 0-10 for user-friendliness)
        final_score = np.clip(score, 1, 10)
        
        if return_factors:
            return final_score, factors
        return final_score, None
    
    def get_risk_level(self, score: float) -> str:
        """
        Convert numeric score to risk level string.
        
        Args:
            score: Risk score (1-10)
            
        Returns:
            Risk level string
        """
        if score < config.RISK_THRESHOLDS["low"]:
            return "Low"
        elif score < config.RISK_THRESHOLDS["medium"]:
            return "Low-Moderate"
        elif score < config.RISK_THRESHOLDS["high"]:
            return "Moderate"
        elif score < 9:
            return "High"
        else:
            return "Very High"
    
    def get_risk_color(self, score: float) -> str:
        """
        Get color code for risk level visualization.
        
        Args:
            score: Risk score (1-10)
            
        Returns:
            Hex color code
        """
        if score < 3:
            return "#28a745"  # Green
        elif score < 5:
            return "#7cb342"  # Light green
        elif score < 7:
            return "#ffc107"  # Yellow
        elif score < 8.5:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red
    
    def get_contributing_factors(
        self,
        factors: RiskFactors,
        threshold: float = 5.0,
    ) -> List[Dict]:
        """
        Get list of factors contributing most to risk.
        
        Args:
            factors: RiskFactors instance
            threshold: Minimum score to be considered contributing
            
        Returns:
            List of contributing factor dictionaries
        """
        factor_dict = factors.to_dict()
        
        contributing = []
        
        factor_descriptions = {
            "historical_complaints": "Historical complaint volume in the area",
            "forecast_risk": "Predicted future complaint activity",
            "restaurant_violations": "Nearby restaurant rodent violations",
            "building_age": "Age of buildings in the area",
            "nearby_construction": "Construction activity nearby",
            "seasonal_factor": "Seasonal activity patterns",
            "image_evidence": "Evidence detected in uploaded image",
        }
        
        for name, value in factor_dict.items():
            if value >= threshold:
                contributing.append({
                    "name": name,
                    "score": value,
                    "description": factor_descriptions.get(name, name),
                    "weight": self.weights.get(name, 0),
                })
                
        # Sort by weighted contribution
        contributing.sort(key=lambda x: x["score"] * x["weight"], reverse=True)
        
        return contributing
    
    def explain_score(
        self,
        score: float,
        factors: RiskFactors,
    ) -> str:
        """
        Generate a text explanation of the risk score.
        
        Args:
            score: Overall risk score
            factors: Individual factor scores
            
        Returns:
            Explanation string
        """
        risk_level = self.get_risk_level(score)
        contributing = self.get_contributing_factors(factors)
        
        explanation_parts = [
            f"**Risk Score: {score:.1f}/10 ({risk_level})**\n",
        ]
        
        if contributing:
            explanation_parts.append("\n**Key Contributing Factors:**\n")
            for factor in contributing[:3]:
                explanation_parts.append(
                    f"- {factor['description']}: {factor['score']:.1f}/10\n"
                )
        else:
            explanation_parts.append(
                "\nNo major risk factors identified.\n"
            )
            
        return "".join(explanation_parts)


class LocationRiskAssessor:
    """
    Complete risk assessment for a specific location.
    
    Combines data retrieval, scoring, and report generation.
    """
    
    def __init__(
        self,
        risk_scorer: RiskScorer = None,
        historical_data: pd.DataFrame = None,
    ):
        """
        Initialize the assessor.
        
        Args:
            risk_scorer: RiskScorer instance
            historical_data: Historical data for lookups
        """
        self.risk_scorer = risk_scorer or RiskScorer()
        self.historical_data = historical_data
        
    def assess_location(
        self,
        zip_code: str = None,
        borough: str = None,
        latitude: float = None,
        longitude: float = None,
        image_result: Dict = None,
        forecast_result: float = None,
    ) -> Dict:
        """
        Perform complete risk assessment for a location.
        
        Args:
            zip_code: ZIP code
            borough: Borough name
            latitude: Latitude
            longitude: Longitude
            image_result: Image classification results
            forecast_result: Forecasting model prediction
            
        Returns:
            Assessment dictionary
        """
        # Get historical data for location
        historical_stats = self._get_location_stats(zip_code, borough)
        
        # Calculate risk score
        score, factors = self.risk_scorer.calculate_risk_score(
            historical_complaints=historical_stats.get("total_complaints", 0),
            predicted_complaints=forecast_result,
            restaurant_violations=historical_stats.get("restaurant_violations", 0),
            building_age=historical_stats.get("building_age", 50),
            image_result=image_result,
            return_factors=True,
        )
        
        # Build assessment result
        assessment = {
            "risk_score": score,
            "risk_level": self.risk_scorer.get_risk_level(score),
            "risk_color": self.risk_scorer.get_risk_color(score),
            "factors": factors.to_dict() if factors else {},
            "contributing_factors": self.risk_scorer.get_contributing_factors(
                factors
            ) if factors else [],
            "historical_stats": historical_stats,
            "location": {
                "zip_code": zip_code,
                "borough": borough,
                "latitude": latitude,
                "longitude": longitude,
            },
            "explanation": self.risk_scorer.explain_score(score, factors) if factors else "",
        }
        
        return assessment
    
    def _get_location_stats(
        self,
        zip_code: str = None,
        borough: str = None,
    ) -> Dict:
        """Get historical statistics for a location."""
        if self.historical_data is None:
            return {
                "total_complaints": 0,
                "recent_complaints": 0,
                "restaurant_violations": 0,
                "building_age": 50,
            }
            
        # Filter by location
        df = self.historical_data
        
        if zip_code and "zip_code" in df.columns:
            location_df = df[df["zip_code"] == zip_code]
        elif borough and "borough" in df.columns:
            location_df = df[df["borough"] == borough]
        else:
            location_df = df
            
        if len(location_df) == 0:
            return {
                "total_complaints": 0,
                "recent_complaints": 0,
                "restaurant_violations": 0,
                "building_age": 50,
            }
            
        # Calculate stats
        stats = {
            "total_complaints": location_df["complaint_count"].sum() if "complaint_count" in location_df.columns else 0,
            "recent_complaints": location_df["complaint_count"].tail(6).sum() if "complaint_count" in location_df.columns else 0,
            "restaurant_violations": location_df["restaurant_violations_nearby"].mean() if "restaurant_violations_nearby" in location_df.columns else 0,
            "building_age": location_df["building_age_mean"].mean() if "building_age_mean" in location_df.columns else 50,
        }
        
        return stats


if __name__ == "__main__":
    # Test risk scoring
    logger.info("Testing risk scorer...")
    
    scorer = RiskScorer()
    
    # Test score calculation
    score, factors = scorer.calculate_risk_score(
        historical_complaints=25,
        predicted_complaints=8,
        restaurant_violations=3,
        building_age=65,
        month=7,  # July (peak season)
        return_factors=True,
    )
    
    print(f"Risk Score: {score:.1f}/10")
    print(f"Risk Level: {scorer.get_risk_level(score)}")
    print(f"\nFactors: {factors.to_dict()}")
    print(f"\n{scorer.explain_score(score, factors)}")
