"""
MICAP FastAPI Main Application

Market Intelligence & Competitor Analysis Platform REST API
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables for ML models (will be loaded in lifespan)
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting MICAP API...")
    
    try:
        # Initialize ML models here
        logger.info("📦 Loading ML models...")
        # Note: In a real implementation, you would load your trained models here
        # ml_models["sentiment"] = load_sentiment_model()
        # ml_models["brand_recognition"] = load_brand_model()
        
        logger.info("✅ MICAP API startup complete")
        
    except Exception as e:
        logger.error(f"❌ Failed to start MICAP API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down MICAP API...")
    ml_models.clear()
    logger.info("✅ MICAP API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="MICAP API",
    description="Market Intelligence & Competitor Analysis Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment")
    models_loaded: Dict[str, bool] = Field(..., description="Model availability")


class TextAnalysisRequest(BaseModel):
    """Text analysis request model."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=10000)
    include_details: bool = Field(False, description="Include detailed analysis results")


class SentimentResponse(BaseModel):
    """Sentiment analysis response model."""
    text: str = Field(..., description="Original text")
    sentiment: str = Field(..., description="Predicted sentiment (positive/negative/neutral)")
    confidence: float = Field(..., description="Prediction confidence (0.0-1.0)")
    scores: Dict[str, float] = Field(..., description="Raw sentiment scores")
    details: Optional[Dict] = Field(None, description="Additional analysis details")


class BatchAnalysisRequest(BaseModel):
    """Batch analysis request model."""
    texts: List[str] = Field(..., description="List of texts to analyze", max_items=100)
    include_details: bool = Field(False, description="Include detailed analysis results")


class BatchAnalysisResponse(BaseModel):
    """Batch analysis response model."""
    results: List[SentimentResponse] = Field(..., description="Analysis results")
    summary: Dict[str, int] = Field(..., description="Summary statistics")


# API Endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {"message": "MICAP API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns system health status and model availability.
    """
    try:
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            environment=os.getenv("ENV", "development"),
            models_loaded={
                "sentiment": "sentiment" in ml_models,
                "brand_recognition": "brand_recognition" in ml_models,
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/analyze", response_model=SentimentResponse, tags=["Analysis"])
async def analyze_sentiment(request: TextAnalysisRequest) -> SentimentResponse:
    """
    Analyze sentiment of a single text.
    
    This endpoint performs sentiment analysis on the provided text using
    the loaded ML models.
    """
    try:
        # Mock implementation - replace with actual model inference
        logger.info(f"Analyzing text: {request.text[:50]}...")
        
        # Simple mock sentiment analysis
        text_lower = request.text.lower()
        if any(word in text_lower for word in ["good", "great", "excellent", "love", "amazing"]):
            sentiment = "positive"
            confidence = 0.85
            scores = {"positive": 0.85, "negative": 0.10, "neutral": 0.05}
        elif any(word in text_lower for word in ["bad", "terrible", "hate", "awful", "horrible"]):
            sentiment = "negative"
            confidence = 0.80
            scores = {"positive": 0.10, "negative": 0.80, "neutral": 0.10}
        else:
            sentiment = "neutral"
            confidence = 0.70
            scores = {"positive": 0.30, "negative": 0.25, "neutral": 0.45}
        
        details = None
        if request.include_details:
            details = {
                "word_count": len(request.text.split()),
                "char_count": len(request.text),
                "model_version": "mock-v1.0",
                "processing_time_ms": 50
            }
        
        return SentimentResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence,
            scores=scores,
            details=details
        )
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.post("/batch/analyze", response_model=BatchAnalysisResponse, tags=["Analysis"])
async def batch_analyze_sentiment(
    request: BatchAnalysisRequest, 
    background_tasks: BackgroundTasks
) -> BatchAnalysisResponse:
    """
    Analyze sentiment of multiple texts in batch.
    
    This endpoint performs batch sentiment analysis for improved efficiency
    when processing multiple texts.
    """
    try:
        logger.info(f"Batch analyzing {len(request.texts)} texts...")
        
        results = []
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for text in request.texts:
            # Reuse single analysis logic
            analysis_request = TextAnalysisRequest(
                text=text, 
                include_details=request.include_details
            )
            result = await analyze_sentiment(analysis_request)
            results.append(result)
            sentiment_counts[result.sentiment] += 1
        
        summary = {
            "total_analyzed": len(results),
            "positive": sentiment_counts["positive"],
            "negative": sentiment_counts["negative"],
            "neutral": sentiment_counts["neutral"]
        }
        
        return BatchAnalysisResponse(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Batch analysis failed")


@app.get("/trends/sentiment", tags=["Analytics"])
async def get_sentiment_trends(
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze")
) -> Dict:
    """
    Get sentiment trends over time.
    
    Returns aggregated sentiment trends for the specified time period.
    """
    try:
        logger.info(f"Getting sentiment trends for {days} days...")
        
        # Mock trend data - replace with actual database queries
        mock_trends = {
            "period_days": days,
            "trends": [
                {"date": "2024-01-01", "positive": 0.45, "negative": 0.35, "neutral": 0.20},
                {"date": "2024-01-02", "positive": 0.50, "negative": 0.30, "neutral": 0.20},
                {"date": "2024-01-03", "positive": 0.48, "negative": 0.32, "neutral": 0.20},
            ],
            "summary": {
                "avg_positive": 0.48,
                "avg_negative": 0.32,
                "avg_neutral": 0.20,
                "total_samples": 15000
            }
        }
        
        return mock_trends
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Trend analysis failed")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MICAP API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    logger.info(f"Starting MICAP API on {args.host}:{args.port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main() 