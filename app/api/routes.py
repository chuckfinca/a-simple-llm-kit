from fastapi import APIRouter, HTTPException, Request
from app.api.schemas import QueryRequest, QueryResponse
from app.services.prediction import PredictionService
from datetime import datetime

router = APIRouter()

@router.post("/predict", response_model=QueryResponse)
async def predict(request: Request, query: QueryRequest):
    try:
        prediction_service = PredictionService(request.app.state.model_manager)
        result = await prediction_service.predict(query)
        
        return QueryResponse(
            response=result,
            model_used=query.model_id,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "temperature": query.temperature,
                "max_tokens": query.max_tokens
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))