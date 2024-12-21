from app.models.predictor import Predictor
from app.api.schemas import QueryRequest
import dspy

class PredictionService:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    async def predict(self, request: QueryRequest) -> str:
        with self.model_manager.get_model(request.model_id) as lm:
            dspy.configure(lm=lm)
            predictor = dspy.Predict(Predictor, lm)
            result = predictor(input=request.prompt)
            return result.output