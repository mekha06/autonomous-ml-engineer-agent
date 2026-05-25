from pydantic import BaseModel


class PredictionRequest(BaseModel):
    data: dict