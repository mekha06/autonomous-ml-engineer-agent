from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import shutil
import os
import pandas as pd
import joblib

from config import UPLOAD_DIR, MODEL_DIR
from pipelines.automl_pipeline import AutoMLPipeline
from schemas.prediction_schema import PredictionRequest

app = FastAPI(title="Autonomous ML Engineer Agent API")

pipeline = AutoMLPipeline()


@app.get("/")
def home():
    return {"message": "Autonomous ML Engineer Agent API Running"}


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = pipeline.run(
            file_path=file_path,
            target_column=target_column
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        model_path = os.path.join(MODEL_DIR, "best_model.pkl")

        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail="No trained model found."
            )

        model = joblib.load(model_path)

        input_df = pd.DataFrame([request.data])

        prediction = model.predict(input_df)

        return {
            "prediction": prediction.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))