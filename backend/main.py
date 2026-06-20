from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import pandas as pd
import joblib

from config import UPLOAD_DIR, MODEL_DIR, REPORT_DIR
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


@app.get("/plot/{plot_name}")
def get_plot(plot_name: str):
    try:
        plot_path = os.path.join(REPORT_DIR, "plots", plot_name)

        if not os.path.exists(plot_path):
            raise HTTPException(
                status_code=404,
                detail="Plot not found."
            )

        return FileResponse(plot_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/download-report/{report_filename}")
def download_report(report_filename: str):
    try:
        report_path = os.path.join(REPORT_DIR, report_filename)

        if not os.path.exists(report_path):
            raise HTTPException(
                status_code=404,
                detail="Report not found."
            )

        return FileResponse(
            path=report_path,
            filename=report_filename,
            media_type="text/html"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))