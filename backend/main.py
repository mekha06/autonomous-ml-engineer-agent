from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import json
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

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        model_path = os.path.join(MODEL_DIR, "best_model.pkl")
        metadata_path = os.path.join(MODEL_DIR, "best_model_metadata.json")

        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail="No trained model found."
            )

        model = joblib.load(model_path)
        metadata = {}

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

        expected_features = metadata.get("feature_columns", [])

        if expected_features:
            missing_features = [
                feature
                for feature in expected_features
                if feature not in request.data
            ]

            unknown_features = [
                feature
                for feature in request.data
                if feature not in expected_features
            ]

            if missing_features:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "Prediction payload is missing required features.",
                        "missing_features": missing_features
                    }
                )

            if unknown_features:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "Prediction payload has unknown features.",
                        "unknown_features": unknown_features
                    }
                )

            prediction_data = {
                feature: request.data[feature]
                for feature in expected_features
            }
        else:
            prediction_data = request.data

        input_df = pd.DataFrame([prediction_data])

        prediction = model.predict(input_df)
        prediction_values = prediction.tolist()
        label_classes = metadata.get("label_classes")

        if label_classes:
            prediction_values = [
                label_classes[int(value)]
                for value in prediction_values
            ]

        return {
            "prediction": prediction_values,
            "model_metadata": {
                "model_name": metadata.get("best_model_name"),
                "task_type": metadata.get("task_type"),
                "target_column": metadata.get("target_column"),
                "best_score": metadata.get("best_score")
            }
        }

    except HTTPException:
        raise

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

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/download-report/{report_filename}")
def download_report(report_filename: str):
    try:
        report_path = os.path.join(REPORT_DIR, report_filename)
        media_type = "application/pdf"

        if report_filename.lower().endswith(".html"):
            media_type = "text/html"

        if not os.path.exists(report_path):
            raise HTTPException(
                status_code=404,
                detail="Report not found."
            )

        return FileResponse(
            path=report_path,
            filename=report_filename,
            media_type=media_type
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
