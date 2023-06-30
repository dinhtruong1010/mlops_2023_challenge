import argparse
import logging
import os
import random
import time
import numpy as np
import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request, BackgroundTasks
from pandas.util import hash_pandas_object
from pydantic import BaseModel
# import cv2
import json
from problem_config import ProblemConst, create_prob_config
from raw_data_processor import RawDataProcessor
from utils import AppConfig, AppPath
import asyncio
import concurrent.futures

PREDICTOR_API_PORT = 5040


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)
        # logging.info(self.category_index)
        # load model
        model_uri = os.path.join(
            "models:/", self.config["model_name"], str(self.config["model_version"])
        )
        # self.model = mlflow.pyfunc.load_model(model_uri)
        self.model = mlflow.sklearn.load_model(model_uri)

    def predict(self, data: Data):
        start_time = time.time()
        
        # # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )

        try:
            feature_df.drop(['is_drift', 'batch_id'], axis=1,inplace=True)
        except:
            pass

        pro_prediction = self.model.predict_proba(feature_df)

        prediction = np.argmax(pro_prediction, axis=1)
        is_drifted = 0 if sum(np.max(pro_prediction, axis=1))/ len(pro_prediction) > 0.7 else 1

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction takes {run_time} ms")

        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        name = str(filename)
        
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        feature_df.to_parquet(output_file_path, index=False)
        return output_file_path


default_config_path = "../data/model_config/phase-1/prob-1/model-1.yaml"
default_config_path2 = "../data/model_config/phase-1/prob-2/model-1.yaml"
# default_config_path = (
#     ".." + AppPath.MODEL_CONFIG_DIR
#     / ProblemConst.PHASE1
#     / ProblemConst.PROB1
#     / "model-1.yaml"
# ).as_posix()
# default_config_path2 = (
#     ".." + AppPath.MODEL_CONFIG_DIR
#     / ProblemConst.PHASE1
#     / ProblemConst.PROB2
#     / "model-1.yaml"
# ).as_posix()
predictor = ModelPredictor(config_file_path=default_config_path)
predictor2 = ModelPredictor(config_file_path=default_config_path2)
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "hello"}

@app.post("/phase-1/prob-1/predict")
async def predict(data: Data, request: Request, background_tasks:BackgroundTasks):
    _log_request(request)
    response = predictor.predict(data)
    _log_response(response)
    return response

@app.post("/phase-1/prob-2/predict")
async def predictP2(data: Data, request: Request):
    _log_request(request)
    response = predictor2.predict(data)
    _log_response(response)
    return response

@staticmethod
def _log_request(request: Request):
    pass

@staticmethod
def _log_response(response: dict):
    pass

def run(port):
    uvicorn.run("__main__:app", host="0.0.0.0", port=port, workers= 2)#, reload= True) 

if __name__ == "__main__":
    run(PREDICTOR_API_PORT)



