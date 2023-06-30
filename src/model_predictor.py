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
        logging.info(self.category_index)
        # load model
        model_uri = os.path.join(
            "models:/", self.config["model_name"], str(self.config["model_version"])
        )
        # self.model = mlflow.pyfunc.load_model(model_uri)
        self.model = mlflow.sklearn.load_model(model_uri)
    
    def detect_drift2(self, pro_prediction) -> int:
        # watch drift between coming requests and training data
        sum = 0
        for i in range(len(pro_prediction)):
            if pro_prediction[i][0] > pro_prediction[i][1]:
                sum += pro_prediction[i][0]
            else:
                sum += pro_prediction[i][1]
             
        if sum/len(pro_prediction) > 0.7:
            return 0
        else:
            return 1
    def detect_drift(self, feature_df):
        time.sleep(0.02)
        return random.choice([0, 1])

    def predict(self, data: Data):
        start_time = time.time()
        
        # # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        # save request data for improving models
        # ModelPredictor.save_request_data(
        #     raw_df, self.prob_config.captured_data_dir, data.id
        # )
        try:
            feature_df.drop(['is_drift', 'batch_id'], axis=1,inplace=True)
        except:
            pass
        # logging.info(f"{feature_df}")
        ##############################################DETECT NORMAL############################################
        # prediction = self.model.predict(feature_df)
        # is_drifted = self.detect_drift(feature_df)
        ##############################################DETECT PROBA#############################################
        
        pro_prediction = self.model.predict_proba(feature_df)
        
        # logging.info(f"{pro_prediction}")
        #######################################GET PREDICTION AND DRIFT########################################
        # prdtime = time.time()
        # prediction = []
        # for i in range(len(pro_prediction)):
        #     prediction.append(np.argmax(pro_prediction[i]))
        # prediction = np.array(prediction)
        # is_drifted = self.detect_drift2(pro_prediction)
        # run_time = round((time.time() - prdtime) * 1000, 0)
        # logging.info(f"infrence1 {run_time} ms")
        
        # prdtime = time.time()
        prediction = np.argmax(pro_prediction, axis=1)
        is_drifted = 0 if sum(np.max(pro_prediction, axis=1))/ len(pro_prediction) > 0.7 else 1
        # run_time = round((time.time() - prdtime) * 1000, 0)
        # logging.info(f"infrence2 {run_time} ms")
        # logging.info(f"{prediction}")

        #######################################SAVE RESULT TO JSON FILE#########################################
        # datab = {}
        # datab['id'] = data.id
        # datab['predictions'] = prediction.tolist()
        # datab['drift'] = is_drifted
        # filename = hash_pandas_object(feature_df).sum()
        # with open("/Volumes/DATA/Works/mlopsvn/mlops-mara-sample-public-main/data/output2/" + str(filename) +".json", 'w') as f:
        #     json.dump(datab, f)
        ##################################################
        ###########
        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction takes {run_time} ms")
        # logging.info(f"id:{data.id}predictions{prediction.tolist()}drift{is_drifted}")
        # filename = hash_pandas_object(feature_df).sum()
        # with open("/Volumes/DATA/Works/mlopsvn/mlops-mara-sample-public-main/data/output/" + str(filename) +".json") as f:
        #     return json.load(f)
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


class PredictorApi:
    def __init__(self, predictor: ModelPredictor, predictor2: ModelPredictor):
        self.predictor = predictor
        self.predictor2 = predictor2
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/phase-1/prob-1/predict")
        async def predict(data: Data, request: Request, background_tasks:BackgroundTasks):
            self._log_request(request)
            # loop = asyncio.get_running_loop()
            # with concurrent.futures.ProcessPoolExecutor() as pool:
            #     response = await loop.run_in_executor(pool, self.predictor.predict(data))
            response = self.predictor.predict(data)
            # background_tasks.add_task()
            self._log_response(response)
            return response

        @self.app.post("/phase-1/prob-2/predict")
        async def predictP2(data: Data, request: Request):
            self._log_request(request)
            # loop = asyncio.get_running_loop()
            # with concurrent.futures.ProcessPoolExecutor() as pool:
            #     response = await loop.run_in_executor(pool, self.predictor2.predict(data))
            response = self.predictor2.predict(data)
            self._log_response(response)
            return response

    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port) 


if __name__ == "__main__":
    default_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE1
        / ProblemConst.PROB1
        / "model-1.yaml"
    ).as_posix()
    default_config_path2 = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE1
        / ProblemConst.PROB2
        / "model-1.yaml"
    ).as_posix()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default=default_config_path)
    parser.add_argument("--config-path2", type=str, default=default_config_path2)
    parser.add_argument("--port", type=int, default=PREDICTOR_API_PORT)
    args = parser.parse_args()

    predictor = ModelPredictor(config_file_path=args.config_path)
    predictor2 = ModelPredictor(config_file_path=args.config_path2)
    # api = asyncio.run(PredictorApi(predictor, predictor2))
    # asyncio.run(api.run(port=args.port))
    api = PredictorApi(predictor, predictor2)
    api.run(port=args.port)