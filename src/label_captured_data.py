import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from problem_config import ProblemConfig, ProblemConst, get_prob_config
from utils import AppConfig, AppPath
import yaml
from problem_config import ProblemConst, create_prob_config
import mlflow
import os
from tqdm import tqdm
from raw_data_processor import RawDataProcessor
def label_captured_data(prob_config: ProblemConfig):
    train_x = pd.read_parquet(prob_config.train_x_path).to_numpy()
    train_y = pd.read_parquet(prob_config.train_y_path).to_numpy()
    ml_type = prob_config.ml_type

    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    for file_path in prob_config.captured_data_dir.glob("*.parquet"):
        captured_data = pd.read_parquet(file_path)
        captured_x = pd.concat([captured_x, captured_data])
    try:
        captured_x = captured_x.drop(['is_drift', 'batch_id'], axis=1)
    except:
        pass
    # captured_x = captured_x.iloc[:, : -2]
    np_captured_x = captured_x.to_numpy()
    
    n_captured = len(np_captured_x)
    logging.info(f"len input{n_captured}")
    n_samples = len(train_x) + n_captured
    logging.info(f"Loaded {n_captured} captured samples, {n_samples} train + captured")

    logging.info("Initialize and fit the clustering model")
    n_cluster = int(n_samples / 10) * len(np.unique(train_y))
    kmeans_model = MiniBatchKMeans(
        n_clusters=n_cluster, random_state=prob_config.random_state
    ).fit(train_x)

    logging.info("Predict the cluster assignments for the new data")
    kmeans_clusters = kmeans_model.predict(np_captured_x)

    logging.info(
        "Assign new labels to the new data based on the labels of the original data in each cluster"
    )
    new_labels = []
    for i in range(n_cluster):
        mask = kmeans_model.labels_ == i  # mask for data points in cluster i
        cluster_labels = train_y[mask]  # labels of data points in cluster i
        if len(cluster_labels) == 0:
            # If no data points in the cluster, assign a default label (e.g., 0)
            new_labels.append(0)
        else:
            # For a linear regression problem, use the mean of the labels as the new label
            # For a logistic regression problem, use the mode of the labels as the new label
            if ml_type == "regression":
                new_labels.append(np.mean(cluster_labels.flatten()))
            else:
                new_labels.append(
                    np.bincount(cluster_labels.flatten().astype(int)).argmax()
                )

    approx_label = [new_labels[c] for c in kmeans_clusters]
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])

    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)

def using_model_cluster(prob_config_path):
    
    with open(prob_config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"model-config: {config}")

    mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

    prob_config = create_prob_config(
        config["phase_id"], config["prob_id"]
    )

    # load category_index
    # category_index = RawDataProcessor.load_category_index(prob_config)

    # load model
    model_uri = os.path.join(
        "models:/", config["model_name"], str(config["model_version"])
    )
    model = mlflow.sklearn.load_model(model_uri)
    ######predict unlabeled data######
    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    thresh = 0.7
    approx_label = []
    for file_path in prob_config.captured_data_dir.glob("*.parquet"):
        print(file_path)
        captured_data = pd.read_parquet(file_path)
        try:
            captured_data.drop(['is_drift', 'batch_id'], axis=1,inplace=True)
        except:
            pass
        for i in tqdm(range(len(captured_data))):
            # print(captured_data.iloc[[i]])
            prob_predict = model.predict_proba(captured_data.iloc[[i]])
            
            if np.max(prob_predict) > thresh:
                # print(prob_predict)
                captured_x = pd.concat([captured_x, captured_data.iloc[[i]]])
                approx_label.append(np.argmax(prob_predict))
    
    
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])
    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)
        
def using_model_cluster_withDataCategory(prob_config_path):
    
    with open(prob_config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"model-config: {config}")

    mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

    prob_config = create_prob_config(
        config["phase_id"], config["prob_id"]
    )

    # load category_index
    category_index = RawDataProcessor.load_category_index(prob_config)

    # load model
    model_uri = os.path.join(
        "models:/", config["model_name"], str(config["model_version"])
    )
    model = mlflow.sklearn.load_model(model_uri)
    ######predict unlabeled data######
    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    thresh = 0.7
    approx_label = []
    sum = 0
    selected = 0
    for file_path in prob_config.captured_data_dir.glob("*.parquet"):
        print(file_path)
        captured_data = pd.read_parquet(file_path)
        # raw_df = pd.DataFrame(captured_data.rows, columns=captured_data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=captured_data,
            categorical_cols= prob_config.categorical_cols,
            category_index= category_index,
        )
        try:
            feature_df.drop(['is_drift', 'batch_id'], axis=1,inplace=True)
        except:
            pass
        for i in range(len(feature_df)):
            # print(captured_data.iloc[[i]])
            prob_predict = model.predict_proba(feature_df.iloc[[i]])
            sum += 1
            if np.max(prob_predict) > thresh:
                print(prob_predict)
                selected += 1
                captured_x = pd.concat([captured_x, feature_df.iloc[[i]]])
                approx_label.append(np.argmax(prob_predict))
    
    
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])
    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)
    print(selected, "---", sum)



if __name__ == "__main__":
    default_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE1
        / ProblemConst.PROB1
        / "model-1.yaml"
    ).as_posix()
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    using_model_cluster_withDataCategory(default_config_path)
    # label_captured_data(prob_config)
