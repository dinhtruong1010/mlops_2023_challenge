import sys
from glob import glob
import pandas as pd
import argparse

# def show_parquet(path: str):
#     df = pd.read_parquet(path=path)
#     print(df)

def show_folder_parquet(args):
    path = input("path:")
    file_list = []
    if path.endswith(".parquet"):
        file_list.append(path)
    else: 
        file_list = glob(path + "/*.parquet")
    for i in file_list:
        print(i)
        df = pd.read_parquet(path=i)
        print(df)
        print(df["feature13"].nunique())
        print(df["feature12"].nunique())
        print(df["feature11"].nunique())
        print(df["feature1"].nunique())
        print(df["feature2"].nunique())
        # print("value:", df["is_drift"][1])
# Usage: python utils/show_parquet.py data/train_data/phase-1/prob-1/test_x.parquet
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="input")
    args = parser.parse_args()
    show_folder_parquet(args)
    exit()
    if len(sys.argv) >= 2:
        show_parquet(sys.argv[1])
    else:
        print("missing path")
        exit(1)
