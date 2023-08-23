import sys
import pandas as pd

MODEL_PATH = "../models/"
DATA_PATH = "../data/"

def read_path(path_model, path_data):
    model = path_model
    data = path_data
    if not MODEL_PATH in path_model:
        model = MODEL_PATH + path_model

    if not DATA_PATH in path_data:
        data = DATA_PATH + path_data
    return model, data


if __name__ == "__main__":
    path_models, path_data = read_path(sys.argv[-2], sys.argv[-1])
    pred_df = pd.read_parquet(path_data)
    model = pd.read_pickle(path_models)
    pred_tsales = model.predict(pred_df)
    pred_df["total_sales"] = pred_tsales
    pred_df.to_csv(DATA_PATH+"result.csv", index=False)