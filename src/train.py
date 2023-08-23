import sys
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import pyarrow.parquet as pq

DATA_DIR = "../data/"
MODELS_DIR = "../models/"

def train_model(df):
    #Train-test-split
    X = df.drop("total_sales", axis=1)
    y = df["total_sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

    model = RandomForestRegressor(n_estimators=100, random_state=195)
    model.fit(X_train, y_train)
    
    return model

def save_model(model):
    file_path = MODELS_DIR + "model.pkl"

    # Save the model as a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    path_arg = sys.argv[-1]

    table = pq.read_table(path_arg)
    df = table.to_pandas()

    model = train_model(df)
    save_model(model)