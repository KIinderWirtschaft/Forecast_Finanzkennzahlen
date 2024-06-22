import pandas as pd
import pickle
import sys
import argparse
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path
import numpy as np

# sklearn
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, PassiveAggressiveRegressor, TheilSenRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

def get_regressor(algorithm_name: str):
    regressors = {
        "RandomForest": RandomForestRegressor(random_state=123),
        "XGBoost": XGBRegressor(random_state=123),
        "GradientBoosting": GradientBoostingRegressor(random_state=123),
        "AdaBoost": AdaBoostRegressor(random_state=123),
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "KNeighbors": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(random_state=123),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "BayesianRidge": BayesianRidge(),
        "Huber": HuberRegressor(),
        "PassiveAggressive": PassiveAggressiveRegressor(),
        "TheilSen": TheilSenRegressor(),
        "Bagging": BaggingRegressor(random_state=123),
        "ExtraTrees": ExtraTreesRegressor(random_state=123),
        "KernelRidge": KernelRidge()
    }
    return regressors.get(algorithm_name, RandomForestRegressor(random_state=123))

def run(args, config: dict):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Run training on input csv")
    train_parser.add_argument("input", type=str, help="Input csv file")

    inference_parser = subparsers.add_parser(
        "infer", help="Run inference with new data"
    )
    inference_parser.add_argument("input", type=str, help="CSV file with test data")
    inference_parser.add_argument("model", type=str, help="Model as pkl")

    args = parser.parse_args(args)

    if args.command == "train":
        train(args.input, config)
    elif args.command == "infer":
        test(args.input, args.model, config)

def split_data(df: pd.DataFrame, steps: int):
    df_train = df[:-steps]
    df_test = df[-steps:]
    return df_train, df_test

def read_and_preprocess_csv(
    csv_path: Union[str, Path],
    date_col_name: str = "Datum ",
    delimiter: str = ";",
    interpolate: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, delimiter=delimiter)
    df["Datum "] = pd.to_datetime(df["Datum "], format="%Y-%m-%d")
    df = df.set_index(date_col_name)
    df = df.asfreq("MS")
    df = df.sort_index()

    if interpolate:
        for col in df.columns:
            df[col] = df[col].interpolate(method="linear")

    return df

def calculate_metrics(test: pd.Series, predicted: pd.Series):
    mae = np.mean(np.abs(test - predicted))
    mse = np.mean((test - predicted) ** 2)
    rmse = np.sqrt(mse)
    mfe = np.mean(test - predicted)
    return mae, mse, rmse, mfe

def train(csv_path: Union[str, Path], config: dict):
    df = read_and_preprocess_csv(
        csv_path,
        date_col_name=config["date_name"],
        interpolate=config["fill_missing_data"],
    )

    # Split data into train and test set
    data_train, data_test = split_data(df, steps=config["test_steps"])

    regressor = get_regressor(config["algorithm"])
    forecast_model = ForecasterAutoreg(
        regressor=regressor, lags=config["lags"]
    )
    forecast_model.fit(y=data_train["Menge "])

    # run a prediction
    predictions = forecast_model.predict(steps=config["test_steps"])

    # Calculate metrics
    mae, mse, rmse, mfe = calculate_metrics(data_test["Menge "], predictions)

    # Speichern der Ergebnisse in einer CSV-Datei
    results = pd.DataFrame({
        'Datum': data_test.index,
        'train': data_train["Menge "].reindex(data_test.index, method='ffill'),
        'test': data_test["Menge "],
        'predicted': predictions
    })
    results.to_csv("results.csv", index=False)

    plot_results(
        data_train, data_test, predictions, y_name=config["y_name"], plot_name="train",
        csv_path=csv_path, algorithm=config["algorithm"], mae=mae, mse=mse, rmse=rmse, mfe=mfe
    )

    # save model
    with open("model.pkl", "wb") as f:
        pickle.dump(forecast_model, f, pickle.HIGHEST_PROTOCOL)

def plot_results(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    predictions: pd.DataFrame,
    y_name: str,
    plot_name: str = "test",
    csv_path: Union[str, Path] = "",
    algorithm: str = "",
    mae: float = None,
    mse: float = None,
    rmse: float = None,
    mfe: float = None
):
    fig, ax = plt.subplots()
    data_train[y_name].plot(ax=ax, label="train", color="b")
    data_test[y_name].plot(ax=ax, label="test", color="r")
    predictions.plot(ax=ax, label="predicted", color="g")
    ax.legend()

    metrics_text = f"MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | MFE: {mfe:.2f}"
    plt.xlabel("Date")
    plt.ylabel(y_name)
    plt.figtext(0.5, -0.1, f"File: {csv_path} | Algorithm: {algorithm}\n{metrics_text}", wrap=True, horizontalalignment='center', fontsize=10)

    fig.savefig(f"{plot_name}.png", bbox_inches='tight')

def test(csv_path: Union[str, Path], model_path: Union[str, Path], config: dict):
    # read data
    df = read_and_preprocess_csv(
        csv_path,
        date_col_name=config["date_name"],
        interpolate=config["fill_missing_data"],
    )

    # read saved model
    with open(model_path, "rb") as m:
        forecast_model = pickle.load(m)

    # infer on new data
    data_past, data_future = split_data(df, steps=config["test_steps"])

    # predict new data
    predictions = forecast_model.predict(
        last_window=data_past[config["y_name"]], steps=config["test_steps"]
    )

    # Calculate metrics
    mae, mse, rmse, mfe = calculate_metrics(data_future["Menge "], predictions)

    # Speichern der Ergebnisse in einer CSV-Datei
    results = pd.DataFrame({
        'Datum': data_future.index,
        'train': data_past["Menge "].reindex(data_future.index, method='ffill'),
        'test': data_future["Menge "],
        'predicted': predictions
    })
    results.to_csv("results.csv", index=False)

    plot_results(
        data_past, data_future, predictions, y_name=config["y_name"], plot_name="test",
        csv_path=csv_path, algorithm=config["algorithm"], mae=mae, mse=mse, rmse=rmse, mfe=mfe
    )

if __name__ == "__main__":
    args = sys.argv[1:]

    config = {}
    config.update({"date_name": "Datum "})
    config.update({"y_name": "Menge "})
    config.update({"test_steps": 12})
    config.update({"lags": 12})
    config.update({"fill_missing_data": True})
    
    # Erklärungstext
    print("Bitte wählen Sie einen der folgenden Algorithmen und geben Sie ihn in der Konfiguration an:")
    print("RandomForest: Random Forest Regressor")
    print("XGBoost: XGBoost Regressor")
    print("GradientBoosting: Gradient Boosting Regressor")
    print("AdaBoost: AdaBoost Regressor")
    print("LinearRegression: Lineare Regression")
    print("SVR: Support Vector Regressor")
    print("KNeighbors: K-Nearest Neighbors Regressor")
    print("DecisionTree: Decision Tree Regressor")
    print("Ridge: Ridge Regressor")
    print("Lasso: Lasso Regressor")
    print("ElasticNet: ElasticNet Regressor")
    print("BayesianRidge: Bayesian Ridge Regressor")
    print("Huber: Huber Regressor")
    print("PassiveAggressive: Passive Aggressive Regressor")
    print("TheilSen: Theil Sen Regressor")
    print("Bagging: Bagging Regressor")
    print("ExtraTrees: Extra Trees Regressor")
    print("KernelRidge: Kernel Ridge Regressor")
    
    # Hier den gewünschten Algorithmus angeben
    config.update({"algorithm": "RandomForest"})  # Standardmäßig RandomForest; ändern Sie dies nach Bedarf

    run(args, config)
