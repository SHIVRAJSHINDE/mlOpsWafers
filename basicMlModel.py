import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import argparse


def get_data():
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e


def evaluate(y_true, y_pred, pred_prob):
    accuracy = accuracy_score(y_true, y_pred)
    rc_score = roc_auc_score(y_true, pred_prob, multi_class='ovr')
    return accuracy, rc_score


def main(n_estimators, max_depth):
    df = get_data()
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    X_train, X_test = train.drop("quality", axis=1), test.drop("quality", axis=1)
    y_train, y_test = train["quality"], test["quality"]

    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        pred_prob = rf.predict_proba(X_test)

        accuracy, rc_score = evaluate(y_test, pred, pred_prob)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc_score", rc_score)

        mlflow.sklearn.log_model(rf, "randomforestmodel")

        print(f"Accuracy: {accuracy}, ROC AUC Score: {rc_score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a RandomForestClassifier on wine quality data.")
    parser.add_argument("--n_estimators", "-n", default=50, type=int, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", "-m", default=5, type=int, help="Maximum depth of the trees.")

    parse_args = parser.parse_args()

    try:
        main(n_estimators=parse_args.n_estimators, max_depth=parse_args.max_depth)
    except Exception as e:
        raise e
