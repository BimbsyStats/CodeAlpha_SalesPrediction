"""
CodeAlpha Data Science Internship — Task 4: Sales Prediction using Python

Predicts sales based on advertising spend across TV, Radio, and Newspaper
channels using Linear Regression, and analyzes which channel drives the
most impact.

Run from the project root:
    python sales_prediction.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join("data", "Advertising.csv")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def explore_data(df):
    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nSummary statistics:\n{df.describe()}")

    # Correlation with sales
    corr = df.corr()["Sales"].drop("Sales").sort_values(ascending=False)
    print(f"\nCorrelation with Sales:\n{corr}")

    # Scatter plots: spend vs sales for each channel
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["TV", "Radio", "Newspaper"]):
        sns.regplot(data=df, x=col, y="Sales", ax=ax, scatter_kws={"alpha": 0.5})
        ax.set_title(f"{col} Spend vs Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spend_vs_sales.png"), dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    print(f"\nSaved spend_vs_sales.png and correlation_heatmap.png to {OUT_DIR}/")


def train_and_evaluate(df):
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION (Linear Regression)")
    print("=" * 60)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")

    print("\nFeature coefficients (impact per $1000 spend):")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    # Actual vs predicted plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#3FB6A8")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    print(f"\nSaved actual_vs_predicted.png to {OUT_DIR}/")

    # Rank channels by impact
    impact = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
    print(f"\nChannels ranked by sales impact (coefficient):\n{impact}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    df = load_data()
    explore_data(df)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
