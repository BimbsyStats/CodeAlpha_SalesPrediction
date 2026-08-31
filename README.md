[README.md](https://github.com/user-attachments/files/31620415/README.md)
# CodeAlpha_SalesPrediction

Task 4 of the CodeAlpha Data Science Internship — predicting sales from
advertising spend across TV, Radio, and Newspaper channels.

## Dataset

`data/Advertising.csv` — 200 records of ad spend (in thousands of dollars)
per channel and the resulting sales (in thousands of units). No missing
values.

## Demo Video

[

![Watch the demo](https://img.youtube.com/vi/iaIH__Q2JCQ/0.jpg)

](https://youtu.be/iaIH__Q2JCQ)


## Approach

1. **Exploration** — summary statistics, correlation of each channel with
   sales, scatter plots with regression lines per channel, and a correlation
   heatmap.
2. **Modeling** — trained a Linear Regression model on an 80/20 train/test
   split using all three spend channels as features.
3. **Evaluation** — MAE, RMSE, R², and an actual-vs-predicted scatter plot.

## Results

- **R² = 0.899** — the model explains ~90% of the variance in sales
- **MAE = 1.46**, **RMSE = 1.78** (in thousands of units)

**Channel impact ranking** (regression coefficients, effect per $1,000 spend):
1. **Radio** — 0.189 (strongest impact)
2. **TV** — 0.045
3. **Newspaper** — 0.003 (negligible impact)

Despite TV having the highest raw correlation with sales (0.78 vs Radio's
0.58), Radio has a substantially larger *marginal* effect per dollar spent
once all three channels are considered together — this is a useful,
non-obvious insight for budget allocation. Newspaper spend shows almost no
measurable effect on sales.

## How to run

```
pip install pandas numpy matplotlib seaborn scikit-learn
python sales_prediction.py
```

Outputs (plots) are saved to `outputs/`.

## Project structure

```
.
├── README.md
├── sales_prediction.py
├── data/
│   └── Advertising.csv
└── outputs/
    ├── spend_vs_sales.png
    ├── correlation_heatmap.png
    └── actual_vs_predicted.png
```
