# Best Buy Product Scraping, NLP, and Machine Learning

## Overview
This project scrapes Best Buy product pages, builds a structured dataset, and trains machine learning models to understand/predict price-related outcomes. The pipeline covers:
1) Web scraping and CSV consolidation
2) Feature engineering and EDA
3) Supervised learning with tree-based models

## Data Sources
Scraped Best Buy product/category pages and aggregated CSVs:
- BestBuy_Products1.csv
- BestBuy_Products2.csv
- BestBuy_ProductsTotal.csv
- Bestbuy_APIFY_P3.csv

(Your notebook shows both requests/BeautifulSoup scraping and an APIFY export.)

## Environment
Python 3.x

Main libraries:
- requests, bs4 (BeautifulSoup)
- pandas, numpy
- matplotlib
- statsmodels
- scikit-learn

## Project Structure
- BestbuyQ1-9.ipynb
  - Q1: Scraping and data assembly
  - Q2: Pearson correlation and VIF
  - Q3: Lasso regression (feature selection; analysis)
  - Q4: Decision Tree (classification/regression per target)
  - Q5: Random Forest (classification/regression)
  - Q6: SVM (analysis section present)
  - Q7: Neural network (analysis/diagram)
  - Q8: NN line diagram
  - Q9: Wrap-up answers

## Data Schema (observed)
Common columns used in modeling:
- Title (str)
- TitleID (str/int)
- Price (float)
- AverageReviewScore (float)
- NumberOfReviews (int)
- MonthlyPayment (float)
- Savings (target; numeric or binary depending on modeling choice)

Note: Ensure you define whether `Savings` is treated as a regression target (amount) or a classification target (has_savings yes/no).

## Web Scraping
- HTTP requests to category/product pages (`requests`)
- HTML parsing (`bs4.BeautifulSoup`)
- Extracted fields: product title/ID, price, review score, review count, monthly payment, and savings
- Wrote results to CSVs and merged into a master file

Tips:
- Respect robots.txt and rate limits
- Use robust selectors; handle missing fields and parsing errors gracefully

## EDA and Feature Engineering
- Basic cleaning (types, missing values)
- Pearson correlations
- Multicollinearity check via VIF
- Optional text features from `Title` (e.g., brand keywords, size/capacity tokens)
- Scaling/encoding only if the chosen model requires it

## Modeling
Primary models implemented and evaluated:
- Decision Tree
- Random Forest

Metrics:
- Accuracy (observed in code)
- Recommend adding: precision, recall, F1, ROC-AUC (for classification) or MAE/RMSE (for regression)

Train/test split:
- Use a fixed `random_state` for reproducibility
- Consider k-fold cross-validation when reporting final metrics

Class balance:
- If treating `Savings` as binary, check class distribution; consider stratified split

Feature importance:
- For tree-based models, report and plot feature importances (e.g., Price, AverageReviewScore, NumberOfReviews, MonthlyPayment)

## How to Run
1. Install requirements
   ```bash
   pip install -r requirements.txt
