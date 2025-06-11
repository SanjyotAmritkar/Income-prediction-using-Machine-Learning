# Income Prediction using Machine Learning

This project applies a suite of machine learning models to predict whether an individual earns more than \$50K per year based on demographic and employment-related features from the UCI Adult Census dataset. It was developed as a course project for AMS 580: Statistical Learning at Stony Brook University (Spring 2025).

## Project Overview

The goal is to build and evaluate classification models that categorize individuals into income groups (`<=50K` or `>50K`) based on attributes like age, education, occupation, hours-per-week, etc. The project emphasizes rigorous data preprocessing, feature selection, model tuning, and performance evaluation.

## Dataset

- Source: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- Features: 14 predictors including both numerical and categorical variables.
- Target: Binary classification – income (`<=50K`, `>50K`)
- Files: `train.csv`, `test.csv`

## Preprocessing Steps

- Handled missing values (`?`) by imputing with mode.
- Label encoded categorical variables for model compatibility.
- Scaled numerical features using `StandardScaler`.
- Balanced classes using `RandomOverSampler`.
- Applied log transformation on skewed variables like `age`, `fnlwgt`.

## Exploratory Data Analysis

- Countplots, histograms, KDEs, and boxplots for feature distribution.
- Visual analysis of income trends across age, education, occupation, workclass, and native country.
- Correlation matrix to examine multicollinearity.
- Insights:
  - Age 30–60 most likely to earn >\$50K.
  - Higher income correlated with advanced education and managerial roles.

## Models Trained

| Model            | Accuracy | Sensitivity | Specificity | AUC Score |
|------------------|----------|-------------|-------------|-----------|
| Logistic Regression | 77.5%   | 78.8%       | 76.2%       | 0.8575    |
| K-Nearest Neighbors | 84.0%   | 89.5%       | 78.6%       | 0.9112    |
| Decision Tree      | 91.6%   | 97.0%       | 86.3%       | 0.9165    |
| **Random Forest**  | **92.8%** | **97.6%**   | **88.0%**   | **0.9844**|
| XGBoost            | 87.3%   | 91.2%       | 83.4%       | 0.9478    |
| AdaBoost           | 83.1%   | 85.3%       | 81.0%       | 0.9148    |
| Gradient Boosting  | 83.7%   | 86.8%       | 80.7%       | 0.9220    |

**Final Model Selected**: Random Forest – based on its high test accuracy, strong recall, and robustness to noise and imbalance.

## Feature Selection & Tuning

- Used Forward Feature Selection based on AIC to reduce dimensionality.
- Hyperparameter tuning performed using `RandomizedSearchCV`.
- 10-fold cross-validation ensured generalizability.

## Technologies & Libraries

- Python 3.11
- `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`
- `imbalanced-learn`, `xgboost`, `statsmodels`
- `RandomizedSearchCV`, `LabelEncoder`, `StandardScaler`

## Key Takeaways

- Feature engineering and class balancing significantly impact model performance.
- Ensemble models like Random Forest and XGBoost outperform simpler models.
- Hyperparameter tuning and feature selection improve generalization.

## License

This project is for academic purposes only under the AMS 580 course at Stony Brook University.

---


