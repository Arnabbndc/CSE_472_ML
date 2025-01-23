# Assignment 2: Logistic Regression with Bagging and Stacking

## Overview

This assignment focuses on implementing logistic regression models with bagging and stacking techniques to improve model performance. The goal is to build, evaluate, and compare the performance of these ensemble methods on a given dataset.


## Files

- **1905065.ipynb**: Jupyter notebook containing the implementation of logistic regression with bagging and stacking.

## Datasets

1. [**Telco Customer Churn Dataset**](https://www.kaggle.com/blastchar/telco-customer-churn):
    - The dataset contains information about customers of a telecommunications company and whether they churned or not.
    - Features include customer demographics, services subscribed, and monthly charges.
    - Target variable is 'Churn' indicating whether the customer churned or not.

2. [**Adult Income Dataset**](https://archive.ics.uci.edu/ml/datasets/adult):
    - The dataset contains information about individuals and their income levels.
    - Features include age, education, occupation, and work hours per week.
    - Target variable is 'Income' indicating whether the individual earns more than $50K or not.

3. [**Credit Card Fraud Detection Dataset**](https://www.kaggle.com/mlg-ulb/creditcardfraud):
    - The dataset contains transactions made by credit cards in September 2013 by European cardholders.
    - Features include time, amount, and anonymized transaction features.
    - Target variable is 'Class' indicating whether the transaction is fraudulent or not.

## Logistic Regression with Bagging

1. **Base Learners**:
    - Multiple logistic regression models are trained as base learners.
    - Each base learner is trained on a different bootstrap sample of the training data.

2. **Aggregation**:
    - The predictions from the base learners are aggregated (e.g., by averaging) to make the final prediction.

## Logistic Regression with Stacking

1. **Base Learners**:
    - Multiple logistic regression models are trained as base learners.
    - Each base learner is trained on the entire training dataset.

2. **Meta-Learner**:
    - A meta-learner (another logistic regression model) is trained on the predictions of the base learners.
    - The meta-learner combines the predictions of the base learners to make the final prediction.

## How to Run

Follow instructions in the [`1905065_results.pdf`](./1905065_results.pdf) file to run the code and visualize the results.

## Results

- The performance of the logistic regression models with bagging and stacking is evaluated using various metrics.
- Violin plots are generated to visualize the performance metrics of the base learners.

## Conclusion

This assignment demonstrates the implementation of logistic regression models with bagging and stacking techniques. The ensemble methods improve the performance of the logistic regression models, as shown by the evaluation metrics and visualizations.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---
