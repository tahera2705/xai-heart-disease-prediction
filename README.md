# Heart Disease Prediction Project

## Project Overview

This project focuses on developing and evaluating machine learning models to predict heart disease using a comprehensive dataset. The goal is to identify individuals at risk of heart disease based on various clinical parameters, ultimately aiding in early diagnosis and intervention.

## Dataset

The project utilizes the `heart.csv` dataset, which contains [brief description of dataset, e.g., 'patient health records with various attributes and a binary target variable indicating the presence or absence of heart disease.']. The dataset was preprocessed to handle categorical features, scale numerical data, and split into training and testing sets.

## Project Structure

The notebook follows a structured approach:

1.  **Data Loading and Initial Exploration:** Loading the dataset and performing initial checks.
2.  **Data Preprocessing:** Handling missing values, one-hot encoding categorical features, and scaling numerical features.
3.  **Exploratory Data Analysis (EDA):** Visualizing feature distributions and correlations to gain insights into the data.
4.  **Model Development & Evaluation:** Implementing and evaluating several classification models:
    *   Logistic Regression
    *   Support Vector Machine (SVM)
    *   XGBoost Classifier
    *   Random Forest Classifier
5.  **Hyperparameter Tuning:** Utilizing `GridSearchCV` with k-fold cross-validation to optimize hyperparameters for each model.
6.  **Ensemble Methods:** Building and evaluating advanced ensemble techniques:
    *   Hybrid models combining predictions.
    *   Stacking Ensembles with optimized base learners and a meta-learner (optimized XGBoost).
7.  **Feature Engineering & Selection:** Creating new features and selecting the most impactful ones using `SelectKBest`.
8.  **Model Interpretability:** Applying SHAP (SHapley Additive exPlanations) for global feature importance and LIME (Local Interpretable Model-agnostic Explanations) for local, instance-level explanations of model predictions.

## Key Features and Techniques

*   **Machine Learning Models:** Logistic Regression, SVM, XGBoost, Random Forest.
*   **Ensemble Learning:** Hybrid predictions, Stacking Classifiers.
*   **Hyperparameter Optimization:** `GridSearchCV` with `KFold` cross-validation.
*   **Feature Engineering:** Creation of interaction terms and polynomial features.
*   **Feature Selection:** `SelectKBest` with `f_classif`.
*   **Interpretability:** SHAP and LIME for understanding model decisions.
*   **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score.

## Results Summary

The project compares the performance of various optimized models. The **Final Hybrid (LR + SVM)** model achieved a strong F1 Score of **0.917** on the test set, indicating robust performance in predicting heart disease. The **New Stacking Ensemble (XGBoost Meta-Learner) with Selected Features** also showed competitive performance with an F1 score of **0.911**.

| Model                               | Metric    | Accuracy | Precision | Recall   | F1 Score |
| :---------------------------------- | :-------- | :------- | :-------- | :------- | :------- |
| Optimized XGBoost                   | Test      | 0.875    | 0.891     | 0.882    | 0.887    |
| Optimized Random Forest             | Test      | 0.880    | 0.870     | 0.922    | 0.895    |
| Optimized Logistic Regression       | Test      | 0.880    | 0.877     | 0.912    | 0.894    |
| Optimized LR (Threshold)            | Test      | 0.902    | 0.912     | 0.912    | 0.912    |
| Optimized SVM                       | Test      | 0.897    | 0.867     | 0.961    | 0.912    |
| Final Hybrid (LR + SVM)             | Test      | 0.908    | 0.913     | 0.922    | **0.917**|
| Stacking Ensemble                   | Test      | 0.891    | 0.894     | 0.912    | 0.903    |
| Optimized Stacking Ensemble         | Test      | 0.886    | 0.901     | 0.892    | 0.897    |
| New Stacking Ensemble (XGBoost Meta-Learner) | Test      | 0.897    | 0.882     | 0.941    | 0.911    |

## Setup and Usage

To run this notebook locally, ensure you have the following libraries installed:

```bash
pip install pandas scikit-learn xgboost matplotlib seaborn keras tensorflow lime shap
Then, you can open and run the Jupyter Notebook ([Your_Notebook_Name].ipynb) to reproduce the analysis and model training.

Future Work
[Suggest potential improvements, e.g., 'Exploring additional feature engineering techniques.']
[Suggest exploring other advanced models or ensemble strategies, e.g., 'Investigating deep learning architectures like LSTMs for time-series data if applicable.']
[Suggest more extensive hyperparameter tuning or different cross-validation strategies, e.g., 'Implementing more robust cross-validation methods, such as nested cross-validation.']
Author
[Your Name/GitHub Username]

License
[e.g., 'This project is licensed under the MIT License - see the LICENSE.md file for details.'] ```# xai-heart-disease-prediction
Explainable AI project using ML models and SHAP/LIME for heart disease prediction
