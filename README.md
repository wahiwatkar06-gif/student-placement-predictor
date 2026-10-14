# Student Placement Prediction System

An educational machine learning project that compares four
classification algorithms for student placement prediction.

## Dataset

Indian Engineering Students: Placement & Career Readiness Dataset (2026)

- 15,000 synthetic records
- 25 original columns
- Source: https://github.com/vishwajeetchandole/indian-engineering-placement-dataset
- Dataset license: CC BY 4.0
- Note: The dataset is synthetic, not real student placement records.

## Algorithms

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

## Features

- Student placement prediction
- Estimated placement probability
- Four-model comparison dashboard
- Batch predictions using CSV uploads
- Downloadable prediction results

## Technology Stack

Python, pandas, NumPy, scikit-learn, joblib and Streamlit.

## Run Locally

Install dependencies:

    pip install -r requirements.txt

Start the application:

    streamlit run app.py

Make sure model_bundle.joblib is in the same folder as app.py.

## Deployment

Deploy this GitHub repository using Streamlit Community Cloud.
Set the main file path to app.py.

## Disclaimer

This is an educational demonstration trained on synthetic data.
Predictions are not guarantees of real-world employment outcomes.

## Dataset Attribution

Vishwajeet Chandole, Indian Engineering Students: Placement &
Career Readiness Dataset (2026).

Dataset source:
https://github.com/vishwajeetchandole/indian-engineering-placement-dataset
