# Student Placement Prediction System

An educational machine learning project that predicts student placement
status and compares four classification algorithms.

## Algorithms
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

## Features
- Individual student placement prediction
- Estimated class probabilities
- Four-model performance comparison
- Batch predictions from a CSV file
- Downloadable prediction results

## Technology
Python, pandas, NumPy, scikit-learn, joblib and Streamlit.

## Dataset
The project uses the dataset uploaded by the user in Google Colab.
Check the original dataset repository and license before redistribution.

If the selected dataset is synthetic, its results should be described as
synthetic-data results and not real-world placement performance.

## Run locally

Install dependencies:

    pip install -r requirements.txt

Start the application:

    streamlit run app.py

Keep app.py and model_bundle.joblib in the same directory.
