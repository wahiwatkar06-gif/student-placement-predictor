import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = "model_bundle.joblib"


@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model_bundle.joblib is missing. "
            "Upload it to the same GitHub repository as app.py."
        )

    return joblib.load(MODEL_PATH)


try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


model = bundle["best_model"]
model_name = bundle["best_model_name"]
feature_columns = bundle["feature_columns"]
feature_config = bundle["feature_config"]
metrics = pd.DataFrame(bundle["metrics"])
target_column = bundle["target_column"]
positive_label = bundle["positive_label"]


st.title("🎓 Student Placement Prediction System")
st.write(
    "An interactive machine learning application that predicts "
    "placement status using student academic and career-readiness features."
)

st.caption(f"Selected model: {model_name}")

st.info(
    "Academic demonstration only. If the training dataset is synthetic, "
    "predictions reflect patterns in that synthetic data and do not "
    "guarantee actual employment outcomes."
)


# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Home & Prediction",
        "Model Comparison",
        "Batch Prediction",
        "About Project"
    ]
)


# ---------------- HOME AND PREDICTION ----------------

if page == "Home & Prediction":

    st.header("Predict Individual Student Placement")
    st.write("Enter the student's details below.")

    with st.form("student_prediction_form"):

        inputs = {}

        left_col, right_col = st.columns(2)

        for index, feature in enumerate(feature_columns):

            current_col = (
                left_col if index % 2 == 0 else right_col
            )

            config = feature_config[feature]
            label = feature.replace("_", " ").title()

            with current_col:

                if config["type"] == "numeric":

                    inputs[feature] = st.number_input(
                        label,
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config["step"]),
                        key=f"individual_{feature}"
                    )

                else:
                    options = config.get("options", [])

                    if options:
                        inputs[feature] = st.selectbox(
                            label,
                            options=options,
                            key=f"individual_{feature}"
                        )
                    else:
                        inputs[feature] = st.text_input(
                            label,
                            key=f"individual_{feature}"
                        )

        submitted = st.form_submit_button(
            "Predict Placement",
            use_container_width=True
        )

    if submitted:

        try:
            student_df = pd.DataFrame(
                [[inputs[col] for col in feature_columns]],
                columns=feature_columns
            )

            prediction = model.predict(student_df)[0]

            st.divider()
            st.subheader("Prediction Result")

            if str(prediction).strip().lower() in [
                "placed", "yes", "selected", "true", "1"
            ]:
                st.success(f"Predicted placement status: {prediction}")

            elif str(prediction).strip().lower() in [
                "not placed", "not_placed", "unplaced",
                "no", "false", "0"
            ]:
                st.warning(f"Predicted placement status: {prediction}")

            else:
                st.success(f"Predicted class: {prediction}")

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(student_df)[0]
                classes = model.classes_

                probability_df = pd.DataFrame({
                    "Placement Class": classes,
                    "Estimated Probability (%)": probabilities * 100
                })

                st.subheader("Estimated Class Probabilities")

                st.dataframe(
                    probability_df.style.format({
                        "Estimated Probability (%)": "{:.2f}%"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                if positive_label in classes:
                    positive_index = list(classes).index(positive_label)

                    st.metric(
                        f"Estimated probability of '{positive_label}'",
                        f"{probabilities[positive_index] * 100:.2f}%"
                    )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ---------------- MODEL COMPARISON ----------------

elif page == "Model Comparison":

    st.header("Compare ML Algorithms")

    st.write(
        "The models were evaluated on the same held-out test set. "
        "The selected model was chosen using weighted F1-score."
    )

    st.subheader("Performance Metrics")

    if not metrics.empty:

        st.dataframe(
            metrics.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "ROC-AUC": lambda value: (
                    "N/A" if pd.isna(value) else f"{value:.4f}"
                )
            }),
            use_container_width=True,
            hide_index=True
        )

        chart_columns = [
            col for col in [
                "Accuracy", "Precision", "Recall",
                "F1 Score", "ROC-AUC"
            ]
            if col in metrics.columns
        ]

        if chart_columns:
            chart_data = metrics.set_index("Model")[chart_columns]
            st.bar_chart(chart_data)

        st.success(f"Best model by weighted F1-score: {model_name}")

    else:
        st.warning("Model evaluation metrics are not available.")


# ---------------- BATCH PREDICTION ----------------

elif page == "Batch Prediction":

    st.header("Batch Student Placement Prediction")

    st.write(
        "Upload a CSV containing the same input feature columns "
        "used during model training. The CSV must not need the target "
        "column to generate predictions."
    )

    st.caption(
        "For best results, use a CSV with the original feature column "
        "names and compatible values. Extra columns will be ignored."
    )

    uploaded_file = st.file_uploader(
        "Upload student CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            missing_columns = [
                col for col in feature_columns
                if col not in batch_df.columns
            ]

            if missing_columns:
                st.error(
                    "The uploaded CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )

            elif st.button("Predict All Students"):

                input_df = batch_df[feature_columns].copy()

                for col in feature_columns:
                    if feature_config[col]["type"] == "categorical":
                        input_df[col] = input_df[col].map(
                            lambda value: (
                                str(value) if pd.notna(value) else None
                            )
                        )

                    else:
                        input_df[col] = pd.to_numeric(
                            input_df[col],
                            errors="coerce"
                        )

                predictions = model.predict(input_df)

                output_df = batch_df.copy()
                output_df["Predicted_" + target_column] = predictions

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(input_df)
                    classes = list(model.classes_)

                    for i, class_name in enumerate(classes):
                        safe_name = str(class_name).replace(" ", "_")

                        output_df[
                            f"Probability_{safe_name}_Percent"
                        ] = (probabilities[:, i] * 100).round(2)

                st.subheader("Prediction Results")
                st.dataframe(output_df, use_container_width=True)

                csv_data = output_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Predictions CSV",
                    data=csv_data,
                    file_name="placement_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.header("About the Project")

    st.markdown("""
    **Project:** Student Placement Prediction System

    **Objective:** Estimate student placement status from available
    academic and career-readiness features using machine learning.

    **Algorithms compared:**
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Gradient Boosting

    **Technology stack:**
    Python, pandas, NumPy, scikit-learn, joblib and Streamlit.

    **Limitations:**
    - Predictions depend on the dataset and its quality.
    - Synthetic data cannot establish real-world placement performance.
    - Estimated probabilities are not guarantees of employment.
    - The tool is intended for educational demonstration, not consequential
      decisions about individual students.
    """)

    st.write("Target variable:", target_column)
    st.write("Saved model:", model_name)


# ---------------- FOOTER ----------------

st.divider()
st.caption(
    "Student Placement Prediction System | "
    "Machine Learning Academic Project"
)
