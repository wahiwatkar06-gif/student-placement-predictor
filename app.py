import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_bundle():
    return joblib.load("model_bundle.joblib")

bundle = load_bundle()

model = bundle["best_model"]
model_name = bundle["best_model_name"]
feature_columns = bundle["feature_columns"]
feature_config = bundle["feature_config"]
metrics = pd.DataFrame(bundle["metrics"])
positive_label = bundle["positive_label"]

st.title("🎓 Student Placement Prediction System")
st.write(
    "An ML application that compares four algorithms "
    "to estimate student placement outcomes."
)

st.info(
    "Academic demonstration only. The underlying dataset is synthetic. "
    "Predictions are not guarantees of actual employment and should "
    "not be used to make consequential decisions about students."
)

# Sidebar
st.sidebar.header("Project Information")
st.sidebar.write("Dataset: Indian Engineering Placement 2026")
st.sidebar.write(f"Records used: {bundle['dataset_rows']:,}")
st.sidebar.write(f"Input features: {len(feature_columns)}")
st.sidebar.write("Algorithms: 4")

tab1, tab2, tab3 = st.tabs([
    "Placement Prediction",
    "Model Comparison",
    "Bulk CSV Prediction"
])

with tab1:
    st.subheader("Enter Student Details")
    st.caption(
        "Enter the student's current academic and skill profile. "
        "Do not enter student IDs or salary information."
    )

    with st.form("placement_form"):
        inputs = {}
        input_columns = st.columns(2)

        for i, col in enumerate(feature_columns):
            config = feature_config[col]

            with input_columns[i % 2]:
                if config["type"] == "categorical":
                    options = config["options"]

                    if not options:
                        st.warning(f"No categories available for {col}")
                        inputs[col] = ""
                    else:
                        inputs[col] = st.selectbox(
                            col.replace("_", " "),
                            options=options,
                            key=f"input_{col}"
                        )
                else:
                    inputs[col] = st.number_input(
                        col.replace("_", " "),
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config["step"]),
                        key=f"input_{col}"
                    )

        submitted = st.form_submit_button(
            "Predict Placement",
            type="primary",
            use_container_width=True
        )

    if submitted:
        input_df = pd.DataFrame([inputs], columns=feature_columns)

        prediction = model.predict(input_df)[0]
        st.subheader("Prediction Result")

        if str(prediction).strip().lower() == str(positive_label).lower():
            st.success(f"Predicted outcome: {prediction}")
        else:
            st.warning(f"Predicted outcome: {prediction}")

        st.write(f"Model used: **{model_name}**")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.named_steps["classifier"].classes_)

            if positive_label in classes:
                index = classes.index(positive_label)
                probability = float(probabilities[index])

                st.metric(
                    "Estimated placement probability",
                    f"{probability:.1%}"
                )

                st.progress(probability)

                st.caption(
                    "This is a model estimate based on synthetic training "
                    "data, not a calibrated real-world employment probability."
                )

        with st.expander("View entered student details"):
            st.dataframe(input_df, use_container_width=True)

with tab2:
    st.subheader("Four-Algorithm Performance Comparison")

    display_metrics = metrics.set_index("Model")
    st.dataframe(
        display_metrics.style.format("{:.3f}"),
        use_container_width=True
    )

    chart_columns = [
        "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
    ]

    st.bar_chart(display_metrics[chart_columns])

    st.success(f"Top model by weighted F1-score: {model_name}")

    st.caption(
        "Scores were calculated on the held-out test set. "
        "Results may differ when evaluated on real student data."
    )

with tab3:
    st.subheader("Predict Multiple Students")
    st.write(
        "Upload a CSV containing the same input feature columns used "
        "during training. The app will ignore extra columns."
    )

    uploaded_file = st.file_uploader(
        "Upload student CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            missing_columns = [
                col for col in feature_columns
                if col not in batch_df.columns
            ]

            if missing_columns:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )
            elif batch_df.empty:
                st.warning("The uploaded CSV contains no rows.")
            else:
                batch_inputs = batch_df[feature_columns].copy()
                predictions = model.predict(batch_inputs)

                output_df = batch_df.copy()
                output_df["Predicted_Placement_Status"] = predictions

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(batch_inputs)
                    classes = list(
                        model.named_steps["classifier"].classes_
                    )

                    if positive_label in classes:
                        idx = classes.index(positive_label)
                        output_df["Estimated_Placement_Probability"] = (
                            probabilities[:, idx].round(4)
                        )

                st.write(f"Predictions generated: {len(output_df):,}")
                st.dataframe(output_df.head(100), use_container_width=True)

                st.download_button(
                    "Download prediction results",
                    data=output_df.to_csv(index=False).encode("utf-8"),
                    file_name="placement_predictions.csv",
                    mime="text/csv"
                )

        except Exception as exc:
            st.error(f"Could not process this file: {exc}")

st.divider()
st.caption(
    "Developed as a third-year engineering ML project. "
    "Dataset: Indian Engineering Students: Placement & Career Readiness "
    "Dataset (2026), by Vishwajeet Chandole."
)
