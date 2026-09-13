import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="centered"
)

ARTIFACT_FILE = "customer_segmentation_artifacts.joblib"
SEGMENT_FILE = "customer_segments.csv"

artifacts = joblib.load(ARTIFACT_FILE)
segmented = pd.read_csv(SEGMENT_FILE)

classifier = artifacts["classifier"]
numeric_columns = artifacts["numeric_columns"]
feature_columns = artifacts["feature_columns"]

st.title("👥 Interactive Customer Segmentation")
st.write(
    "Masukkan profil customer untuk melihat segment yang paling sesuai "
    "berdasarkan pola yang ditemukan dari data transaksi."
)

st.info(
    "Segment dihasilkan melalui unsupervised learning (K-Means). "
    "Label segment bukan label bisnis yang diberikan sebelumnya."
)

st.subheader("Customer Profile")

# Build input fields from the features available in the trained data.
input_data = {}

for col in feature_columns:
    if col in numeric_columns:
        default_value = float(segmented[col].median())
        input_data[col] = st.number_input(
            col,
            value=default_value
        )
    else:
        values = sorted(segmented[col].dropna().astype(str).unique().tolist())
        if values:
            input_data[col] = st.selectbox(col, values)
        else:
            input_data[col] = ""

if st.button("Predict Customer Segment", type="primary"):
    new_customer = pd.DataFrame([input_data])
    predicted_segment = int(classifier.predict(new_customer)[0])

    st.success(f"Customer berada pada **Segment {predicted_segment}**.")

    segment_rows = segmented[segmented["Segment"] == predicted_segment]

    st.subheader("Segment Overview")
    st.write(
        f"Segment {predicted_segment} berisi "
        f"**{len(segment_rows):,}** data pada dataset pelatihan."
    )

    available_numeric = [
        c for c in numeric_columns
        if c in segmented.columns
    ]

    if available_numeric:
        comparison = pd.DataFrame({
            "Feature": available_numeric,
            "Customer": [new_customer[c].iloc[0] for c in available_numeric],
            "Segment Average": [
                segment_rows[c].mean() for c in available_numeric
            ]
        })

        comparison["Difference"] = (
            comparison["Customer"] - comparison["Segment Average"]
        )

        st.dataframe(
            comparison.style.format({
                "Customer": "{:.2f}",
                "Segment Average": "{:.2f}",
                "Difference": "{:.2f}"
            }),
            use_container_width=True
        )

    st.caption(
        "Interpretation: the predicted segment represents the group whose "
        "learned characteristics most closely match this customer profile."
    )
