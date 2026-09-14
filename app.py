import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)


# =========================================================
# LOAD ARTIFACTS
# =========================================================

ARTIFACT_FILE = "customer_segmentation_artifacts.joblib"
SEGMENT_FILE = "customer_segments.csv"

artifacts = joblib.load(ARTIFACT_FILE)
segmented = pd.read_csv(SEGMENT_FILE)

classifier = artifacts["classifier"]
numeric_columns = artifacts["numeric_columns"]
feature_columns = artifacts["feature_columns"]


# =========================================================
# HEADER
# =========================================================

st.title("👥 Interactive Customer Segmentation")

st.write(
    "Masukkan profil customer untuk melihat segment yang paling sesuai "
    "berdasarkan pola yang ditemukan dari data transaksi."
)

st.info(
    "Segment dihasilkan melalui unsupervised learning (K-Means). "
    "Label segment bukan label bisnis yang diberikan sebelumnya."
)


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.subheader("Customer Profile")

input_data = {}

# Use 3 columns to make the profile more compact.
columns = st.columns(3, gap="small")

for i, col in enumerate(feature_columns):

    current_col = columns[i % 3]

    with current_col:

        if col in numeric_columns:

            default_value = float(
                segmented[col].median()
            )

            input_data[col] = st.number_input(
                col,
                value=default_value
            )

        else:

            values = sorted(
                segmented[col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if values:

                input_data[col] = st.selectbox(
                    col,
                    values
                )

            else:

                input_data[col] = ""


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

if st.button(
    "Predict Customer Segment",
    type="primary",
    use_container_width=True
):

    # Convert input into dataframe
    new_customer = pd.DataFrame([input_data])

    # Predict segment
    predicted_segment = int(
        classifier.predict(new_customer)[0]
    )

    # Show prediction
    st.success(
        f"Customer berada pada **Segment {predicted_segment}**."
    )

    # =====================================================
    # SEGMENT OVERVIEW
    # =====================================================

    segment_rows = segmented[
        segmented["Segment"] == predicted_segment
    ]

    st.subheader("Segment Overview")

    st.write(
        f"Segment {predicted_segment} berisi "
        f"**{len(segment_rows):,}** data pada dataset pelatihan."
    )

    # =====================================================
    # CUSTOMER VS SEGMENT AVERAGE
    # =====================================================

    available_numeric = [
        c
        for c in numeric_columns
        if c in segmented.columns
    ]

    if available_numeric:

        comparison = pd.DataFrame({
            "Feature": available_numeric,

            "Customer": [
                new_customer[c].iloc[0]
                for c in available_numeric
            ],

            "Segment Average": [
                segment_rows[c].mean()
                for c in available_numeric
            ]
        })

        comparison["Difference"] = (
            comparison["Customer"]
            - comparison["Segment Average"]
        )

        st.dataframe(
            comparison.style.format({
                "Customer": "{:.2f}",
                "Segment Average": "{:.2f}",
                "Difference": "{:.2f}"
            }),
            use_container_width=True
        )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.caption(
        "Interpretation: the predicted segment represents the group "
        "whose learned characteristics most closely match this "
        "customer profile."
    )