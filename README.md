# Interactive Customer Segmentation

An interactive machine learning application for assigning new customer
profiles to learned customer segments based on patterns discovered from
transaction data.

## Overview

This project explores customer segmentation using unsupervised learning.
Customer segments were discovered using K-Means clustering, while the
Streamlit application allows users to enter a new customer profile and
predict its learned segment.

The application also compares the customer's numerical features with the
average values of the predicted segment.

## Features

- Customer segmentation using K-Means
- Silhouette score evaluation for comparing cluster configurations
- PCA visualization of discovered customer segments
- Interactive customer profile input
- Customer segment prediction
- Customer vs. segment average comparison

## Machine Learning Workflow

1. Clean and prepare the dataset
2. Remove irrelevant features
3. Encode categorical features
4. Scale the features
5. Compare different values of K using silhouette score
6. Train the K-Means clustering model
7. Visualize the discovered segments using PCA
8. Save the trained artifacts
9. Build an interactive Streamlit application for new customer profiles

## Application

The application provides:

- A customer profile form
- Predicted customer segment
- Number of records in the predicted segment
- Comparison between customer values and segment averages

The predicted segment represents the group whose learned characteristics
most closely match the submitted customer profile.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

## Project Structure

```text
.
├── app.py
├── customer_segmentation_artifacts.joblib
├── customer_segments.csv
├── interactive-customer-segmentation.ipynb
├── requirements.txt
└── .gitignore
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Context

This project was encountered through a course and independently
reimplemented to strengthen my understanding of unsupervised learning,
customer segmentation, and interactive machine learning applications.

## Note

The segments are data-driven clusters and do not automatically represent
predefined business categories. Further validation would be required before
using the segments for real-world business decisions.
