🛡️ Fraud Detection for E-commerce and Bank Transactions
Project Overview

Fraud detection is a critical challenge in financial technology due to the high cost of missed fraud and the customer friction caused by false positives.
This project aims to build robust fraud detection pipelines for:

E-commerce transactions

Bank credit card transactions

using data analysis, feature engineering, and machine learning techniques that are well-suited for highly imbalanced datasets.

This repository currently contains the work completed for Interim-1, which focuses on data understanding, preprocessing, exploratory data analysis (EDA), geolocation integration, and feature engineering.

Business Objective

Adey Innovations Inc. seeks to improve fraud detection accuracy while maintaining a balance between:

Security (catching fraudulent transactions)

User experience (avoiding unnecessary transaction blocks)

This requires:

Careful feature engineering

Proper handling of class imbalance

Interpretable and explainable modeling approaches (to be developed in later stages)

Datasets Used
1. E-commerce Fraud Dataset (Fraud_Data.csv)

Contains transaction-level and user-level information:

User details: user_id, age, sex

Transaction details: purchase_value, device_id, browser, source

Time features: signup_time, purchase_time

Network feature: ip_address

Target variable: class

1 → Fraudulent transaction

0 → Legitimate transaction

2. IP Address to Country Mapping (IpAddress_to_Country.csv)

Maps ranges of IP addresses to countries using:

lower_bound_ip_address

upper_bound_ip_address

country

This dataset enables geolocation-based fraud analysis.

3. Credit Card Transactions Dataset (creditcard.csv)

Contains anonymized bank transaction data:

Time: seconds elapsed since first transaction

V1–V28: PCA-transformed features

Amount: transaction value

Target variable: Class

1 → Fraud

0 → Legitimate

This dataset is extremely imbalanced, with fraud cases representing ~0.17% of all transactions.

Interim-1: Work Completed
1. Data Cleaning & Preprocessing

Checked and confirmed no missing values in critical columns

Removed duplicate records

Converted date columns to proper datetime format

Converted IP addresses to integer format for range-based mapping

2. Exploratory Data Analysis (EDA)
Class Imbalance

Both datasets are highly imbalanced

Fraud transactions represent a very small fraction of total data

Accuracy alone is unsuitable as an evaluation metric

Key Observations

Transaction amounts are right-skewed

Fraud patterns vary across traffic sources, browsers, and countries

Credit card fraud cases are extremely rare but financially significant

3. Geolocation Integration

IP addresses were mapped to countries using range-based lookup

A new country feature was added to the e-commerce dataset

Certain countries exhibit higher fraud rates, though some may be influenced by low transaction volumes

4. Feature Engineering

Engineered features include:

Time-based features

hour_of_day

day_of_week

time_since_signup

Behavioral features

transactions_per_user

Categorical encoding

One-hot encoding for source, browser, sex, and country

Feature scaling

Numerical features standardized using StandardScaler

These features aim to capture user behavior, temporal patterns, and geolocation risk.

5. Class Imbalance Strategy

Due to extreme class imbalance:

SMOTE (Synthetic Minority Over-sampling Technique) is planned for use

Resampling will be applied only to the training data during modeling

This approach preserves legitimate transactions while improving fraud signal learning

Key Insights from Interim-1

Fraud detection is a rare-event prediction problem

Time since signup and transaction frequency are strong behavioral indicators

Geolocation features provide valuable risk signals

Both datasets are now clean, feature-rich, and ready for modeling
