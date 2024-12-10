# CREDIT-CARD-FRAUD-DETECTION
A machine learning project for detecting credit card fraud using Random Forest and Isolation Forest. It identifies fraudulent transactions with high accuracy, leveraging SMOTE for imbalanced data and feature scaling. Built with Python, it ensures scalable, automated fraud detection in real-time scenarios.
# Credit Card Fraud Detection

## Overview

This project implements a machine learning-based solution for detecting fraudulent credit card transactions. It leverages **Random Forest** and **Isolation Forest** models to classify transactions as fraudulent or legitimate. By using techniques like **SMOTE** for handling imbalanced data and **feature scaling** for preprocessing, the project achieves high accuracy in fraud detection.

The system is designed to be scalable and can be adapted for real-time fraud detection in financial systems.

---

## Features

- **Fraud Detection**: Classifies transactions into fraudulent and legitimate categories.
- **Machine Learning Models**: Includes Random Forest for robust classification and Isolation Forest for anomaly detection.
- **Imbalanced Data Handling**: Uses **SMOTE (Synthetic Minority Over-sampling Technique)** to handle data imbalance.
- **High Accuracy**: Achieves >99% accuracy in fraud detection using Random Forest.
- **Scalable**: Can process large datasets, making it suitable for real-world applications.

---

## Dataset

The dataset used contains anonymized credit card transactions, including the following:
- **Features**: Numerical features (V1-V28), transaction amount, and time.
- **Target Variable**: Indicates whether a transaction is fraudulent (`1`) or legitimate (`0`).
- **Imbalance**: Fraudulent transactions make up only a small percentage of the dataset.

---

## Technologies Used

- **Python**: Programming language used for development.
- **Scikit-learn**: Machine learning library for training and evaluation.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Imbalanced-learn (imblearn)**: SMOTE for handling imbalanced datasets.
- **Matplotlib & Seaborn**: Data visualization libraries.

---

## Getting Started

### Prerequisites

- Python 3.x installed on your system.
- Install dependencies listed in the `requirements.txt` file.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
