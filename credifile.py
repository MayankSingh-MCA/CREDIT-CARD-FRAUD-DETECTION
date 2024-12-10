# Import required libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


# Define a reusable function to preprocess and scale the data
def preprocess_data(transactions_path, cc_info_path):
    # Step 1: Load the datasets
    transactions = pd.read_csv(r"C:\Users\bdrin\Desktop\projectCredit\credit\transactions.csv")
    cc_info = pd.read_csv(r"C:\Users\bdrin\Desktop\projectCredit\credit\cc_info.csv")

    # Merge datasets on `credit_card`
    data = transactions.merge(cc_info, on="credit_card", how="inner")

    # Convert `date` to datetime format and extract features like day, month, hour
    data['date'] = pd.to_datetime(data['date'])
    data['transaction_day'] = data['date'].dt.day
    data['transaction_month'] = data['date'].dt.month
    data['transaction_hour'] = data['date'].dt.hour
    data.drop(columns=['date'], inplace=True)  # Drop the original `date` column

    # Feature scaling for numerical columns
    scaler = StandardScaler()
    numerical_cols = ['transaction_dollar_amount', 'credit_card_limit', 'Long', 'Lat']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Label the data (flag transactions exceeding 70% of credit limit as fraud)
    data['is_fraud'] = (data['transaction_dollar_amount'] > 0.7 * data['credit_card_limit']).astype(int)

    return data


# Define a function for training and evaluating the Random Forest model
def train_random_forest(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_test)

    # Evaluate Random Forest
    print("Random Forest Classifier:")
    print(classification_report(y_test, y_pred_rfc))
    print("Accuracy:", accuracy_score(y_test, y_pred_rfc))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rfc))


# Define a function for training and evaluating the Isolation Forest model
def train_isolation_forest(X_train, X_test, y_test):
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_train)
    y_pred_if = iso_forest.predict(X_test)
    y_pred_if = [1 if x == -1 else 0 for x in y_pred_if]  # Convert anomalies to fraud

    # Evaluate Isolation Forest
    print("\nIsolation Forest:")
    print(classification_report(y_test, y_pred_if))
    print("Accuracy:", accuracy_score(y_test, y_pred_if))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_if))


# Main program execution
if __name__ == "__main__":
    # File paths (update these paths as necessary)
    transactions_path = r'C:\Users\bdrin\Desktop\projectCredit\credit\transactions.csv'
    cc_info_path = r'C:\Users\bdrin\Desktop\projectCredit\credit\cc_info.csv'

    # Preprocess data
    data = preprocess_data(transactions_path, cc_info_path)

    # Prepare features and target
    X = data.drop(columns=['is_fraud', 'credit_card', 'city', 'state', 'zipcode'])
    y = data['is_fraud']

    # Handle imbalanced data using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Train and evaluate Random Forest Classifier
    train_random_forest(X_train, X_test, y_train, y_test)

    # Train and evaluate Isolation Forest
    train_isolation_forest(X_train, X_test, y_test)
