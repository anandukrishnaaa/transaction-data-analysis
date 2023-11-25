import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
import joblib

# Constants
RF_MODEL_PATH = "models/random_forest_model.pkl"
LR_MODEL_PATH = "models/logistic_regression_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
NUMERICAL_FEATURES_PATH = "models/numerical_features.pkl"
CATEGORICAL_FEATURES_PATH = "models/categorical_features.pkl"


# Function to train models
def train_models(dataset_path, RF_MODEL_PATH, LR_MODEL_PATH):
    df = pd.read_csv(dataset_path)

    # Extract features and target variable
    X = df.drop(["isFraud", "isFlaggedFraud"], axis=1)
    y = df["isFraud"]

    # Define numerical and categorical features
    numerical_features = X.select_dtypes(include=["float64", "int64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # Create transformers for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Create a preprocessor with separate transformers for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
    )

    # Create pipelines for Random Forest and Logistic Regression models
    rf_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(class_weight="balanced", random_state=42),
            ),
        ]
    )

    lr_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", random_state=42),
            ),
        ]
    )

    # Split the dataset into training and testing sets using GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["nameOrig"]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train the models
    rf_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    # Evaluate models on the test set (optional)
    rf_predictions = rf_pipeline.predict(X_test)
    lr_predictions = lr_pipeline.predict(X_test)

    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_predictions)}")
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_predictions)}")

    # Save the trained models and preprocessor to files
    joblib.dump(rf_pipeline, RF_MODEL_PATH)
    joblib.dump(lr_pipeline, LR_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(numerical_features, NUMERICAL_FEATURES_PATH)
    joblib.dump(categorical_features, CATEGORICAL_FEATURES_PATH)


# Function to preprocess data before making predictions
def preprocess_data(df, preprocessor):
    return preprocessor.transform(df)


# Function to predict using the Random Forest model
def predict_rf(X_processed, RF_MODEL_PATH):
    # Load the trained Random Forest model
    rf_model = joblib.load(RF_MODEL_PATH)

    # Make predictions using the Random Forest model
    predictions = rf_model.predict(X_processed)

    return predictions


# Function to predict using the Logistic Regression model
def predict_lr(X_processed, LR_MODEL_PATH):
    # Load the trained Logistic Regression model
    lr_model = joblib.load(LR_MODEL_PATH)

    # Make predictions using the Logistic Regression model
    predictions = lr_model.predict(X_processed)

    return predictions


# Function to get fraud information
def get_fraud_info(dataset_path, name_orig, RF_MODEL_PATH, LR_MODEL_PATH, preprocessor):
    # Load the trained models
    rf_model = joblib.load(RF_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)

    # Load your dataset
    df = pd.read_csv(dataset_path)

    # Filter data for the specific customer
    customer_df = df[df["nameOrig"] == name_orig].copy()

    # Feature engineering (customize based on your dataset)
    customer_df["transaction_frequency"] = customer_df.groupby("nameOrig")[
        "step"
    ].transform("count")
    customer_df["time_since_last_transaction"] = customer_df.groupby("nameOrig")[
        "step"
    ].diff()

    # Separate the target variable
    y = customer_df["isFraud"]

    # Prepare features for anomaly detection
    features = customer_df[
        [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "transaction_frequency",
            "time_since_last_transaction",
        ]
    ]

    # Preprocess your features using the preprocessor fit on the training data
    X_processed = preprocessor.transform(features)

    # Make predictions using the Random Forest model
    rf_predictions = rf_model.predict(X_processed)

    # Make predictions using the Logistic Regression model
    lr_predictions = lr_model.predict(X_processed)

    # Calculate the victim probability using the Logistic Regression model
    victim_probability = lr_model.predict_proba(X_processed)[:, 1].mean()

    # Calculate the perpetrator probability using the Random Forest model
    perpetrator_probability = rf_model.predict_proba(X_processed)[:, 1].mean()

    # Determine if the person has been a victim of fraud
    has_been_frauded = 1 if victim_probability > 0 else 0

    # Determine if the person has committed fraud
    has_committed_fraud = 1 if perpetrator_probability > 0 else 0

    fraud_info = {
        "has_been_frauded": has_been_frauded,
        "has_committed_fraud": has_committed_fraud,
        "victim_probability": victim_probability,
        "perpetrator_probability": perpetrator_probability,
    }

    return fraud_info


# Function to predict fraud-prone customers
def predict_fraud_prone_customers(df, RF_MODEL_PATH, LR_MODEL_PATH, preprocessor):
    # Load the trained models
    rf_model = joblib.load(RF_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)

    # Load feature names
    numerical_features = joblib.load(NUMERICAL_FEATURES_PATH)
    categorical_features = joblib.load(CATEGORICAL_FEATURES_PATH)

    # Feature engineering (you can customize this based on your dataset)
    df["transaction_frequency"] = df.groupby("nameOrig")["step"].transform("count")
    df["time_since_last_transaction"] = df.groupby("nameOrig")["step"].diff()

    # Include only the features used during training
    numerical_data = df[numerical_features]
    categorical_data = df[categorical_features]

    # Combine numerical and categorical features
    features = pd.concat([numerical_data, categorical_data], axis=1)

    # Preprocess your features
    features_processed = preprocess_data(features, preprocessor)

    # Make predictions using the Logistic Regression model
    lr_predictions = lr_model.predict(features_processed)

    # Make predictions using the Random Forest model
    rf_predictions = rf_model.predict(features_processed)

    # Identify fraud-prone customers based on both models
    fraud_prone_customers = df[(lr_predictions == 1) & (rf_predictions == 1)][
        "nameOrig"
    ].unique()

    return fraud_prone_customers


# Function to compute fraud probabilities
def compute_fraud_probabilities(df, RF_MODEL_PATH, LR_MODEL_PATH, preprocessor):
    # Load the trained models
    rf_model = joblib.load(RF_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)

    # Load feature names
    numerical_features = joblib.load(NUMERICAL_FEATURES_PATH)
    categorical_features = joblib.load(CATEGORICAL_FEATURES_PATH)

    # Feature engineering (you can customize this based on your dataset)
    df["transaction_frequency"] = df.groupby("nameOrig")["step"].transform("count")
    df["time_since_last_transaction"] = df.groupby("nameOrig")["step"].diff()

    # Include only the features used during training
    numerical_data = df[numerical_features]
    categorical_data = df[categorical_features]

    # Combine numerical and categorical features
    features = pd.concat([numerical_data, categorical_data], axis=1)

    # Preprocess your features
    features_processed = preprocess_data(features, preprocessor)

    # Make predictions using the Logistic Regression model
    lr_predictions = lr_model.predict(features_processed)

    # Make predictions using the Random Forest model
    rf_predictions = rf_model.predict(features_processed)

    # Calculate fraud probabilities based on both models
    fraud_probabilities = pd.DataFrame(
        {
            "nameOrig": df["nameOrig"],
            "fraud_probability_lr": lr_model.predict_proba(features_processed)[:, 1],
            "fraud_probability_rf": rf_model.predict_proba(features_processed)[:, 1],
        }
    )

    # Identify customers with high fraud probabilities
    high_fraud_probabilities = df[
        (fraud_probabilities["fraud_probability_lr"] > 0.5)
        & (fraud_probabilities["fraud_probability_rf"] > 0.5)
    ]["nameOrig"].unique()

    return fraud_probabilities, high_fraud_probabilities


# Function to find the amount associated with the most fraudulent transaction
def find_most_fraudulent_amount(df, RF_MODEL_PATH, preprocessor):
    # Load the trained Random Forest model
    rf_model = joblib.load(RF_MODEL_PATH)

    # Load feature names
    numerical_features = joblib.load(NUMERICAL_FEATURES_PATH)
    categorical_features = joblib.load(CATEGORICAL_FEATURES_PATH)

    # Feature engineering (you can customize this based on your dataset)
    df["transaction_frequency"] = df.groupby("nameOrig")["step"].transform("count")
    df["time_since_last_transaction"] = df.groupby("nameOrig")["step"].diff()

    # Include only the features used during training
    numerical_data = df[numerical_features]
    categorical_data = df[categorical_features]

    # Combine numerical and categorical features
    features = pd.concat([numerical_data, categorical_data], axis=1)

    # Preprocess your features
    features_processed = preprocess_data(features, preprocessor)

    # Make predictions using the Random Forest model
    rf_predictions = rf_model.predict(features_processed)

    # Identify transactions with high anomaly scores (potential fraud)
    high_anomaly_transactions = df[rf_predictions == 1]

    # Find the amount associated with the most fraudulent transaction
    most_fraudulent_amount = high_anomaly_transactions.loc[
        high_anomaly_transactions["amount"].idxmax()
    ]["amount"]

    return most_fraudulent_amount


# Function to get anomalies and patterns for the entire dataset
def get_anomalies_and_patterns(df, RF_MODEL_PATH, preprocessor):
    # Load the trained Random Forest model
    rf_model = joblib.load(RF_MODEL_PATH)

    # Load feature names
    numerical_features = joblib.load(NUMERICAL_FEATURES_PATH)
    categorical_features = joblib.load(CATEGORICAL_FEATURES_PATH)

    # Feature engineering (customize based on your dataset)
    df["transaction_frequency"] = df.groupby("nameOrig")["step"].transform("count")
    df["time_since_last_transaction"] = df.groupby("nameOrig")["step"].diff()

    # Include only the features used during training
    numerical_data = df[numerical_features]
    categorical_data = df[categorical_features]

    # Combine numerical and categorical features
    features = pd.concat([numerical_data, categorical_data], axis=1)

    # Preprocess your features
    features_processed = preprocess_data(features, preprocessor)

    # Make predictions using the Random Forest model
    rf_predictions = rf_model.predict(features_processed)

    # Identify transactions with high anomaly scores (potential fraud)
    high_anomaly_transactions = df[rf_predictions == 1]

    # Find the amount associated with the most fraudulent transaction
    most_fraudulent_amount = high_anomaly_transactions.loc[
        high_anomaly_transactions["amount"].idxmax()
    ]["amount"]

    # Extract anomalies and patterns
    anomalies_and_patterns = {
        "most_fraudulent_amount": most_fraudulent_amount,
        "high_anomaly_transactions": high_anomaly_transactions,
        # Add more features, anomalies, and patterns as needed
    }

    return anomalies_and_patterns


# Function to get anomalies and patterns for a specific customer
def get_anomalies_and_patterns_for_customer(
    df, customer_name, RF_MODEL_PATH, preprocessor
):
    # Load the trained Random Forest model
    rf_model = joblib.load(RF_MODEL_PATH)

    # Load feature names
    numerical_features = joblib.load(NUMERICAL_FEATURES_PATH)
    categorical_features = joblib.load(CATEGORICAL_FEATURES_PATH)

    # Filter data for the specific customer
    customer_df = df[df["nameOrig"] == customer_name].copy()

    # Feature engineering (customize based on your dataset)
    customer_df["transaction_frequency"] = customer_df.groupby("nameOrig")[
        "step"
    ].transform("count")
    customer_df["time_since_last_transaction"] = customer_df.groupby("nameOrig")[
        "step"
    ].diff()

    # Include only the features used during training
    numerical_data = customer_df[numerical_features]
    categorical_data = customer_df[categorical_features]

    # Combine numerical and categorical features
    features = pd.concat([numerical_data, categorical_data], axis=1)

    # Preprocess your features
    features_processed = preprocess_data(features, preprocessor)

    # Make predictions using the Random Forest model
    rf_predictions = rf_model.predict(features_processed)

    # Identify transactions with high anomaly scores (potential fraud)
    high_anomaly_transactions = customer_df[rf_predictions == 1]

    # Find the amount associated with the most fraudulent transaction
    most_fraudulent_amount = high_anomaly_transactions.loc[
        high_anomaly_transactions["amount"].idxmax()
    ]["amount"]

    # Extract anomalies and patterns for the specific customer
    anomalies_and_patterns = {
        "most_fraudulent_amount": most_fraudulent_amount,
        "high_anomaly_transactions": high_anomaly_transactions,
        # Add more features, anomalies, and patterns as needed
    }

    return anomalies_and_patterns


# Example usage
path = r"C:\Users\AK\Projects\code\coding-projects\transaction-data-analysis\dataset\sample.csv"

# Train models
# train_models(path, RF_MODEL_PATH, LR_MODEL_PATH)

# # Load preprocessor and feature names
preprocessor = joblib.load(PREPROCESSOR_PATH)

# # Perform predictions and analyses
df = pd.read_csv(path)
name_orig = "C1666544295"

fraud_info = get_fraud_info(path, name_orig, RF_MODEL_PATH, LR_MODEL_PATH, preprocessor)
print(fraud_info)

# fraud_prone_customers = predict_fraud_prone_customers(
#     df, RF_MODEL_PATH, LR_MODEL_PATH, preprocessor
# )
# print(fraud_prone_customers)

# fraud_probabilities, high_fraud_probabilities = compute_fraud_probabilities(
#     df, RF_MODEL_PATH, LR_MODEL_PATH, preprocessor
# )
# print(fraud_probabilities)
# print(high_fraud_probabilities)

# most_fraudulent_amount = find_most_fraudulent_amount(df, RF_MODEL_PATH, preprocessor)
# print(f"Most Fraudulent Amount: {most_fraudulent_amount}")

# anomalies_and_patterns = get_anomalies_and_patterns(df, RF_MODEL_PATH, preprocessor)
# print(anomalies_and_patterns)

# customer_name = "C1666544295"  # Replace with an actual customer name
# anomalies_and_patterns_customer = get_anomalies_and_patterns_for_customer(
#     df, customer_name, RF_MODEL_PATH, preprocessor
# )
# print(anomalies_and_patterns_customer)
