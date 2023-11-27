# Basic libraries
import pandas as pd
import numpy as np
import os
from django.conf import settings

# Visualization libraries

import plotly.graph_objects as go

# preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ML libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack


# Metrics Libraries
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


# Misc libraries
import warnings

from .logger_config import set_logger

ic = set_logger(print_to_console=False)

warnings.filterwarnings("ignore")

# Model Training

BEST_MODEL_PATH = settings.BASE_DIR / "main/utils/models/nbModel_grid.pkl"
SCALER_PATH = settings.BASE_DIR / "main/utils/models/scaler.pkl"
VECTORIZER_DEST_PATH = settings.BASE_DIR / "main/utils/models/vectorizer_dest.pkl"
VECTORIZER_ORG_PATH = settings.BASE_DIR / "main/utils/models/vectorizer_org.pkl"

"""
Preprocessing the data, putting it through different classification algos to find the one best suited for our purpose and fine tune it to be accurate and robust.
"""


def train_main_model(file_path):
    """
    Feature engineering:
        Time to get our hands dirty with feature engineering. With the available information it is hard to train the model and get better results. Hence we move onto create new features by altering the existing features. In this we create three functions which creates a highly relevant feature for the domain

            - Difference in balance: It is an universal truth that the amount debited from senders account gets credited into the receivers account without any deviation in cents. But what if there is a deviation incase of the amount debited and credited. Some could be due to the charges levied by the service providers, yet we need to flag such unusual instances

            - Surge indicator: Also we have to trigger flag when large amount are involved in the transaction. From the distribution of amount we understood that we have a lot of outliers with high amount in transactions. Hence we consider the 75th percentile(450k) as our threshold and amount which is greater than 450k will be triggered as a flag

            - Frequency indicator: Here we flag the user and not the transaction. When there is a receiver who receives money from a lot of people, it could be a trigger as it can be for some illegal games of chance or luck. Hence it is flagged when there is a receiver who receives money for more than 20 times.

            - Merchant indicator: The customer ids in receiver starts with 'M' which means that they are merchants and they obviously will have a lot of receiving transactions. So we also flag whenever there is a merchant receiver
    """
    # Load dataset
    paysim = pd.read_csv(file_path)

    # Tallying the balance
    def balance_diff(data):
        """balance_diff checks whether the money debited from sender has exactly credited to the receiver
        then it creates a new column which indicates 1 when there is a deviation else 0
        """
        # Sender's balance
        orig_change = data["newbalanceOrig"] - data["oldbalanceOrg"]
        orig_change = orig_change.astype(int)
        for i in orig_change:
            if i < 0:
                data["orig_txn_diff"] = round(data["amount"] + orig_change, 2)
            else:
                data["orig_txn_diff"] = round(data["amount"] - orig_change, 2)
        data["orig_txn_diff"] = data["orig_txn_diff"].astype(int)
        data["orig_diff"] = [1 if n != 0 else 0 for n in data["orig_txn_diff"]]

        # Receiver's balance
        dest_change = data["newbalanceDest"] - data["oldbalanceDest"]
        dest_change = dest_change.astype(int)
        for i in dest_change:
            if i < 0:
                data["dest_txn_diff"] = round(data["amount"] + dest_change, 2)
            else:
                data["dest_txn_diff"] = round(data["amount"] - dest_change, 2)
        data["dest_txn_diff"] = data["dest_txn_diff"].astype(int)
        data["dest_diff"] = [1 if n != 0 else 0 for n in data["dest_txn_diff"]]

        data.drop(["orig_txn_diff", "dest_txn_diff"], axis=1, inplace=True)

    # Surge indicator
    def surge_indicator(data):
        """Creates a new column which has 1 if the transaction amount is greater than the threshold
        else it will be 0"""
        data["surge"] = [1 if n > 450000 else 0 for n in data["amount"]]

    # Frequency indicator
    def frequency_receiver(data):
        """Creates a new column which has 1 if the receiver receives money from many individuals
        else it will be 0"""
        data["freq_Dest"] = data["nameDest"].map(data["nameDest"].value_counts())
        data["freq_dest"] = [1 if n > 20 else 0 for n in data["freq_Dest"]]

        data.drop(["freq_Dest"], axis=1, inplace=True)

    # Tracking the receiver as merchant or not
    def merchant(data):
        """We also have customer ids which starts with M in Receiver name, it indicates merchant
        this function will flag if there is a merchant in receiver end"""
        values = ["M"]
        conditions = list(map(data["nameDest"].str.contains, values))
        data["merchant"] = np.select(conditions, "1", "0")

    # Applying balance_diff function
    balance_diff(paysim)

    # print(paysim["orig_diff"].value_counts()) // TODO: for out
    # print(paysim["dest_diff"].value_counts()) // TODO: for out

    # Applying surge_indicator function
    surge_indicator(paysim)
    # print(paysim["surge"].value_counts())  // TODO: for out

    # Applying frequency_receiver function
    frequency_receiver(paysim)
    # print(paysim["freq_dest"].value_counts())  // TODO: for out

    """
    Pre-processing data
        Before moving to build a machine learning model, it is mandatory to pre-process the data so that the model trains without any error and can learn better to provide better results

        - Balancing the target
            From the pie chart below we can clearly see that the target label is heavily imbalance as we usually have only 0.2% of fraudulent data which is in-sufficient for machine to learn and flag when fraud transactions happen.
    """
    # Creating a copy
    paysim_1 = paysim.copy()

    # Checking for balance in target
    balance_target_plot = go.Figure(
        data=[
            go.Pie(
                labels=["Not Fraud", "Fraud"],
                values=paysim_1["isFraud"].value_counts(),
                title="Checking for balance in target",
            )
        ]
    )

    # Getting the max size
    max_size = paysim_1["isFraud"].value_counts().max()

    # Balancing the target label
    lst = [paysim_1]
    for class_index, group in paysim_1.groupby("isFraud"):
        lst.append(group.sample(max_size - len(group), replace=True))
    paysim_1 = pd.concat(lst)

    # Checking the balanced target
    check_balance_target_plot = go.Figure(
        data=[
            go.Pie(
                labels=["Not Fraud", "Fraud"],
                values=paysim_1["isFraud"].value_counts(),
                title="Checking the balanced target",
            )
        ]
    )

    """
        - One hot encoding
            One of the most important feature we have is type which is categorical in type. Since it doesnt have any ordinal nature and since the classes are less, we prefer applying one hot encoding.
    """
    # One hot encoding
    paysim_1 = pd.concat(
        [paysim_1, pd.get_dummies(paysim_1["type"], prefix="type_")], axis=1
    )
    paysim_1.drop(["type"], axis=1, inplace=True)

    # print(paysim_1.head()) // TODO: for out

    """
        - Split and Standardize
            In this module we create the independent and dependent feature, then split them into train and test data where training size is 70%. Later we collect all the numerical features and apply StandardScaler() function which transforms the distribution so that the mean becomes 0 and standard deviation becomes 1
    """
    # Splitting dependent and independent variable
    paysim_2 = paysim_1.copy()
    X = paysim_2.drop("isFraud", axis=1)
    y = paysim_2["isFraud"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=111
    )

    # Standardizing the numerical columns
    col_names = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    features_train = X_train[col_names]
    features_test = X_test[col_names]
    scaler = StandardScaler().fit(features_train.values)
    # Save the instances to files
    if scaler:
        joblib.dump(scaler, SCALER_PATH)

    features_train = scaler.transform(features_train.values)
    features_test = scaler.transform(features_test.values)
    X_train[col_names] = features_train
    X_test[col_names] = features_test

    """
        - Tokenization
            We had the customer ids and merchant ids stored in object type. It is bad to apply one hot encoding in it as it can lead to more features and curse of dimensionality can incur. Hence we are applying tokenization here as it can create an unique id number which is in 'int' type for each customer id
    """

    # Tokenization using scikit-learn CountVectorizer
    vectorizer_org = CountVectorizer()
    vectorizer_dest = CountVectorizer()

    # Fit and transform on the training data
    customers_train_org = vectorizer_org.fit_transform(X_train["nameOrig"])
    customers_train_dest = vectorizer_dest.fit_transform(X_train["nameDest"])

    # Transform the test data
    customers_test_org = vectorizer_org.transform(X_test["nameOrig"])
    customers_test_dest = vectorizer_dest.transform(X_test["nameDest"])

    # Save the instances to files
    if vectorizer_org:
        joblib.dump(vectorizer_org, VECTORIZER_ORG_PATH)
    if vectorizer_dest:
        joblib.dump(vectorizer_dest, VECTORIZER_DEST_PATH)

    # Reset the index of X_train and X_test
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    # Concatenate the DataFrames
    X_train = pd.concat(
        [
            X_train.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1),
            pd.DataFrame(hstack([customers_train_org, customers_train_dest]).toarray()),
        ],
        axis=1,
        ignore_index=True,
    )

    X_test = pd.concat(
        [
            X_test.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1),
            pd.DataFrame(hstack([customers_test_org, customers_test_dest]).toarray()),
        ],
        axis=1,
        ignore_index=True,
    )

    """
    - Model Building
    We have successfully processed the data and it is time for serving the data to the model. It is time consuming to find out which model works best for our data. Hence I have utlized pipeline to run our data through all the classification algorithm and select the best which gives out the maximum accuracy.
    """

    # creating the objects
    logreg_cv = LogisticRegression(solver="liblinear", random_state=123)
    dt_cv = DecisionTreeClassifier(random_state=123)
    knn_cv = KNeighborsClassifier()
    svc_cv = SVC(kernel="linear", random_state=123)
    nb_cv = GaussianNB()
    rf_cv = RandomForestClassifier(random_state=123)
    cv_dict = {
        0: "Logistic Regression",
        1: "Decision Tree",
        2: "KNN",
        3: "SVC",
        4: "Naive Bayes",
        5: "Random Forest",
    }
    cv_models = [logreg_cv, dt_cv, knn_cv, svc_cv, nb_cv, rf_cv]

    # Create an empty dictionary to store results
    algo_compare_results = {}  # TODO: for out

    for i, model in enumerate(cv_models):
        # Perform cross-validation and store results
        accuracy_scores = cross_val_score(
            model, X_train, y_train, cv=10, scoring="accuracy"
        )
        algo_compare_results[cv_dict[i]] = {
            "mean_accuracy": accuracy_scores.mean(),
            "standard_deviation": accuracy_scores.std(),
            "accuracy_scores": accuracy_scores.tolist(),
        }

    # Print or save the results as needed
    # for model_name, results in algo_compare_results.items():
    #     print(f"{model_name}:\n{results}\n")

    """
    - Hyperparameter Tuning
        Lets fit the Naive bayes model by tuning the model with its parameters. Here we are gonna tune var_smoothing which is a stability calculation to widen (or smooth) the curve and therefore account for more samples that are further away from the distribution mean. In this case, np.logspace returns numbers spaced evenly on a log scale, starts from 0, ends at -9, and generates 100 samples.
    """

    param_grid_nb = {"var_smoothing": np.logspace(0, -9, num=100)}

    nbModel_grid = GridSearchCV(
        estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1
    )
    nbModel_grid.fit(X_train, y_train)
    # Save the model after fitting the test data
    if nbModel_grid:
        joblib.dump(nbModel_grid, BEST_MODEL_PATH)

    # print(nbModel_grid.best_estimator_)  //TODO: for out

    """
    - Evaluation of model
        Time to explore the truth of high numbers by evaluating against testing data
    """

    # Function for Confusion matrix
    def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix"):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        # Create a trace for the heatmap
        trace = go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Count"),
        )

        layout = go.Layout(
            title=title,
            xaxis=dict(title="Predicted label"),
            yaxis=dict(title="True label"),
        )

        confusion_matrix_plot = go.Figure(data=[trace], layout=layout)
        return confusion_matrix_plot

    # Predict with the selected best parameter
    y_pred = nbModel_grid.predict(X_test)

    # Plotting confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    confusion_matrix_plot = plot_confusion_matrix(
        cm, classes=["Not Fraud", "Fraud"]
    )  # // TODO: for out

    """
    The model has identified false positives but never let even a single false negative which is more important than False Positive. Since we can't miss out a fraud transactions, but we can manage false positive results by investigating them
    """

    # Classification metrics
    classification_report_result = classification_report(
        y_test, y_pred, target_names=["Not Fraud", "Fraud"]
    )  # TODO: for out

    train_main_model_result = {
        "orig_diff_count": ic(paysim["orig_diff"].value_counts()),
        "dest_diff_count": ic(paysim["dest_diff"].value_counts()),
        "surge_count": ic(paysim["surge"].value_counts()),
        "freq_dest": ic(paysim["freq_dest"].value_counts()),
        "balance_target_plot": ic(balance_target_plot),
        "check_balance_target_plot": ic(check_balance_target_plot),
        "data_head": ic(
            paysim_1.head().to_html(
                classes="table table-bordered table-striped", escape=False, index=False
            )
        ),
        "algo_compare_results": ic(algo_compare_results),
        "nbModel_grid_estimate": ic(nbModel_grid.best_estimator_),
        "confusion_matrix_plot": ic(confusion_matrix_plot),
        "classification_report_result": ic(classification_report_result),
    }

    """
    When we found that our false negatives are more important than false positives, we have to look at the recall number and we have 100% recall in finding the fraud transactions and 100% precision in finding the non fraud tranactions and on an average our model performs more than 70% accurate which is pretty good and there are possible chance to improve the performance of this model.
    """

    return train_main_model_result


"""
Various functions making use of the trained ML model to derive insights from the dataset.
"""

# Load the instances from files
# scaler = joblib.load(SCALER_PATH)
# vectorizer_org = joblib.load(VECTORIZER_ORG_PATH)
# vectorizer_dest = joblib.load(VECTORIZER_DEST_PATH)


def custom_model_check(file_path):
    if not os.path.exists(BEST_MODEL_PATH):
        # File does not exist exists, load the train_main_model library to generate BEST_MODEL_PATH file
        ic("Custom model does not exist at BEST_MODEL_PATH, attempting to create anew.")
        ic(train_main_model(file_path))
        nbModel_grid = joblib.load(BEST_MODEL_PATH)
        model = nbModel_grid
    else:
        nbModel_grid = joblib.load(BEST_MODEL_PATH)
        model = nbModel_grid
    # model = GaussianNB()
    return model


# Data preprocessing
def load_and_prep_df(file_path):
    # Loading into dataframe
    df = pd.read_csv(file_path)
    df["type"] = df["type"].map(
        {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
    )
    df["isFraud"] = df["isFraud"].map({0: "No Fraud", 1: "Fraud"})

    return df


# Function to train the model
def train_sub_model(df, model):
    x = np.array(df[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
    y = np.array(df["isFraud"])
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.10, random_state=42
    )
    model.fit(xtrain, ytrain)
    return model, xtest, ytest


# Get unique customer names to be shown in the template
def get_customer_ids(df):
    # Fetch unique values from the "nameOrig" column
    customer_ids = df["nameOrig"].unique()

    # Convert unique values to a list for dropdown
    customer_ids_list = customer_ids.tolist()

    return customer_ids_list


# Find the most fraudulent amount in the given dataset
def find_most_fraudulent_amount(df):
    fraud_data = df[df["isFraud"] == "Fraud"]
    most_fraudulent_amount = fraud_data["amount"].max()
    most_fraudulent_amount = most_fraudulent_amount.item()

    return most_fraudulent_amount


# TODO: for out


# Get anomalies and patterns - find all other interesting anomalies, patterns, and their indications
def get_anomalies_and_patterns(df, model):
    anomalies = df[
        model.predict(
            np.array(df[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
        )
        == "Fraud"
    ]
    return anomalies  # TODO: for out


# Predict fraud-prone customers - find the top x (x can be any number, for example, 10) most fraud-prone customers
def predict_fraud_prone_customers(model, df, top_x):
    all_customers = df["nameOrig"].unique()
    fraud_probabilities = []

    for customer_id in all_customers:
        customer_data = df[df["nameOrig"] == customer_id]
        features = np.array(
            customer_data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
        )
        fraud_probability = model.predict_proba(features)[:, 1].mean()
        fraud_probabilities.append(
            {"customer_id": customer_id, "fraud_probability": fraud_probability}
        )

    fraud_probabilities.sort(key=lambda x: x["fraud_probability"], reverse=True)
    return fraud_probabilities[:top_x]  # TODO: for out


# Predict least fraud-prone customers - find the bottom x (x can be any number, for example, 10) least fraud-prone customers
def predict_least_fraud_prone_customers(model, df, bottom_x):
    all_customers = df["nameOrig"].unique()
    fraud_probabilities = []

    for customer_id in all_customers:
        customer_data = df[df["nameOrig"] == customer_id]
        features = np.array(
            customer_data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
        )
        fraud_probability = model.predict_proba(features)[:, 1].mean()
        fraud_probabilities.append(
            {"customer_id": customer_id, "fraud_probability": fraud_probability}
        )

    fraud_probabilities.sort(key=lambda x: x["fraud_probability"])
    return fraud_probabilities[:bottom_x]


# Compute fraud probabilities - compute the fraud probability of each unique customer in the dataset
def compute_fraud_probabilities(model, df, x=None):
    all_customers = df["nameOrig"].unique()
    fraud_probabilities = []

    # Limit the number of customer records to fetch
    if x is not None:
        all_customers = all_customers[:x]

    for customer_id in all_customers:
        customer_data = df[df["nameOrig"] == customer_id]
        features = np.array(
            customer_data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
        )
        fraud_probability = model.predict_proba(features)[:, 1].mean()
        fraud_probabilities.append(
            {"customer_id": customer_id, "fraud_probability": fraud_probability}
        )

    return fraud_probabilities  # TODO: for out


# Get anomalies and patterns for a customer - find all other interesting anomalies, patterns, and their indications for a single customer
def get_anomalies_and_patterns_for_customer(df, model, customer_id):
    customer_data = df[df["nameOrig"] == customer_id]
    customer_data.drop(["isFraud", "isFlaggedFraud"], axis=1, inplace=True)
    # Check if the condition is met
    if (
        model.predict(
            np.array(
                customer_data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
            )
        )
        == "Fraud"
    ).any():
        anomalies = customer_data[
            model.predict(
                np.array(
                    customer_data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
                )
            )
            == "Fraud"
        ]
    else:
        # If no fraud detected, create a DataFrame with all columns as "-" for the first row
        columns = customer_data.columns
        data = {col: ["-"] * len(columns) for col in columns}
        anomalies = pd.DataFrame(data).head(1)

    return anomalies  # TODO: for out


# Find out the probability % of a given customer (nameOrig) has_been_frauded, has_committed_fraud, victim_probability, perpetrator_probability
def get_customer_probabilities(model, df, customer_id):
    customer_data = df[df["nameOrig"] == customer_id]
    features = np.array(
        customer_data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
    )
    fraud_probability = model.predict_proba(features)[
        :, 1
    ]  # Probability of being fraud
    return {
        "has_been_frauded": fraud_probability.mean(),
        "has_committed_fraud": 1 - fraud_probability.mean(),
        "victim_probability": (model.predict(features) == "No Fraud").mean(),
        "perpetrator_probability": (model.predict(features) == "Fraud").mean(),
    }  # TODO: for out


def get_frauds(df):
    fraud_data = df[df["isFraud"] == "Fraud"]
    fraud_customers = fraud_data["nameOrig"].unique().tolist()
    return fraud_customers
