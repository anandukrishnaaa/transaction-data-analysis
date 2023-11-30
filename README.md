
# Online Payments Fraud Detection using ML

Machine learning project that helps detect fraudelent transactions from a given dataset, written in `Python` built using `django`, a pinch of `javascript` and *styled* with `bootstrap5`.

Online payments are the most common way people make transactions today. However, as online payments increase, so does the risk of fraud. This study aims to distinguish between fraudulent and non-fraudulent payments using a dataset from Kaggle. The dataset contains information about past fraudulent transactions that can be analyzed to detect fraud in online payments.

## Dataset 

> Dataset can be downloaded from [here](https://we.tl/t-uOwSw3oO1x) till 03 December 2023 18.00 IST. 

Place it inside the `dataset` folder and run the `sample_generator.py` file to create sample csv files. 

The default settings for the `sample_generator.py`:
- `num_files = 10` (*10 sample files*) 
- `chunk_size = 100` (*100 records each*).

### The dataset includes 10 variables:

- `step`: maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps
744 (30 days simulation).
- `type`: CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
- `amount`: amount of the transaction in local currency.
- `nameOrig`: customer who started the transaction
- `oldbalanceOrg`: initial balance before the transaction
- `newbalanceOrig`: new balance after the transaction
- `nameDest`: customer who is the recipient of the transaction
- `oldbalanceDest`: initial balance recipient before the transaction. Note that there is not
information for customers that start with M (Merchants).
- `newbalanceDest`:  new balance recipient after the transaction. Note that there is not
information for customers that start with M (Merchants).
- `isFraud`: This is the transactions made by the fraudulent agents inside the simulation. In this
specific dataset the fraudulent behavior of the agents aims to profit by taking control or
customers accounts and try to empty the funds by transferring to another account and then
cashing out of the system.
- `isFlaggedFraud`: The business model aims to control massive transfers from one account to
another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer
more than 200.000 in a single transaction.

## Feature engineering

Time to get our hands dirty with feature engineering. With the available information it is hard to
train the model and get better results. Hence we move onto create new features by altering the
existing features. In this we create three functions which creates a highly relevant feature for the
domain

1. **Difference in balance**: It is an universal truth that the amount debited from senders account
gets credited into the receivers account without any deviation in cents. But what if there is a
deviation incase of the amount debited and credited. Some could be due to the charges levied
by the service providers, yet we need to flag such unusual instances

2. **Surge indicator**: Also we have to trigger flag when large amount are involved in the
transaction. From the distribution of amount we understood that we have a lot of outliers with
high amount in transactions. Hence we consider the 75th percentile(450k) as our threshold and
amount which is greater than 450k will be triggered as a flag

3. **Frequency indicator**: Here we flag the user and not the transaction. When there is a receiver
who receives money from a lot of people, it could be a trigger as it can be for some illegal
games of chance or luck. Hence it is flagged when there is a receiver who receives money for
more than 20 times.

4. **Merchant indicator**: The customer ids in receiver starts with 'M' which means that they are
merchants and they obviously will have a lot of receiving transactions. So we also flag
whenever there is a merchant receiver

## Basic analysis results

### Exploratory Data Analysis

#### 1. Info Output
**Function Name:** `exploratory_analysis`  
**Visualization:** Displays information about the dataset, including data types and non-null counts.


#### 2. Min-Max Values of Numerical Columns
**Function Name:** `exploratory_analysis`  
**Visualization:** Presents the minimum and maximum values for each numerical column to understand the data range.


#### 3. Duplicate Check
**Function Name:** `exploratory_analysis`  
**Visualization:** Counts the number of duplicate rows in the dataset to identify potential data integrity issues.


### Univariate Data Visualization

#### 1. Step Occurrences Count
**Function Name:** `univariate_data_visualization`  
**Visualization:** Bar chart displaying the count of occurrences for each step to identify step-wise transaction patterns.


#### 2. Customer Count
**Function Name:** `univariate_data_visualization`  
**Visualization:** Displays the total number of unique customers in the dataset to understand customer engagement.


#### 3. Transaction Type Distribution
**Function Name:** `univariate_data_visualization`  
**Visualization:** Count plot showing the distribution of transaction types to analyze the prevalence of each type.


### Bivariate Data Visualization

#### 1. Transaction Type Plot
**Function Name:** `bivariate_data_visualization`  
**Visualization:** Stack bar chart showing the count of transactions for each type, differentiated by fraud status.


#### 2. Fraud Amount Plot
**Function Name:** `bivariate_data_visualization`  
**Visualization:** Grouped bar chart depicting the count of transactions based on amount categories and fraud status.


### Multivariate Data Visualization

#### 1. Correlation Matrix Heatmap
**Function Name:** `multivariate_data_visualization`  
**Visualization:** Displays a heatmap illustrating the correlation between numerical features in the dataset.

## Machine model training and application results

#### 1. Balance in Target Pie Chart
**Function Name:** `train_main_model`  
**Visualization:** Displays two pie charts representing the balance in the target variable 'isFraud' before and after balancing.

#### 2. Type Distribution Plot
**Function Name:** `train_main_model`  
**Visualization:** Plots the distribution of transaction types after one-hot encoding.

#### 3. Surge Indicator Pie Chart
**Function Name:** `train_main_model`  
**Visualization:** Illustrates the distribution of surge indicators (transactions with amounts greater than 450,000).

#### 4. Frequency Indicator Pie Chart
**Function Name:** `train_main_model`  
**Visualization:** Presents a pie chart of the distribution of frequency indicators for receiver entities.

#### 5. Confusion Matrix Heatmap
**Function Name:** `train_main_model`  
**Visualization:** Generates a heatmap of the confusion matrix to evaluate model performance.

#### 6. Anomalies and Patterns Table
**Function Name:** `get_anomalies_and_patterns`  
**Visualization:** Outputs a table containing data records identified as anomalies by the model.

#### 7. Algorithm Comparison Results
**Function Name:** `train_main_model`  
**Visualization:** Compares the performance of different classification algorithms using cross-validation.

#### 8. Classification Report Results
**Function Name:** `train_main_model`  
**Visualization:** Provides detailed metrics, including precision, recall, and F1-score, for fraud and non-fraud classes.

#### 9. Customer Fraud Probability Table
**Function Name:** `compute_fraud_probabilities`  
**Visualization:** Outputs a table showing the fraud probability for each unique customer.

#### 10. Individual Customer Anomalies Table
**Function Name:** `get_anomalies_and_patterns_for_customer`  
**Visualization:** Presents a table with detailed information about anomalies and patterns for a specific customer.

#### 11. Customer Fraud Probability Metrics
**Function Name:** `get_customer_probabilities`  
**Usage:** Calculates various fraud probability metrics for a given customer.

#### 12. Training Sub-Model
**Function Name:** `train_sub_model`  
**Usage:** Trains a sub-model on a subset of data.

#### 13. Predict Fraud-Prone Customers
**Function Name:** `predict_fraud_prone_customers`  
**Usage:** Predicts the top x fraud-prone customers based on fraud probability.

#### 14. Predict Least Fraud-Prone Customers
**Function Name:** `predict_least_fraud_prone_customers`  
**Usage:** Predicts the bottom x least fraud-prone customers based on fraud probability.

#### 15. Custom Model Check
**Function Name:** `custom_model_check`  
**Usage:** Checks if a custom model exists, if not, creates and saves a new one.

#### 16. Load and Prep DataFrame
**Function Name:** `load_and_prep_df`  
**Usage:** Loads and preprocesses the dataset.

#### 17. Find Most Fraudulent Amount
**Function Name:** `find_most_fraudulent_amount`  
**Usage:** Identifies the most fraudulent amount in the dataset.

#### 18. Get Customer IDs
**Function Name:** `get_customer_ids`  
**Usage:** Fetches unique customer IDs from the dataset.

#### 19. Get Frauds
**Function Name:** `get_frauds`  
**Usage:** Retrieves a list of customers involved in fraudulent transactions.


## Powered by

- **Python version:** Python 3.12.0

- **Dependency manager:** [pipenv 2023.10.3](https://pypi.org/project/pipenv/)

### Packages and dependencies graph 

```python
arabic-reshaper==3.0.0
black==23.11.0
├── click [required: >=8.0.0, installed: 8.1.7]
│   └── colorama [required: Any, installed: 0.4.6]
├── mypy-extensions [required: >=0.4.3, installed: 1.0.0]
├── packaging [required: >=22.0, installed: 23.2]
├── pathspec [required: >=0.9.0, installed: 0.11.2]
└── platformdirs [required: >=2, installed: 4.0.0]
django-auto-logout==0.5.1
django-bootstrap5==23.3
└── django [required: >=3.2, installed: 4.2.7]
    ├── asgiref [required: >=3.6.0,<4, installed: 3.7.2]
    ├── sqlparse [required: >=0.3.1, installed: 0.4.4]
    └── tzdata [required: Any, installed: 2023.3]
djlint==1.34.0
├── click [required: >=8.0.1,<9.0.0, installed: 8.1.7]
│   └── colorama [required: Any, installed: 0.4.6]
├── colorama [required: >=0.4.4,<0.5.0, installed: 0.4.6]
├── cssbeautifier [required: >=1.14.4,<2.0.0, installed: 1.14.11]
│   ├── editorconfig [required: >=0.12.2, installed: 0.12.3]
│   ├── jsbeautifier [required: Any, installed: 1.14.11]
│   │   ├── editorconfig [required: >=0.12.2, installed: 0.12.3]
│   │   └── six [required: >=1.13.0, installed: 1.16.0]
│   └── six [required: >=1.13.0, installed: 1.16.0]
├── html-tag-names [required: >=0.1.2,<0.2.0, installed: 0.1.2]
├── html-void-elements [required: >=0.1.0,<0.2.0, installed: 0.1.0]
├── jsbeautifier [required: >=1.14.4,<2.0.0, installed: 1.14.11]
│   ├── editorconfig [required: >=0.12.2, installed: 0.12.3]
│   └── six [required: >=1.13.0, installed: 1.16.0]
├── json5 [required: >=0.9.11,<0.10.0, installed: 0.9.14]
├── pathspec [required: >=0.11.0,<0.12.0, installed: 0.11.2]
├── PyYAML [required: >=6.0,<7.0, installed: 6.0.1]
├── regex [required: >=2023.0.0,<2024.0.0, installed: 2023.10.3]
└── tqdm [required: >=4.62.2,<5.0.0, installed: 4.66.1]
    └── colorama [required: Any, installed: 0.4.6]
html5lib==1.1
├── six [required: >=1.9, installed: 1.16.0]
└── webencodings [required: Any, installed: 0.5.1]
icecream==2.1.3
├── asttokens [required: >=2.0.1, installed: 2.4.1]
│   └── six [required: >=1.12.0, installed: 1.16.0]
├── colorama [required: >=0.3.9, installed: 0.4.6]
├── executing [required: >=0.3.1, installed: 2.0.1]
└── pygments [required: >=2.2.0, installed: 2.17.2]
imblearn==0.0
└── imbalanced-learn [required: Any, installed: 0.11.0]
    ├── joblib [required: >=1.1.1, installed: 1.3.2]
    ├── numpy [required: >=1.17.3, installed: 1.26.2]
    ├── scikit-learn [required: >=1.0.2, installed: 1.3.2]
    │   ├── joblib [required: >=1.1.1, installed: 1.3.2]
    │   ├── numpy [required: >=1.17.3,<2.0, installed: 1.26.2]
    │   ├── scipy [required: >=1.5.0, installed: 1.11.4]
    │   │   └── numpy [required: >=1.21.6,<1.28.0, installed: 1.26.2]
    │   └── threadpoolctl [required: >=2.0.0, installed: 3.2.0]
    ├── scipy [required: >=1.5.0, installed: 1.11.4]
    │   └── numpy [required: >=1.21.6,<1.28.0, installed: 1.26.2]
    └── threadpoolctl [required: >=2.0.0, installed: 3.2.0]
keras==2.15.0
plotly==5.18.0
├── packaging [required: Any, installed: 23.2]
└── tenacity [required: >=6.2.0, installed: 8.2.3]
pyHanko==0.20.1
├── asn1crypto [required: >=1.5.1, installed: 1.5.1]
├── click [required: >=7.1.2, installed: 8.1.7]
│   └── colorama [required: Any, installed: 0.4.6]
├── cryptography [required: >=3.3.1, installed: 41.0.5]
│   └── cffi [required: >=1.12, installed: 1.16.0]
│       └── pycparser [required: Any, installed: 2.21]
├── pyhanko-certvalidator [required: ==0.24.*, installed: 0.24.1]
│   ├── asn1crypto [required: >=1.5.1, installed: 1.5.1]
│   ├── cryptography [required: >=3.3.1, installed: 41.0.5]
│   │   └── cffi [required: >=1.12, installed: 1.16.0]
│   │       └── pycparser [required: Any, installed: 2.21]
│   ├── oscrypto [required: >=1.1.0, installed: 1.3.0]
│   │   └── asn1crypto [required: >=1.5.1, installed: 1.5.1]
│   ├── requests [required: >=2.24.0, installed: 2.31.0]
│   │   ├── certifi [required: >=2017.4.17, installed: 2023.11.17]
│   │   ├── charset-normalizer [required: >=2,<4, installed: 3.3.2]
│   │   ├── idna [required: >=2.5,<4, installed: 3.4]
│   │   └── urllib3 [required: >=1.21.1,<3, installed: 2.1.0]
│   └── uritools [required: >=3.0.1, installed: 4.0.2]
├── pyyaml [required: >=5.3.1, installed: 6.0.1]
├── qrcode [required: >=6.1, installed: 7.4.2]
│   ├── colorama [required: Any, installed: 0.4.6]
│   ├── pypng [required: Any, installed: 0.20220715.0]
│   └── typing-extensions [required: Any, installed: 4.8.0]
├── requests [required: >=2.24.0, installed: 2.31.0]
│   ├── certifi [required: >=2017.4.17, installed: 2023.11.17]
│   ├── charset-normalizer [required: >=2,<4, installed: 3.3.2]
│   ├── idna [required: >=2.5,<4, installed: 3.4]
│   └── urllib3 [required: >=1.21.1,<3, installed: 2.1.0]
└── tzlocal [required: >=4.3, installed: 5.2]
    └── tzdata [required: Any, installed: 2023.3]
pypdf==3.17.1
python-bidi==0.4.2
└── six [required: Any, installed: 1.16.0]
python-dotenv==1.0.0
rlPyCairo==0.3.0
├── freetype-py [required: >=2.3, installed: 2.3.0]
└── pycairo [required: >=1.20.0, installed: 1.25.1]
seaborn==0.13.0
├── matplotlib [required: >=3.3,!=3.6.1, installed: 3.8.2]
│   ├── contourpy [required: >=1.0.1, installed: 1.2.0]
│   │   └── numpy [required: >=1.20,<2.0, installed: 1.26.2]
│   ├── cycler [required: >=0.10, installed: 0.12.1]
│   ├── fonttools [required: >=4.22.0, installed: 4.45.1]
│   ├── kiwisolver [required: >=1.3.1, installed: 1.4.5]
│   ├── numpy [required: >=1.21,<2, installed: 1.26.2]
│   ├── packaging [required: >=20.0, installed: 23.2]
│   ├── pillow [required: >=8, installed: 10.1.0]
│   ├── pyparsing [required: >=2.3.1, installed: 3.1.1]
│   └── python-dateutil [required: >=2.7, installed: 2.8.2]
│       └── six [required: >=1.5, installed: 1.16.0]
├── numpy [required: >=1.20,!=1.24.0, installed: 1.26.2]
└── pandas [required: >=1.2, installed: 2.1.3]
    ├── numpy [required: >=1.26.0,<2, installed: 1.26.2]
    ├── python-dateutil [required: >=2.8.2, installed: 2.8.2]
    │   └── six [required: >=1.5, installed: 1.16.0]
    ├── pytz [required: >=2020.1, installed: 2023.3.post1]
    └── tzdata [required: >=2022.1, installed: 2023.3]
svglib==1.5.1
├── cssselect2 [required: >=0.2.0, installed: 0.7.0]
│   ├── tinycss2 [required: Any, installed: 1.2.1]
│   │   └── webencodings [required: >=0.4, installed: 0.5.1]
│   └── webencodings [required: Any, installed: 0.5.1]
├── lxml [required: Any, installed: 4.9.3]
├── reportlab [required: Any, installed: 4.0.7]
│   └── pillow [required: >=9.0.0, installed: 10.1.0]
└── tinycss2 [required: >=0.6.0, installed: 1.2.1]
    └── webencodings [required: >=0.4, installed: 0.5.1]
```

## Prerequisites  

1. `Python` and `pip` must be installed and path set.
2. Install [pipenv 2023.10.3](https://pypi.org/project/pipenv/)

## Steps to use

1. Clone project repo from github
2. Make sure you have `Python` installed
3. Create an empty directory `.venv` inside the project directory by running the command `mkdir transaction_data_analysis\.venv`.
4. Dive into `transaction_data_analysis` using `cd transaction_data_analysis`
5. Run `pipenv install`
6. Wait for all dependencies to be installed.
7. Once installation of all dependencies are complete, run the command `.venv\Scripts\activate` in the terminal to activate the virtual environment. 
8. Locate the `.env.example` file, inside that
   1. Generate a `SECRET_KEY` using [Djecrety](https://djecrety.ir/) and paste it in place of `YOUR_SECRET_KEY`.
   2. Change `your_database_name` to the name you would want for your database file.
   3. Make sure not to include any extra spaces or quotation marks.
9. Rename the file as `.env` (Remove the `.example` from the end.)
10. Make sure you're in the right place in the terminal by typing in `dir` into command prompt - you should be able to see a `manage.py` file in the output, else navigate to the correct `transaction_data_analysis` directory using `cd`.
11. Run the below commands
    1.  `python manage.py makemigrations`
    2.  `python manage.py migrate`
    3.  `python manage.py createsuperuser` - for `admin` panel access
12. Finally run `python manage.py runserver` to deploy the server
13. The application can now be viewed in the browser at `http://127.0.0.1:8000/`