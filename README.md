
# Online Payments Fraud Detection using ML

Machine learning project that helps detect fraudelent transactions from a given dataset.

Online payments are the most common way people make transactions today. However, as online payments increase, so does the risk of fraud. This study aims to distinguish between fraudulent and non-fraudulent payments using a dataset from Kaggle. The dataset contains information about past fraudulent transactions that can be analyzed to detect fraud in online payments.

### Dataset 

The dataset includes 10 variables:

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


### Feature engineering

Time to get our hands dirty with feature engineering. With the available information it is hard to
train the model and get better results. Hence we move onto create new features by altering the
existing features. In this we create three functions which creates a highly relevant feature for the
domain
1. Difference in balance: It is an universal truth that the amount debited from senders account
gets credited into the receivers account without any deviation in cents. But what if there is a
deviation incase of the amount debited and credited. Some could be due to the charges levied
by the service providers, yet we need to flag such unusual instances
2. Surge indicator: Also we have to trigger flag when large amount are involved in the
transaction. From the distribution of amount we understood that we have a lot of outliers with
high amount in transactions. Hence we consider the 75th percentile(450k) as our threshold and
amount which is greater than 450k will be triggered as a flag
3. Frequency indicator: Here we flag the user and not the transaction. When there is a receiver
who receives money from a lot of people, it could be a trigger as it can be for some illegal
games of chance or luck. Hence it is flagged when there is a receiver who receives money for
more than 20 times.
4. Merchant indicator: The customer ids in receiver starts with 'M' which means that they are
merchants and they obviously will have a lot of receiving transactions. So we also flag
whenever there is a merchant receiver
