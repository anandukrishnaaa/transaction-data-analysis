# utils / data_analysis.py

import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from .logger_config import set_logger


ic = set_logger(
    print_to_console=False
)  # Set print_to_console = True for console outputs


# Exploratory data analysis


class DataAnalysis:
    def exploratory_analysis(self):
        # An analysis approach that identifies general patterns in the data
        # -- Init dataframe in local scope --
        df = self.source_df
        # -- Basic summary of the dataset --
        # Create a buffer
        buffer = io.StringIO()
        # Call df.info() and write the output to the buffer
        df.info(buf=buffer)
        # Get the output from the buffer
        info_output = buffer.getvalue()
        # ic(info_output)  # TODO: for out

        # -- Check for missing values --
        # ic(df.isnull().sum())  # TODO: for out

        # Initialize an empty list to store the results
        min_max_cols = []
        # Select only the numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        # Iterate over each column in the numerical DataFrame
        for column in numerical_df.columns:
            # Get the minimum and maximum values for the column
            min_value = numerical_df[column].min()
            max_value = numerical_df[column].max()
            # Append a dictionary containing column name, min, and max values to the list
            min_max_cols.append(
                {"column": column, "min_value": min_value, "max_value": max_value}
            )

        # Results list
        # ic(min_max_cols)  # TODO: for out

        # -- Downcast numerical columns with smaller dtype --
        for col in df.columns:
            if df[col].dtype == "float64":
                df[col] = pd.to_numeric(df[col], downcast="float")
            if df[col].dtype == "int64":
                df[col] = pd.to_numeric(df[col], downcast="unsigned")

        # Use category dtype for categorical column
        df["type"] = df["type"].astype("category")

        # -- Check duplicate values --
        duplicate_count = df.duplicated().sum()
        # ic(duplicate_count)  # TODO: for out

        exploratory_analysis_result = {
            "info": ic(info_output),
            "min_max_cols": ic(min_max_cols),
            "duplicate_check": ic(duplicate_count),
        }

        return exploratory_analysis_result

    def univariate_data_visualization(self):
        # Analysis that involves only a single variable
        # -- Init dataframe in local scope --
        df = self.source_df

        # -- Steps occurrences count --
        step_count = df["step"].value_counts()
        # ic(step_count)  # TODO: for out

        # -- Customer count --
        customer_count = len(df["nameOrig"].value_counts())
        # ic(customer_count)  # TODO: for out

        # -- Count plot of transaction types --
        column = "type"
        transaction_plot = px.histogram(
            df,
            x=column,
            color=column,
            category_orders={column: df[column].value_counts().index},
            title=f"Count plot of {column} transaction",
            labels={column: "Number of transactions"},
        )

        # Adjust layout
        transaction_plot.update_layout(showlegend=False)

        # ic(transaction_plot)  # TODO: for out

        # -- Distribution of transaction amount with KDE plot --
        transaction_distribution_plot = px.histogram(
            df,
            x="amount",
            title="Count plot of transactions",
            labels={"amount": "Number of transactions"},
        )

        # x = oldbalanceOrg, newbalanceOrig, newbalanceDest

        # ic(transaction_distribution_plot)  # TODO: for out

        # -- Non fraudulent v/s fraudulent transactions --

        column = "isFraud"
        non_fraud_vs_fraud_plot = px.bar(
            df,
            x=column,
            color=column,
            category_orders={column: df[column].value_counts().index},
            title=f"Count plot of {column} transaction",
            labels={column: "Number of transactions"},
        )

        # ic(non_fraud_vs_fraud_plot)  # TODO: for out

        univariate_data_visualization_result = {
            "step_count": ic(step_count),
            "customer_count": ic(customer_count),
            "transaction_plot": ic(transaction_plot),
            "transaction_distribution_plot": ic(transaction_distribution_plot),
            "non_fraud_vs_fraud_plot": ic(non_fraud_vs_fraud_plot),
        }

        return univariate_data_visualization_result

    def bivariate_data_visualization(self):
        # Graphs the relationship between two variables that have been measured on a single sample of subjects

        df = self.source_df

        # -- Where fraud transactions occur --
        transaction_type_plot_1 = go.Figure()
        transaction_type_plot_1.add_trace(
            go.Bar(
                x=df["type"],
                y=df.groupby(["type", "isFraud"], observed=False)
                .size()
                .unstack()
                .sum(axis=1),
                name="Non-Fraud",
            )
        )
        transaction_type_plot_1.add_trace(
            go.Bar(
                x=df["type"],
                y=df.groupby(["type", "isFraud"], observed=False).size().unstack()[1],
                name="Fraud",
            )
        )
        transaction_type_plot_1.update_layout(
            title="Count plot of transaction type",
            xaxis_title="Transaction Type",
            yaxis_title="Number of Transactions",
            barmode="stack",
        )

        # Create the second count plot figure
        df_percentage = df.groupby(["type", "isFraud"], observed=True).size().unstack()
        df_percentage = df_percentage.apply(
            lambda x: round(x / sum(x) * 100, 2), axis=1
        )

        transaction_type_plot_2 = go.Figure()
        for col in df_percentage.columns:
            transaction_type_plot_2.add_trace(
                go.Bar(
                    y=df_percentage.index,
                    x=df_percentage[col],
                    orientation="h",
                    name=f"Fraud: {col}",
                )
            )
        transaction_type_plot_2.update_layout(
            title="Count plot of transaction type (percentage)",
            xaxis_title="Percentage",
            yaxis_title="Transaction Type",
            barmode="stack",
        )

        # Pass the figures to the template
        # ic(
        #     transaction_type_plot_1, transaction_type_plot_2
        # )  # TODO: for out

        # -- Fraud amount plot  --
        # Create the count plot using Plotly Express

        df["quantity"] = pd.cut(
            df["amount"], 5, labels=["very low", "low", "moderate", "high", "very high"]
        )

        fraud_amount_plot = px.bar(
            df,
            x="quantity",
            color="isFraud",
            barmode="group",
            category_orders={
                "quantity": ["very low", "low", "moderate", "high", "very high"]
            },
            labels={"quantity": "Amount Quantity", "isFraud": "Fraud"},
        )

        # Update layout
        fraud_amount_plot.update_layout(
            title="Count plot of amount quantity",
            yaxis_title="Number of transactions",
            legend=dict(title="Fraud", x=1.05, y=1, traceorder="normal"),
        )

        # ic(fraud_amount_plot)  # TODO: for out

        # -- Top fraudulent steps --
        # Filter DataFrame for fraudulent transactions
        df_fraudulent = df[df["isFraud"] == 1]

        # Get the top 10 steps for fraudulent transactions
        top_10_steps = df_fraudulent["step"].value_counts().head(10).reset_index()
        top_10_steps.columns = ["Step", "Number of fraudulent transactions"]

        # Create the bar plot using Plotly Express
        top_fraud_plot = px.bar(
            top_10_steps,
            x="Step",
            y="Number of fraudulent transactions",
            color="Step",
            labels={
                "Number of fraudulent transactions": "Number of fraudulent transactions"
            },
            color_discrete_map={"Step": "lightsteelblue"},
        )

        # Update layout
        top_fraud_plot.update_layout(
            title="Top 10 steps that often lead to fraudulent transactions",
            xaxis_title="Step",
            yaxis_title="Number of fraudulent transactions",
            showlegend=False,
            xaxis=dict(tickmode="linear"),  # Forces ticks to appear at all steps
        )

        # ic(top_fraud_plot)  # TODO: for out

        # -- Pre-balance amount vs fraud --
        # Create the count plot using Plotly Express
        df["oldbalanceOrg_amt"] = pd.cut(
            df["oldbalanceOrg"],
            5,
            labels=["very low", "low", "moderate", "high", "very high"],
        )

        pre_balance_transaction_plot = px.bar(
            df,
            x="oldbalanceOrg_amt",
            color="isFraud",
            barmode="group",
            labels={
                "oldbalanceOrg_amt": "Initial Customers Pre-Transaction Balance Amount",
                "isFraud": "Fraud",
            },
            color_discrete_map={"isFraud": "PuBu"},
        )

        # Update layout
        pre_balance_transaction_plot.update_layout(
            title="Count plot of initial customers pre-transaction balance amount",
            xaxis_title="Initial Customers Pre-Transaction Balance Amount",
            yaxis_title="Number of transactions",
            legend=dict(title="Fraud", x=1.05, y=1, traceorder="normal"),
        )

        # Pass the figure to the template
        # ic(pre_balance_transaction_plot)  # TODO: for out

        bivariate_data_visualization_result = {
            "transaction_type_plot_1": ic(transaction_type_plot_1),
            "transaction_type_plot_2": ic(transaction_type_plot_2),
            "fraud_amount_plot": ic(fraud_amount_plot),
            "top_fraud_plot": ic(top_fraud_plot),
            "pre_balance_transaction_plot": ic(pre_balance_transaction_plot),
        }

        return bivariate_data_visualization_result

    def multivariate_data_visualization(self):
        # Visualizing more than one data value in a single renderer. This is done for many reasons, including to: View the relationship between two or more variables. Compare or contrast the difference between two variables

        # -- Init dataframe in local scope --
        df = self.source_df

        # -- Correlation matrix b/w balances --

        # Identify non-numeric columns and exclude them from the correlation matrix
        non_numeric_columns = df.select_dtypes(exclude=["number"]).columns
        df_numeric = df.drop(columns=non_numeric_columns)

        # Calculate the correlation matrix (Spearman) for numeric columns
        corr_matrix = df_numeric.corr("spearman")

        # Create the correlation heatmap using Plotly
        corr_matrix_plot = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="PuBu",
                colorbar=dict(title="Correlation"),
                zmin=-1,  # Adjust based on your correlation range
                zmax=1,  # Adjust based on your correlation range
                text=corr_matrix.values.round(3),
                hoverongaps=False,
            )
        )

        # Update layout
        corr_matrix_plot.update_layout(
            title="Correlation Heatmap",
            xaxis=dict(title="Features"),
            yaxis=dict(title="Features"),
        )

        # Pass the figure to the template
        ic(corr_matrix_plot)  # TODO: for out

        multivariate_data_visualization_result = {
            "corr_matrix_plot": ic(corr_matrix_plot)
        }

        return multivariate_data_visualization_result

    # Pass the figure to the template
    def __init__(self, file_path):
        # Init dataframe to be passed to class method
        self.source_df = pd.read_csv(file_path, index_col=False)
        ic("Source file initialised into dataframe")
