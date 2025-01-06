import numpy as np
import pandas as pd


class FairLending:
    def __init__(self, df, policy, protected_class, comparator_class):
        """
        Initializes the FairLending class.

        Args:
            df: DataFrame containing the data.
            policy: Dictionary defining the current and proposed policy.
            protected_class: List of protected class column names.
            comparator_class: List of comparator class column names.
        """
        self.df = df
        self.policy = policy
        self.protected_class = protected_class
        self.comparator_class = comparator_class

    def generate_condition(self, policy_conditions):
        """
        Generates a combined condition based on the policy conditions.

        Args:
            policy_conditions: List of dictionaries defining policy conditions.

        Returns:
            A boolean pandas Series which can be used for filtering.
        """
        conditions = [
            (self.df[condition['Variable']] >= condition['Lower Bound']) &
            (self.df[condition['Variable']] < condition['Upper Bound'])
            for condition in policy_conditions
        ]
        return np.logical_and.reduce(conditions)

    def generate_summary_table(self, policy_column):
        """
        Generates a summary table based on the policy column.

        Args:
            policy_column: Column name representing the policy.

        Returns:
            A pandas DataFrame containing the summary table.
        """
        summary_table = self.df.groupby(policy_column).agg(
            total_loans=('Total_Record_Count', 'sum'),
            demographic_info_provided=('RaceCheck', 'sum'),
            age_info_provided=('AgeCheck', 'sum'),
            gender_info_provided=('GenderCheck', 'sum'),
            White=('White', 'sum'),
            Minority=('Minority', 'sum'),
            Black=('Black', 'sum'),
            Hispanic=('Hispanic', 'sum'),
            Asian=('Asian', 'sum'),
            Male=('Male', 'sum'),
            Female=('Female', 'sum'),
            Under_62=('Under 62', 'sum'),
            Over_62=('Over 62', 'sum')
        )
        return summary_table

    def calculate_proportion_table(self, summary_table):
        """
        Calculates the proportion for each column based on the summary table.

        Args:
            summary_table: DataFrame containing the summary table.

        Returns:
            A pandas DataFrame containing the proportion table.
        """
        proportion_table = summary_table.div(summary_table.sum(axis=0), axis=1)
        return proportion_table

    def calculate_gaps(self, proportion_table):
        """
        Calculates the gaps between protected and comparator classes for each policy table.

        Args:
            proportion_table: DataFrame containing the proportion table.

        Returns:
            A pandas DataFrame containing the gaps.
        """
        gaps = {}
        for p, c in zip(self.protected_class, self.comparator_class):
            gaps[p] = proportion_table[p] - proportion_table[c]
        return pd.DataFrame(gaps)

    def calculate_air(self, proportion_table):
        """
        Calculates the air between protected and comparator classes for each policy table.

        Args:
            proportion_table: DataFrame containing the proportion table.

        Returns:
            A pandas DataFrame containing the air.
        """
        air = {}
        for p, c in zip(self.protected_class, self.comparator_class):
            air[p] = proportion_table[p] / proportion_table[c]
        return pd.DataFrame(air)

    def apply_policies(self):
        """
        Applies the defined policies to the DataFrame and returns their summaries.

        Returns:
            A dictionary containing summary tables, proportion tables, gaps, and air for each policy.
        """
        results = {}
        for policy_name, conditions in self.policy.items():
            policy_condition = self.generate_condition(conditions)
            self.df[policy_name] = policy_condition
            summary_table = self.generate_summary_table(policy_name)
            proportion_table = self.calculate_proportion_table(summary_table)
            gaps = self.calculate_gaps(proportion_table)
            air = self.calculate_air(proportion_table)
            results[policy_name] = {
                'summary_table': summary_table,
                'proportion_table': proportion_table,
                'gaps': gaps,
                'air': air
            }

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Generate Synthetic data
dates = pd.date_range(start='1/1/2021', end='1/12/2022')
products = ['A', 'B', 'C', 'D']
metrics_A = {'Metric_A': np.random.randint(50, 100, len(dates)),
             'Metric_B': np.random.randint(100, 150, len(dates)),
             'Metric_C': np.random.randint(50, 200, len(dates))}

metrics_B = {'Metric_A': np.random.randint(200, 250, len(dates)),
             'Metric_B': np.random.randint(250, 300, len(dates)),
             'Metric_C': np.random.randint(100, 500, len(dates))}

# I'm creating metrics for 'C' and 'D' similar to 'A' and 'B'
metrics_C = {'Metric_A': np.random.randint(100, 150, len(dates)),
             'Metric_B': np.random.randint(150, 200, len(dates)),
             'Metric_C': np.random.randint(100, 300, len(dates))}

metrics_D = {'Metric_A': np.random.randint(300, 350, len(dates)),
             'Metric_B': np.random.randint(350, 400, len(dates)),
             'Metric_C': np.random.randint(200, 600, len(dates))}

data_A = pd.DataFrame({'date': dates, 'product': 'A', **metrics_A})
data_B = pd.DataFrame({'date': dates, 'product': 'B', **metrics_B})
data_C = pd.DataFrame({'date': dates, 'product': 'C', **metrics_C})
data_D = pd.DataFrame({'date': dates, 'product': 'D', **metrics_D})

df = pd.concat([data_A, data_B, data_C, data_D])

def plot_timeseries(df, selected_metric):
    product_list = df['product'].unique()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    for product, ax in zip(product_list, axs.flatten()):
        data = df[df['product'] == product]
        sns.lineplot(data=data, x='date', y=selected_metric, ax=ax)
        ax.set_title(f'Product {product}')
        
        # Format y labels
        if selected_metric == 'Metric_A':
            ax.yaxis.set_major_formatter(lambda x, pos: '{:.0%}'.format(x))
        elif selected_metric in ['Metric_B', 'Metric_C']:
            ax.yaxis.set_major_formatter(lambda x, pos: '${:.0f}'.format(x))
        
    plt.tight_layout()
    st.pyplot(fig)

def main():
    selected_metric = st.sidebar.selectbox('Select a Metric', ['Metric_A', 'Metric_B', 'Metric_C'])

    # display plots
    st.markdown("## Time Series Plot")
    plot_timeseries(df, selected_metric)


if __name__ == "__main__":
    main()
