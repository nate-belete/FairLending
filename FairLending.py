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
def loan_trends(df, loan_product):
    # Making sure that the dates are in datetime format
    df['date'] = pd.to_datetime(df['date'])
    current_rate = df[df.product == loan_product]['Metric_A'].iloc[-1]

    # Calculate the intervals
    three_months_ago_rate = df[(df.date > df['date'].max() - pd.DateOffset(months=3)) & (df.product == loan_product)]['Metric_A'].mean()
    six_months_ago_rate = df[(df.date > df['date'].max() - pd.DateOffset(months=6)) & (df.product == loan_product)]['Metric_A'].mean()
    twelve_months_ago_rate = df[(df.date > df['date'].max() - pd.DateOffset(months=12)) & (df.product == loan_product)]['Metric_A'].mean()
    eighteen_months_ago_rate = df[(df.date > df['date'].max() - pd.DateOffset(months=18)) & (df.product == loan_product)]['Metric_A'].mean()

    
    # Compare the current metric with past metrics and deduce trends
    trend_three_months = "increasing" if current_rate > three_months_ago_rate else "decreasing"
    trend_six_months = "increasing" if current_rate > six_months_ago_rate else "decreasing"
    trend_twelve_months = "increasing" if current_rate > twelve_months_ago_rate else "decreasing"
    trend_eighteen_months = "increasing" if current_rate > eighteen_months_ago_rate else "decreasing"

    # Generate the narrative
    narrative = f'The current value of Metric_A for {PRODUCT_NAMES[loan_product]} is {current_rate}. Over the past 3 months, it was {three_months_ago_rate} and the trend is {trend_three_months}. Over the last 6 months, it was {six_months_ago_rate} and the trend is {trend_six_months}. Over the last 12 months, it was {twelve_months_ago_rate} and the trend is {trend_twelve_months}. Over the last 18 months, it was {eighteen_months_ago_rate} and the trend is {trend_eighteen_months}.'

    st.markdown(narrative)
