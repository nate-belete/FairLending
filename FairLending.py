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




















 Analyzing Refinanced Student Loan Data in the Ascend Dataset



Project Overview



This ad hoc analysis project focuses on determining if the Ascend dataset contains critical student loan information, specifically interest rate, loan term, loan balance, and scheduled monthly payment, for borrowers who refinanced their student loans with our bank. The objective is to evaluate loan details before and after refinancing, leveraging both the Ascend dataset and our internal data, and to link the datasets for a comprehensive analysis.



Goals of the Analysis

1. Assess Data Availability in the Ascend Dataset

Verify the presence of:

• Annual Percentage Rate (APR)

• Loan term

• Loan balance

• Scheduled monthly payment

2. Identify Refinanced Borrowers

• Use our internal dataset to pinpoint customers who refinanced student loans with our bank.

3. Link Pre- and Post-Refinance Loan Details

• Match customers’ pre-refinance loan details from the Ascend dataset to their post-refinance loan details in our internal dataset.

4. Determine Borrower Type

• Categorize customers as undergraduate or graduate borrowers.



Step-by-Step Approach



1. Extract Loan Data from the Ascend Dataset

• Identify account-level details for borrowers, including:

• APR

• Loan term

• Loan balance

• Scheduled monthly payment



2. Analyze Internal Dataset for Refinanced Loans

• Extract data on borrowers who refinanced their student loans with us, focusing on their:

• APR

• Loan term

• Loan balance

• Scheduled monthly payment



3. Determine Borrower Type

• Identify whether the borrower held an undergraduate or graduate student loan, based on internal categorization or relevant data flags.



4. Link the Two Datasets

• Establish a common identifier (e.g., account number, SSN, or unique customer ID) to connect pre-refinance and post-refinance loan records.

• Create a unified dataset showing:

• Pre-refinance details (from Ascend dataset)

• Post-refinance details (from internal dataset)



5. Generate Comparative Insights

• Analyze and document differences in loan terms, interest rates, and monthly payments before and after refinancing.



Deliverables

1. Unified Dataset: A consolidated view of borrower loan details before and after refinancing.

2. Data Availability Report: Confirmation of the availability of required fields in the Ascend dataset.

3. Analysis Summary:

• Changes in loan terms (e.g., lower interest rates, adjusted loan durations).

• Categorization of undergraduate vs. graduate borrowers.

4. Recommendations: Insights for improving data linkage and potential refinements to our processes for better visibility into borrower profiles.



Key Challenges and Solutions



Challenge Solution

Linking datasets with inconsistent IDs Validate and normalize identifiers.

Missing or incomplete data in Ascend Flag and address gaps for future cycles.

Categorizing borrower type accurately Leverage supplemental data or infer.



Timeline



Task Timeline

Extract Ascend dataset details Day 1

Analyze internal dataset Day 2

Link datasets Day 3

Generate unified dataset and insights Day 4

Prepare summary and final recommendations Day 5



Next Steps

• Validate data availability in the Ascend dataset.

• Initiate extraction and analysis phases.

• Update stakeholders with progress and findings.
