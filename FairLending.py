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





















Informed Choice:

Lack of transparency: "Hidden fees", "fine print", "not disclosed", "unclear terms", "confusing language", "misleading information", "omitted details", "buried in the contract", "difficult to understand".
Lack of knowledge: "Didn't know", "wasn't aware", "not explained", "misunderstood", "surprised by", "shocked to find", "wasn't told".
Unfair:

Exploitation: "Taken advantage of", "vulnerable", "pressure to sign", "coerced", "no other option", "forced to", "preyed upon", "targeted".
Excessive penalties: "Unreasonable fees", "exorbitant charges", "outrageous cost", "disproportionate to service", "penalty for...", "unfairly charged".
Obstacles to cancellation: "Difficult to cancel", "impossible to get a refund", "unable to close account", "trapped in contract", "no way out".
Junk Fees:

Unexpected costs: "Surprise fee", "hidden charge", "extra cost", "didn't expect to pay", "added on", "tacked on", "not itemized", "no explanation for fee".
Questionable purpose: "Unnecessary fee", "useless service", "no value", "bogus charge", "pointless", "rip-off", "scam".
Misleading Promotions:

False advertising: "Deceptive", "misleading", "false claims", "not as advertised", "bait and switch", "promised but not delivered", "lied about", "exaggerated".
Unrealistic promises: "Guaranteed returns", "too good to be true", "impossible results", "false hope", "misrepresented benefits".
Wrongful Conduct:

Illegal activity: "Fraudulent", "illegal", "criminal", "violation of law", "unauthorized", "forged", "stolen", "scammed".
Unethical behavior: "Unfair treatment", "discrimination", "harassment", "abuse of power", "breach of trust", "misuse of information", "privacy violation".
