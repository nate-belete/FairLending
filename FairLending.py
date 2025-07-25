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















import boto3
import cloudpickle
import os
import io
from sentence_transformers import SentenceTransformer
from snowflake.snowpark.functions import col # Optional: if doing Snowpark ops

def model(dbt, session):
dbt.config(
materialized="table",
packages=["boto3", "cloudpickle", "sentence-transformers", "umap-learn", "hdbscan"]
)

# --- Configuration ---
s3_bucket = "sofi-data-science"
s3_prefix = "RISK_ANALYTICS_IRM/Issue_Management/Topic_Modeling/Primary/"
embedding_model_s3_prefix = f"{s3_prefix}primary_embedding_model/"
local_embedding_dir = "/tmp/primary_embedding_model"
s3 = boto3.client("s3")

# --- Helper: Download cloudpickle model from S3 ---
def load_pickle_from_s3(key):
buffer = io.BytesIO()
s3.download_fileobj(s3_bucket, s3_prefix + key, buffer)
buffer.seek(0)
return cloudpickle.load(buffer)

# --- Helper: Download entire embedding model directory from S3 ---
def download_embedding_model_from_s3():
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=s3_bucket, Prefix=embedding_model_s3_prefix):
for obj in page.get("Contents", []):
s3_key = obj["Key"]
rel_path = os.path.relpath(s3_key, embedding_model_s3_prefix)
local_path = os.path.join(local_embedding_dir, rel_path)

os.makedirs(os.path.dirname(local_path), exist_ok=True)
s3.download_file(s3_bucket, s3_key, local_path)

# --- Download models from S3 ---
umap_model = load_pickle_from_s3("primary_umap_model.pkl")
hdbscan_model = load_pickle_from_s3("primary_hdbscan_model.pkl")
topic_model = load_pickle_from_s3("primary_topic_model.pkl")

download_embedding_model_from_s3()
embedding_model = SentenceTransformer(local_embedding_dir)

# --- Load input data from ref table ---
df = dbt.ref("your_input_table") # Replace with actual source table
pandas_df = df.to_pandas()

# --- Inference pipeline ---
documents = pandas_df["issue_description"].tolist()
embeddings = embedding_model.encode(documents)
umap_embeddings = umap_model.transform(embeddings)
labels = hdbscan_model.predict(umap_embeddings)

pandas_df["predicted_topic"] = labels

return session.create_dataframe(pandas_df)
