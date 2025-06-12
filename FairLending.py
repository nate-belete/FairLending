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

















import numpy as np
import pandas as pd
import hdbscan
import umap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# ---- Step 1: Load Data ----
# Assume df is already loaded and contains a column 'finding_description'
# Assume sentence embeddings are precomputed and loaded
embeddings = np.load("issue_embeddings_original_desc.npy") # shape (1169, 384)

# ---- Step 2: Cluster Embeddings using HDBSCAN ----
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric='euclidean', prediction_data=True)
cluster_labels = clusterer.fit_predict(embeddings)

# Add cluster labels to df
df['cluster'] = cluster_labels

# Print cluster stats
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)
print(f"Estimated number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# ---- Step 3: UMAP Visualization ----
umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean')
umap_embeddings = umap_model.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=cluster_labels, cmap='Spectral', s=10)
plt.title(f'HDBSCAN Clusters (Estimated: {n_clusters})')
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()

# ---- Step 4: Remove Noise ----
df = df[df['cluster'] != -1]

# ---- Step 5: Get Top Keywords per Cluster ----
def get_top_keywords(df, text_col='finding_description', cluster_col='cluster', n_keywords=10):
cluster_keywords = {}
for cluster in sorted(df[cluster_col].unique()):
cluster_texts = df[df[cluster_col] == cluster][text_col]

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(cluster_texts)

tfidf_scores = tfidf_matrix.mean(axis=0).A1
feature_names = vectorizer.get_feature_names_out()
top_indices = tfidf_scores.argsort()[::-1][:n_keywords]
top_words = [feature_names[i] for i in top_indices]

cluster_keywords[cluster] = top_words
return cluster_keywords

top_keywords_per_cluster = get_top_keywords(df)

# ---- Step 6: Build Cluster Summary Table ----
cluster_summary = []
for cluster, keywords in top_keywords_per_cluster.items():
example_text = df[df['cluster'] == cluster]['finding_description'].iloc[0]
cluster_summary.append({
'cluster': cluster,
'top_keywords': ', '.join(keywords),
'example_issue': example_text[:250] + ('...' if len(example_text) > 250 else '')
})

summary_df = pd.DataFrame(cluster_summary)
summary_df = summary_df.sort_values('cluster')

# ---- Step 7: Output Summary ----
from ace_tools import display_dataframe_to_user
display_dataframe_to_user(name="Cluster Summary Table", dataframe=summary_df)
