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
import altair as alt
import json
import requests

# Load Unemployment Data (from local CSV file)
@st.cache_data
def load_unemployment_data():
 data = pd.read_csv("./test_data/msa_unemployment.csv")
 data.columns = ["series_id", "series_name", "units", "region_name", "region_code", "unemployment_rate"]
 return data

# Load US MSA GeoJSON Map
@st.cache_data
def load_geojson():
 url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json" # Replace with MSA GeoJSON URL
 response = requests.get(url)
 return response.json()

# Load data
st.title("US Unemployment Rate by Metropolitan Statistical Area (MSA)")
unemployment_data = load_unemployment_data()
geojson_data = load_geojson()

# Merge unemployment data with GeoJSON region mapping
unemployment_data["region_code"] = unemployment_data["region_code"].astype(str) # Ensure codes are strings
merged_data = unemployment_data.rename(columns={"region_code": "id"}) # Ensure Altair compatibility

# Create Altair Choropleth Map
st.subheader("Unemployment Rate by MSA")

map_chart = alt.Chart(alt.Data(values=geojson_data["features"])).mark_geoshape().encode(
 color=alt.Color("unemployment_rate:Q", scale=alt.Scale(scheme="reds")),
 tooltip=["region_name:N", "unemployment_rate:Q"]
).transform_lookup(
 lookup="id",
 from_=alt.LookupData(merged_data, "id", ["region_name", "unemployment_rate"])
).properties(
 width=800,
 height=500,
 title="US MSA Unemployment Rate"
)

# Render the Altair Map in Streamlit
st.altair_chart(map_chart, use_container_width=True)

# Allow filtering by MSA name
selected_msa = st.selectbox("Select an MSA to view details:", unemployment_data["region_name"].unique())
filtered_data = unemployment_data[unemployment_data["region_name"] == selected_msa]

st.write("### Selected MSA Data")
st.write(filtered_data)
