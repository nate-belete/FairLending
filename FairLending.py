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
import re

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_systemic_risk_findings.csv")

df = load_data()

# App Structure
st.set_page_config(page_title="Systemic Risk Dashboard", layout="wide")
st.title("🛡️ Systemic Risk Insights Dashboard")

# Navigation Tabs
tab = st.sidebar.radio("Go to", ["📊 Summary Dashboard", "📋 Findings Table", "🔎 Detail View", "📈 Insights & Visualizations", "⬇️ Export"])

# --- 1. Summary Dashboard ---
if tab == "📊 Summary Dashboard":
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Findings", len(df))
    col2.metric("Systemic Risk %", f"{(df['is_systemic_risk'].mean()*100):.1f}%")
    col3.metric("Avg. Systemic Risk Score", f"{df['systemic_risk_score'].mean():.2f}")

    st.subheader("Findings Over Time")
    df['created_date'] = pd.to_datetime(df['created_date'])
    trend = df.groupby(df['created_date'].dt.to_period("M")).size().reset_index(name="count")
    trend['created_date'] = trend['created_date'].astype(str)
    chart = alt.Chart(trend).mark_line(point=True).encode(
    x='created_date:T',
    y='count:Q'
    ).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)

# --- 2. Findings Table ---
elif tab == "📋 Findings Table":
    st.subheader("Filter Findings")

    score_range = st.slider("Systemic Risk Score", 0, 2, (0, 2))
    severity_filter = st.multiselect("Severity Rating", df['severity_rating'].unique(), default=list(df['severity_rating'].unique()))
    unit_filter = st.multiselect("Assessment Unit", df['assessment_unit'].unique(), default=list(df['assessment_unit'].unique()))

    filtered_df = df[
    (df['systemic_risk_score'] >= score_range[0]) &
    (df['systemic_risk_score'] <= score_range[1]) &
    (df['severity_rating'].isin(severity_filter)) &
    (df['assessment_unit'].isin(unit_filter))
    ]

    st.dataframe(filtered_df, use_container_width=True)

# --- 3. Detail View ---
elif tab == "🔎 Detail View":
    st.subheader("Select a Finding to View Details")
    finding_id = st.selectbox("Choose Finding ID", df['finding_ID'].unique())

    record = df[df['finding_ID'] == finding_id].iloc[0]

    # Keyword highlighting function
    def highlight_keywords(text, keywords):
        pattern = '|'.join(re.escape(word.strip()) for word in keywords.split(','))
        return re.sub(f"\\b({pattern})\\b", r"**\1**", text, flags=re.IGNORECASE)

    st.markdown(f"### 📝 Finding ID: `{record['finding_ID']}`")
    st.markdown(f"**Assessment Unit:** {record['assessment_unit']}")
    st.markdown(f"**Severity Rating:** {record['severity_rating']}")
    st.markdown(f"**Systemic Risk Score:** `{record['systemic_risk_score']}`")
    st.markdown(f"**Matched Keywords:** `{record['matched_keywords']}`")

    st.markdown("#### 🔍 Finding Description")
    st.markdown(highlight_keywords(record['finding_description'], record['matched_keywords']))

    st.markdown("#### 🧠 LLM Generated Outputs")
    st.markdown(f"**Summary:** {record['finding_summary']}")
    st.markdown(f"**Suggestion:** {record['suggestion']}")
    st.markdown(f"**Risk Description:** {record['risk_description']}")
    st.markdown(f"**Recommendation:** {record['recommendation']}")

# --- 4. Visualizations ---
elif tab == "📈 Insights & Visualizations":
    st.subheader("Systemic Risk Score by Assessment Unit")
    chart1 = alt.Chart(df).mark_bar().encode(
    x='assessment_unit:N',
    y='systemic_risk_score:Q',
    color='assessment_unit:N'
    ).properties(width=600, height=400)
    st.altair_chart(chart1, use_container_width=True)

    st.subheader("Keyword Frequency")
    from collections import Counter
    all_keywords = ','.join(df['matched_keywords']).split(',')
    keyword_counts = Counter([kw.strip().lower() for kw in all_keywords])
    keyword_df = pd.DataFrame(keyword_counts.items(), columns=['keyword', 'count']).sort_values(by='count', ascending=False)

    chart2 = alt.Chart(keyword_df).mark_bar().encode(
    x='count:Q',
    y=alt.Y('keyword:N', sort='-x')
    ).properties(width=600, height=400)
    st.altair_chart(chart2, use_container_width=True)

# --- 5. Export ---
elif tab == "⬇️ Export":
    st.subheader("Download Filtered Findings")
    st.markdown("Use filters below to create a subset of data to export.")

    score_range = st.slider("Systemic Risk Score", 0, 2, (0, 2))
    severity_filter = st.multiselect("Severity Rating", df['severity_rating'].unique(), default=list(df['severity_rating'].unique()))
    unit_filter = st.multiselect("Assessment Unit", df['assessment_unit'].unique(), default=list(df['assessment_unit'].unique()))

    filtered_export_df = df[
    (df['systemic_risk_score'] >= score_range[0]) &
    (df['systemic_risk_score'] <= score_range[1]) &
    (df['severity_rating'].isin(severity_filter)) &
    (df['assessment_unit'].isin(unit_filter))
    ]

    st.dataframe(filtered_export_df, use_container_width=True)

    csv = filtered_export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name="filtered_systemic_findings.csv",
    mime="text/csv"
    )
FROM issue_finding_log;
