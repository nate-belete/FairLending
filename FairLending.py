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

















import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import opinion_lexicon
from nltk.corpus import stopwords

nltk.download('opinion_lexicon')
nltk.download('punkt')
nltk.download('stopwords')



# Define filler words 
filler_words = set(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())


df = pd.read_csv('complaints.csv')

df = df[['Date received', 'Consumer complaint narrative']].dropna()
df.columns = ['date', 'complaint']


# Define a function to count all cap texts
def count_all_cap_words(text):
    return len(re.findall(r'\b[A-Z]{2,}\b', text))

# Define a function to count mentions of law, regulation, or reg
def count_law_reg(text):
    return len(re.findall(r'\b(law|regulation|reg)\b', text, re.IGNORECASE))

# Define a function to count filler words
def count_filler_words(text):
    return len([word for word in nltk.word_tokenize(text) if word.lower() in filler_words])

def calculate_polarity(text):
    return TextBlob(text).sentiment.polarity

def calculate_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def count_positive_words(text):
    return sum(word.lower() in positive_words for word in nltk.word_tokenize(text))

def count_negative_words(text):
    return sum(word.lower() in negative_words for word in nltk.word_tokenize(text))

def count_exclamation_marks(text):
    return text.count('!')

def count_question_marks(text):
    return text.count('?')

def count_words(text):
    return len(str(text).split(" "))

def avg_word_length(text):
    words = str(text).split()
    return sum(len(word) for word in words) / len(words)

def count_stopwords(text):
    return len([word for word in str(text).split() if word in stop_words])

def count_numerics(text):
    return len([word for word in str(text).split() if word.isdigit()])


# Text length-based features
df['characters_used'] = df['complaint'].apply(len)
df['word_count'] = df['complaint'].apply(count_words)
df['avg_word_length'] = df['complaint'].apply(avg_word_length)

# Capitalization related features
df['all_cap_words'] = df['complaint'].apply(count_all_cap_words)

# Special character related features
df['num_exclamation_marks'] = df['complaint'].apply(count_exclamation_marks)
df['num_question_marks'] = df['complaint'].apply(count_question_marks)

# Specific word related features
df['law_reg_mentions'] = df['complaint'].apply(count_law_reg)
df['filler_words_used'] = df['complaint'].apply(count_filler_words)
df['positive_word_count'] = df['complaint'].apply(count_positive_words)
df['negative_word_count'] = df['complaint'].apply(count_negative_words)
df['stopwords'] = df['complaint'].apply(count_stopwords)
df['numerics'] = df['complaint'].apply(count_numerics)

# Sentiment related features
df['polarity'] = df['complaint'].apply(calculate_polarity)
df['subjectivity'] = df['complaint'].apply(calculate_subjectivity)
