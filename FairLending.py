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






import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, precision_score, recall_score, confusion_matrix
from sklearn.utils import class_weight
import pandas as pd

# Assuming df is your pandas dataframe and "label" is the target column
X = df.drop('Class', axis=1)
y = df['Class']

# Split into train, validation, and test set (60%, 20%, 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Defining the xgboost model
clf = xgb.XGBClassifier(objective='binary:logistic')

# Defining the hyperparameters to tune
param_grid = { 
    'n_estimators': [ 10, 20],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# Creating the grid search object
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                           scoring='f1', n_jobs=-1, cv=StratifiedKFold(n_splits=10), verbose=3)

# Fitting grid search to the train data
grid_search.fit(X_train, y_train)

# Evaluating the model's performance
y_train_pred = grid_search.predict(X_train)
y_val_pred = grid_search.predict(X_val)
y_test_pred = grid_search.predict(X_test)

# print("Training Metrics:")
# print(classification_report(y_train, y_train_pred))
# print("Validation Metrics:")
# print(classification_report(y_val, y_val_pred))
# print("Testing Metrics:")
# print(classification_report(y_test, y_test_pred))

print("Best Parameters:")
print(grid_search.best_params_)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Your existing code goes here ...

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(cm, columns=['Predicted Non-Fraud', 'Predicted Fraud'], index=['Actual Non-Fraud', 'Actual Fraud'])

    # Adding row and column totals
    cm['Row Total'] = cm.sum(axis=1)
    cm.loc['Column Total'] = cm.sum(axis=0)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues',annot_kws={"size": 10})
    plt.title(title)
    plt.show()
    
# plot confusion matrix for training dataset
plot_confusion_matrix(y_train, y_train_pred, "Training Confusion Matrix")

# plot confusion matrix for validation dataset
plot_confusion_matrix(y_val, y_val_pred, "Validation Confusion Matrix")


# plot confusion matrix for testing dataset
plot_confusion_matrix(y_test, y_test_pred, "Testing Confusion Matrix")
