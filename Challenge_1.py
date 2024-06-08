# Databricks notebook source
# If using scikit-learn to load the dataset directly
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target
data['target_names'] = data['target'].map(dict(enumerate(iris.target_names)))

# Display the first few rows of the dataframe
display(data.head())


# COMMAND ----------

data

# COMMAND ----------

# Import necessary libraries
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = [iris.target_names[i] for i in iris.target]

# Pairplot to visually inspect species similarity
sns.pairplot(data, hue='species', markers=["o", "s", "D"])
plt.title("Pair Plot of Iris Dataset")
display()  # Use display() for showing plots in Databricks


# COMMAND ----------

# MAGIC %md
# MAGIC So we can determine from the above graph that Setosa is quite distinct from Versicolor and Virginica across all feature combinations; it's typically separated with little to no overlap.
# MAGIC 1. Versicolor and Virginica show more overlap in feature space, especially for "petal length" and "petal width." Their distributions are more similar to each other than to Setosa.
# MAGIC Specifically, the scatter plots for "petal length" vs. "petal width" and the corresponding histograms show significant overlap between Versicolor and Virginica.
# MAGIC Based on this visual inspection, Versicolor and Virginica are the two species that are the most similar to each other.

# COMMAND ----------

# Assume the similar species are Versicolor and Virginica
data['combined_species'] = data['species'].replace({'versicolor': 'Class4', 'virginica': 'Class4'})

# Display the first few rows to verify
display(data.head())


# COMMAND ----------

# Check for the existence of the new column
assert 'combined_species' in data.columns, "The combined_species column does not exist."

# Check that only two unique class labels exist
assert data['combined_species'].nunique() == 2, "There should only be two unique class labels."

# Check the count of each class label
print(data['combined_species'].value_counts())

# Display a small random sample of the data to manually inspect
display(data.sample(5))


# COMMAND ----------

from sklearn.model_selection import train_test_split

# Assuming 'data' is your DataFrame and 'combined_species' is the column with the new class labels
X = data.drop(['species', 'combined_species'], axis=1)  # Drop the original species and the combined_species columns to get the features
y = data['combined_species']  # The target is now the combined species column

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# COMMAND ----------

from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Function to train and evaluate a model
def train_eval_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

# Initialize models
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)
logistic_regression = LogisticRegression(random_state=42)

# Train and evaluate models
models = [decision_tree, random_forest, svc, logistic_regression]
for model in models:
    accuracy, report = train_eval_model(model, X_train, y_train, X_test, y_test)
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy}")
    print(report)


# COMMAND ----------

from sklearn.metrics import precision_score, recall_score, f1_score

# Function to gather performance metrics
def get_performance_metrics(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label='Class4')
    recall = recall_score(y_test, predictions, pos_label='Class4')
    f1 = f1_score(y_test, predictions, pos_label='Class4')
    return accuracy, precision, recall, f1

# Dictionary to hold model names and their performance metrics
performance_dict = {}

# Evaluate each model and add its performance to the dictionary
for model in models:
    accuracy, precision, recall, f1 = get_performance_metrics(model, X_train, y_train, X_test, y_test)
    performance_dict[model.__class__.__name__] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Convert the performance dictionary into a DataFrame for nice formatting
performance_df = pd.DataFrame(performance_dict).transpose()

# Display the performance DataFrame
display(performance_df)


# COMMAND ----------


