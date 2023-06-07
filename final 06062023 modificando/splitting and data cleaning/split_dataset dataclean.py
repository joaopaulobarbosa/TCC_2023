# @@@@@@@@@@@@@@@@@@@       DATA PREPROCESSING      @@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@       SPLITTING      @@@@@@@@@@@@@@@@@@@
from pathlib import Path
import warnings
import sys, os
from matplotlib import pyplot as plt

from sklearn.discriminant_analysis import StandardScaler

# # # # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
# # # if not sys.warnoptions:
# # #     warnings.simplefilter("ignore")
# # #     os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# https://towardsdatascience.com/how-to-split-a-dataframe-into-train-and-test-set-with-python-eaa1630ca7b3

# https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

from pathlib import Path
import warnings
import sys, os
from matplotlib import pyplot as plt

from sklearn.discriminant_analysis import StandardScaler
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame
coursera = pd.read_csv(
    "Coursera_original_university_name_fixed.csv", sep=";", encoding="latin-1"
)
# mudei para latin-1 porque com e sem utf-8 estava dando erro de "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe0 in position 9: invalid continuation byte"

# print(coursera.head())


def showDatasetCharacteristics(coursera, tittle):
    print(tittle, " \n")

    print(
        "Dataset columns: \n",
        "\t",
        coursera.columns,
        "\n",
    )
    # Display the initial dimensions of the dataset
    print(
        "Dataset dimensions (rows, columns) : \n",
        "\t",
        coursera.shape,
        "\n",
    )

    # Retrieve the unique classes in the "Difficulty_Level" column
    unique_classes = coursera["Difficulty_Level"].unique()
    print(
        "Unique classes: \n",
        "\t",
        unique_classes,
        "\n",
    )


showDatasetCharacteristics(coursera, "-----> Before text cleaning")


# Define the undesired classes to be removed
undesired_classes = ["Conversant", "Goldman Sachs", "Not Calibrated"]
# Remove rows with undesired classes in the "Difficulty_Level" column
coursera = coursera[~coursera["Difficulty_Level"].isin(undesired_classes)]

# Select the desired columns in the specified order
col = ["Difficulty_Level", "Skills", "University"]
coursera = coursera[col]


# Rename the columns for clarity
coursera.columns = ["Difficulty_Level", "Skills", "University"]


def removeRowsWithMissingValues(coursera):
    # Remove rows with missing values in the "Difficulty_Level" column
    coursera = coursera[pd.notnull(coursera["Difficulty_Level"])]

    # Remove rows with missing values in the "Skills" column
    coursera = coursera[pd.notnull(coursera["Skills"])]

    # Remove rows with boolean values in the "Skills" column
    coursera = coursera[coursera["Skills"].astype(bool)]

    return coursera


# Remove rows with missing values
coursera = removeRowsWithMissingValues(coursera)


# Convert "Skills" column to lowercase
coursera["Skills"] = coursera["Skills"].str.lower()

import re


# Remove punctuation and special characters
def remove_special_characters(text):
    # Use regular expressions to remove punctuation and special characters EXCEPT by the + character used in c++ Skills column.
    clean_text = re.sub(r"[^\w\s+]", "", text)
    return clean_text


# Remove punctuation and special characters
coursera["Skills"] = coursera["Skills"].apply(remove_special_characters)

""" # DISABLED THIS FUNCTION BECAUSE IT HAS A SIDE EFFECT OF REMOVE SAMPLES AS SKILLS "Industry 4.0"
# # # Remove numeric samples
# # def remove_numeric_samples(text):
# #     # Use regular expressions to check if the text contains numeric values
# #     if re.search(r"\d", text):
# #         return np.nan  # Replace numeric samples with NaN
# #     else:
# #         return text

# # # Remove numeric samples from the "Skills" column
# # coursera["Skills"] = coursera["Skills"].apply(remove_numeric_samples) """

# Remove rows with missing values again because of previous data cleaning genereted NaN "Skills" column samples
coursera = removeRowsWithMissingValues(coursera)

# Create a new column "category_id" and assign unique numeric values based on the "Difficulty_Level" column
coursera["category_id"] = coursera["Difficulty_Level"].factorize()[0]


showDatasetCharacteristics(coursera, "-----> After text cleaning")

# print(coursera.head())


def plotClassBalanceGraph(coursera):
    # Display the head of the cleaned dataset
    # print(coursera.head())
    print("CLASS BALANCE")
    # Plot a bar chart showing the count of skills per difficulty level
    fig = plt.figure(figsize=(8, 6))
    coursera.groupby("Difficulty_Level").Skills.count().plot.bar(ylim=0)
    plt.show()


plotClassBalanceGraph(coursera)


def calculateClassBalanceStatistics(coursera):
    # Count the number of occurrences for each class
    class_counts = coursera["Difficulty_Level"].value_counts()

    # Calculate the percentage of each class
    class_percentages = class_counts / class_counts.sum() * 100

    # Create a summary DataFrame with class counts and percentages
    class_imbalance = pd.DataFrame(
        {
            "Class": class_counts.index,
            "Count": class_counts,
            "Percentage": class_percentages,
        }
    )

    # Sort the DataFrame by class count in descending order
    class_imbalance = class_imbalance.sort_values("Count", ascending=False)

    # Print the class imbalance summary
    print("Class Imbalance Summary:")
    print(class_imbalance)


calculateClassBalanceStatistics(coursera)


# X are the features, y are the classes
# Split the dataset into train and test sets with stratified sampling, passing to stratify parameter the target variable that represents the class labels
# Stratified sampling ensures that the class distribution in the original dataset is preserved in the train and test splits. This can help maintain similar class imbalances between the two splits.
# By adding stratify=coursera["Difficulty_Level"], we ensure that the class distribution in the Difficulty_Level column is maintained in both the train and test sets. This helps in reducing the class imbalance during the split and ensures that the train and test splits have the most similar class imbalances.
train, test = train_test_split(
    coursera, test_size=0.2, random_state=0, stratify=coursera["Difficulty_Level"]
)

# When deciding on the test size, you should aim for a test set that is large enough to provide a reliable evaluation of your model's performance. However, it's also crucial to ensure that the test set retains the same class imbalance as the original dataset to reflect real-world scenarios accurately.

# determine the path where to save the train and test file
train_path = Path("./", "train.csv")
test_path = Path("./", "test.csv")

# save the train and test file
# again using the '\t' separator to create tab-separated-values files
train.to_csv(train_path, sep=";", index=False)
test.to_csv(test_path, sep=";", index=False)
