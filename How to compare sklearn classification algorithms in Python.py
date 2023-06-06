# # # # https://www.projectpro.io/recipes/compare-sklearn-classification-algorithms-in-python

# # # import warnings
# # # import sys, os
# # # from matplotlib import pyplot as plt
# # # from sklearn.metrics import (
# # #     fbeta_score,
# # #     make_scorer,
# # #     precision_score,
# # #     recall_score,
# # #     accuracy_score,
# # # )
# # # from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
# # # from sklearn.feature_extraction.text import CountVectorizer
# # # from sklearn.feature_extraction.text import TfidfTransformer
# # # from sklearn.naive_bayes import MultinomialNB
# # # from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, StandardScaler

# # # # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
# # # if not sys.warnoptions:
# # #     warnings.simplefilter("ignore")
# # #     os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

# # # import numpy as np
# # # import pandas as pd

# # # coursera_train = pd.read_csv("train.csv", sep=";")
# # # X_train = coursera_train["Skills"]
# # # y_train = coursera_train["Difficulty_Level"]

# # # coursera_test = pd.read_csv("test.csv", sep=";")
# # # X_test = coursera_test["Skills"]
# # # y_test = coursera_test["Difficulty_Level"]


# # # from sklearn.feature_extraction.text import TfidfVectorizer

# # # tfidf = TfidfVectorizer(
# # #     sublinear_tf=True,
# # #     min_df=2,
# # #     norm="l2",
# # #     encoding="latin-1",
# # #     ngram_range=(1, 2),
# # #     stop_words="english",
# # # )


# # # # Convert a collection of text documents to a matrix of token counts
# # # count_vect = CountVectorizer(stop_words="english")

# # # # Learn the vocabulary dictionary and return document-term matrix. This is equivalent to fit followed by transform, but more efficiently implemented.
# # # X_train_counts = count_vect.fit_transform(X_train)
# # # # Transform a count matrix to a normalized tf or tf-idf representation. Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. This is a common term weighting scheme in information retrieval, that has also found good use in document classification. The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.
# # # tfidf_transformer = TfidfTransformer()
# # # # Fit to data, then transform it. Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.
# # # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# # # # clf = MultinomialNB().fit(X_train_tfidf, y_train)
# # # # print(clf.predict(count_vect.transform(["physics"])))

# # # # END FROM https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

# # # # ATÉ AQUI, DUPLICADO EM TODOS OS ARQUIVOS DE TUNNING.


""" CODIGO FINAL QUE COLOQUEI NO RELATÓRIO EM 05062023 """
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

coursera_train = pd.read_csv("train.csv", sep=";")

df = pd.DataFrame(coursera_train)

# Encode the target labels to numerical values
label_encoder = LabelEncoder()
df["Difficulty_Level_Encoded"] = label_encoder.fit_transform(df["Difficulty_Level"])

# Extract the features (Skills) and labels (Difficulty_Level_Encoded)
X = df["Skills"]
y = df["Difficulty_Level_Encoded"]

models = [
    RandomForestClassifier(
        ccp_alpha=0.0,
        class_weight="balanced",
        criterion="gini",
        max_depth=None,
        max_features=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=400,
        n_jobs=-1,
        verbose=1,
    ),
    LogisticRegression(
        C=1,
        class_weight="balanced",
        fit_intercept=True,
        multi_class="ovr",
        n_jobs=-1,
        penalty="l2",
        solver="newton-cg",
        verbose=1,
    ),
    KNeighborsClassifier(
        algorithm="auto",
        leaf_size=10,
        metric="euclidean",
        n_jobs=-1,
        n_neighbors=9,
        p=2,
        weights="distance",
    ),
    # Add more models here if needed
]


# Create an instance of the TfidfVectorizer to convert text data into numerical features
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Specify the evaluation metric(s) you want to use
# scoring = "roc_auc_ovr"

# Perform cross-validation for each model using the same folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

# Initialize lists to store performance measures
model_names = []
roc_auc_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
macro_avg_roc_auc_scores = []
class_roc_auc_scores = []
weighted_avg_roc_auc_scores = []


# Perform cross-validation and calculate performance measures for each model
for model in models:
    # Perform cross-validation and obtain predicted probabilities
    y_proba = cross_val_predict(model, X_transformed, y, cv=skf, method="predict_proba")
    y_pred = np.argmax(y_proba, axis=1)

    # Calculate ROC AUC scores
    roc_auc_scores.append(
        roc_auc_score(
            pd.get_dummies(y).values,
            y_proba,
            multi_class="ovr",
        )
    )

    # Calculate accuracy
    accuracy_scores.append(accuracy_score(y, y_pred))

    # Calculate precision
    precision_scores.append(precision_score(y, y_pred, average="weighted"))

    # Calculate recall
    recall_scores.append(recall_score(y, y_pred, average="weighted"))

    # Calculate F1 score
    f1_scores.append(f1_score(y, y_pred, average="weighted"))

    # Calculate individual class ROC AUC scores
    class_roc_auc_scores.append(
        roc_auc_score(
            pd.get_dummies(y).values, y_proba, multi_class="ovr", average=None
        )
    )

    # Calculate macro-average ROC AUC
    macro_avg_roc_auc_scores.append(
        roc_auc_score(
            pd.get_dummies(y).values, y_proba, multi_class="ovr", average="macro"
        )
    )

    # Calculate weighted-average ROC AUC
    weighted_avg_roc_auc_scores.append(
        roc_auc_score(
            pd.get_dummies(y).values, y_proba, multi_class="ovr", average="weighted"
        )
    )

    # Append model name to list
    model_names.append(model.__class__.__name__)

# Create a DataFrame to store the performance measures
performance_df = pd.DataFrame(
    {
        "Model": model_names,
        "ROC AUC Scores": roc_auc_scores,
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
        "Recall": recall_scores,
        "F1 Score": f1_scores,
        "Macro-average ROC AUC": macro_avg_roc_auc_scores,
        "Weighted-average ROC AUC": weighted_avg_roc_auc_scores,
    }
)

# Print the performance table
print("Performance Measures:")
print(performance_df)
print()

# Create a DataFrame for the individual class ROC AUC scores
class_roc_auc_df = pd.DataFrame(class_roc_auc_scores, columns=label_encoder.classes_)
class_roc_auc_df["Model"] = model_names
class_roc_auc_df.set_index("Model", inplace=True)

# Print the individual class ROC AUC scores table
print("Individual Class ROC AUC Scores:")
print(class_roc_auc_df)
print()

# Plot the performance graph
plt.figure(figsize=(10, 8))
x = np.arange(len(model_names))
width = 0.2


plt.bar(x, roc_auc_scores, width, label="ROC AUC Scores")
plt.bar(x + width, accuracy_scores, width, label="Accuracy")
plt.bar(x + 2 * width, precision_scores, width, label="Precision")
plt.bar(x + 3 * width, recall_scores, width, label="Recall")
plt.bar(x + 4 * width, f1_scores, width, label="F1 Score")

plt.xlabel("Models")
plt.ylabel("Performance")
plt.title("Performance Comparison of Multiple Models")
plt.xticks(x + 2 * width, model_names, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Plot the ROC AUC curves in a single graph
plt.figure(figsize=(10, 8))
for model in models:
    y_proba = cross_val_predict(model, X_transformed, y, cv=skf, method="predict_proba")
    fpr, tpr, _ = roc_curve(pd.get_dummies(y).values.ravel(), y_proba.ravel())
    roc_auc = roc_auc_score(
        pd.get_dummies(y).values, y_proba, multi_class="ovr", average="weighted"
    )
    plt.plot(fpr, tpr, label=f"{model.__class__.__name__} (ROC AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="r")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC Curves for Multiple Models")
plt.legend()
plt.tight_layout()
plt.show()
