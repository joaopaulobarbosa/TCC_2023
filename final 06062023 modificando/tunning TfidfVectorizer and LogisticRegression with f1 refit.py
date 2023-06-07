import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    accuracy_score,
)

start = time.time()
# Load the dataset
data = pd.read_csv("train.csv", sep=";", encoding="latin-1")
df = pd.DataFrame(data)

# Encode the target labels to numerical values
label_encoder = LabelEncoder()
df["Difficulty_Level_Encoded"] = label_encoder.fit_transform(df["Difficulty_Level"])

# Extract the features (Skills) and labels (Difficulty_Level_Encoded)
X = df["Skills"] + " " + df["University"]
y = df["Difficulty_Level_Encoded"]

# # Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=32)

# Define the parameter grid for the TfidfVectorizer
param_grid = {
    # input is expected to be a sequence of items that can be of type string or byte.
    "tfidf__input": ["content"],
    "tfidf__encoding": ["latin-1"],
    "tfidf__strip_accents": ["unicode"],
    "tfidf__lowercase": [True],
    "tfidf__stop_words": ["english"],
    "tfidf__ngram_range": [(1, 1), (1, 2), (2, 2)],  # Adjust the n-gram range
    "tfidf__sublinear_tf": [True, False],
    "tfidf__min_df": [1, 5, 10],  # Adjust the minimum document frequency
    # Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    "clf__C": [0.001, 0.01, 0.1, 1, 10],  # Adjust the regularization parameter C
    "clf__class_weight": ["balanced"],  # Adjust the class weights
    "clf__fit_intercept": [True, False],  # Adjust the intercept fitting
    "clf__multi_class": ["ovr"],  # Adjust the multi-class strategy
    # 'elasticnet': both L1 and L2 penalty terms are added.
    # Adjust the regularization penalty
    # não usei o "newton-cg" por causa do consumo de memória
    # não usei "sag", "saga" para não precisar preprocessar ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    "clf__penalty": ["l1", "l2", "elasticnet"],
    "clf__solver": ["liblinear", "lbfgs"],  # Adjust the solver algorithm
    "clf__n_jobs": [-1],  # Set the number of parallel jobs
    "clf__verbose": [1],  # Enable verbose output
}

# Create an instance of the TfidfVectorizer
vectorizer = TfidfVectorizer()
# X_train_transformed = vectorizer.fit_transform(X)

# Create the pipeline with the TfidfVectorizer and Random Forest Classifier
pipeline = Pipeline(
    [
        ("tfidf", vectorizer),
        (
            "clf",
            LogisticRegression(),
        ),
    ]
)
# usei o LogisticRegression e não o LogisticRegressionCV porque já vou fazer o cross-validation no proprio gridSearchCV https://stackoverflow.com/questions/46507606/what-does-the-cv-stand-for-in-sklearn-linear-model-logisticregressioncv

meus_scores = {
    "accuracy": make_scorer(accuracy_score),
    "recall": make_scorer(recall_score, average="weighted"),
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    # foi adicionado o weighted por causa do desbalanceamento das classes
    # it can result in an F-score that is not between precision and recall
    "precision": make_scorer(precision_score, average="weighted"),
    # The beta parameter determines the weight of recall in the combined score. beta < 1 lends more weight to precision, while beta > 1 favors recall
    "f1": make_scorer(f1_score, average="weighted", beta=1),
}

# Perform cross-validation for each model using the same folds
# unnecessary in this case
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

# Otimized to f1 score only
# Perform grid search to find the best parameters
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring=meus_scores["f1"],
    refit="f1",
    cv=skf,
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X, y)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)

end = time.time()

# print("RESULTS GridSearchCV \n".join(map(str, results)))

file_path = "results_tunning_tfidfvectorizer.txt"

# Open the file in write mode
with open(file_path, "w") as file:
    # Write the contents of results array to the file
    file.write(str(grid_search.best_params_))
    file.write("\n\n")
    file.write(str(grid_search.best_score_))
    file.write("\n\n")
    file.write(
        str(
            "The tunning of TfidfVectorizer with LogisticRegression process took: ",
            end - start,
        )
    )

print("Finished writing file")

# # # # # https://datascience.stackexchange.com/questions/91225/why-gridsearchcv-returns-nan
# # # # # GridSearchCV provides a score of nan when fitting the model fails
