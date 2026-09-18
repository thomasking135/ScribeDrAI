import json
import os

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "train_split.csv"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "test_split.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

print()
print("=" * 65)
print("ScribeDr AI - Classical Model Training")
print("=" * 65)
print()


# ---------------------------------------------------------
# Load permanent experiment split
# ---------------------------------------------------------

print(
    "Loading permanent train/test split..."
)

print()


if not os.path.exists(TRAIN_FILE):

    print(
        "ERROR: train_split.csv not found."
    )

    print(
        "Expected location:"
    )

    print(TRAIN_FILE)

    raise SystemExit(1)


if not os.path.exists(TEST_FILE):

    print(
        "ERROR: test_split.csv not found."
    )

    print(
        "Expected location:"
    )

    print(TEST_FILE)

    raise SystemExit(1)


train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------

required_columns = {
    "source_id",
    "text",
    "label"
}


if not required_columns.issubset(
    train_df.columns
):

    print(
        "ERROR: train_split.csv does not "
        "contain the required columns."
    )

    raise SystemExit(1)


if not required_columns.issubset(
    test_df.columns
):

    print(
        "ERROR: test_split.csv does not "
        "contain the required columns."
    )

    raise SystemExit(1)


# ---------------------------------------------------------
# Clean data
# ---------------------------------------------------------

train_df = train_df.dropna(
    subset=[
        "source_id",
        "text",
        "label"
    ]
).copy()


test_df = test_df.dropna(
    subset=[
        "source_id",
        "text",
        "label"
    ]
).copy()


train_df["text"] = (
    train_df["text"]
    .astype(str)
)


test_df["text"] = (
    test_df["text"]
    .astype(str)
)


# ---------------------------------------------------------
# Validate source isolation
# ---------------------------------------------------------

train_sources = set(
    train_df["source_id"]
)

test_sources = set(
    test_df["source_id"]
)


overlap = (
    train_sources
    &
    test_sources
)


print(
    f"Training sections: "
    f"{len(train_df):,}"
)

print(
    f"Testing sections:  "
    f"{len(test_df):,}"
)

print()

print(
    f"Training source records: "
    f"{len(train_sources):,}"
)

print(
    f"Testing source records:  "
    f"{len(test_sources):,}"
)

print()

print(
    "Source overlap between "
    "train and test:",
    len(overlap)
)


if overlap:

    raise RuntimeError(
        "Train/test source leakage detected."
    )


print()


# ---------------------------------------------------------
# Display class distributions
# ---------------------------------------------------------

print("=" * 65)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 65)
print()

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)

print()


print("=" * 65)
print("TEST CLASS DISTRIBUTION")
print("=" * 65)
print()

print(
    test_df["label"]
    .value_counts()
    .sort_index()
)

print()


# ---------------------------------------------------------
# Prepare X and y
# ---------------------------------------------------------

X_train = train_df[
    "text"
]

y_train = train_df[
    "label"
]


X_test = test_df[
    "text"
]

y_test = test_df[
    "label"
]


# ---------------------------------------------------------
# TF-IDF feature extraction
# ---------------------------------------------------------

print("=" * 65)
print("TF-IDF FEATURE EXTRACTION")
print("=" * 65)
print()


vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    ngram_range=(1, 2),

    max_features=30000,

    min_df=2,

    sublinear_tf=True
)


X_train_vectorised = (
    vectorizer.fit_transform(
        X_train
    )
)


X_test_vectorised = (
    vectorizer.transform(
        X_test
    )
)


print(
    "TF-IDF features:",
    X_train_vectorised.shape[1]
)

print()


# ---------------------------------------------------------
# Define models
# ---------------------------------------------------------

models = {

    "Logistic Regression":

        LogisticRegression(

            max_iter=3000,

            class_weight="balanced",

            random_state=42
        ),


    "SVM":

        LinearSVC(

            class_weight="balanced",

            random_state=42
        )
}


metrics = {}

reports = {}

prediction_results = test_df[
    [
        "source_id",
        "text",
        "label"
    ]
].copy()


prediction_results = (
    prediction_results.rename(
        columns={
            "label": "true_label"
        }
    )
)


# ---------------------------------------------------------
# Train and evaluate models
# ---------------------------------------------------------

for name, model in models.items():

    print()
    print("=" * 65)
    print(name)
    print("=" * 65)
    print()


    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    model.fit(
        X_train_vectorised,
        y_train
    )


    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    predictions = model.predict(
        X_test_vectorised
    )


    # -----------------------------------------------------
    # Save individual predictions
    # -----------------------------------------------------

    if name == "Logistic Regression":

        prediction_results[
            "logistic_prediction"
        ] = predictions


    elif name == "SVM":

        prediction_results[
            "svm_prediction"
        ] = predictions


    # -----------------------------------------------------
    # Macro metrics
    # -----------------------------------------------------

    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            y_test,

            predictions,

            average="macro",

            zero_division=0
        )
    )


    # -----------------------------------------------------
    # Weighted metrics
    # -----------------------------------------------------

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _
    ) = precision_recall_fscore_support(

        y_test,

        predictions,

        average="weighted",

        zero_division=0
    )


    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    # -----------------------------------------------------
    # Store metrics
    # -----------------------------------------------------

    metrics[name] = {

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "weighted_precision":
            float(weighted_precision),

        "weighted_recall":
            float(weighted_recall),

        "weighted_f1":
            float(weighted_f1),

        "accuracy":
            float(accuracy)
    }


    # -----------------------------------------------------
    # Detailed report
    # -----------------------------------------------------

    report = classification_report(

        y_test,

        predictions,

        output_dict=True,

        zero_division=0
    )


    reports[
        name
    ] = report


    # -----------------------------------------------------
    # Terminal report
    # -----------------------------------------------------

    print(
        classification_report(

            y_test,

            predictions,

            zero_division=0
        )
    )


    print(
        f"Macro Precision: "
        f"{precision:.3f}"
    )

    print(
        f"Macro Recall:    "
        f"{recall:.3f}"
    )

    print(
        f"Macro F1-score:  "
        f"{f1:.3f}"
    )

    print(
        f"Accuracy:        "
        f"{accuracy:.3f}"
    )


# ---------------------------------------------------------
# Save TF-IDF vectorizer
# ---------------------------------------------------------

joblib.dump(

    vectorizer,

    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.joblib"
    )
)


# ---------------------------------------------------------
# Save Logistic Regression
# ---------------------------------------------------------

joblib.dump(

    models[
        "Logistic Regression"
    ],

    os.path.join(
        MODEL_DIR,
        "logistic_regression.joblib"
    )
)


# ---------------------------------------------------------
# Save SVM
# ---------------------------------------------------------

joblib.dump(

    models[
        "SVM"
    ],

    os.path.join(
        MODEL_DIR,
        "svm.joblib"
    )
)


# ---------------------------------------------------------
# Preserve ClinicalBERT metrics if they already exist
# ---------------------------------------------------------

METRICS_FILE = os.path.join(
    MODEL_DIR,
    "metrics.json"
)


if os.path.exists(
    METRICS_FILE
):

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        existing_metrics = json.load(
            file
        )


    if "ClinicalBERT" in existing_metrics:

        metrics[
            "ClinicalBERT"
        ] = existing_metrics[
            "ClinicalBERT"
        ]


# ---------------------------------------------------------
# Save evaluation metrics
# ---------------------------------------------------------

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ---------------------------------------------------------
# Save classification reports
# ---------------------------------------------------------

REPORTS_FILE = os.path.join(
    MODEL_DIR,
    "classification_reports.json"
)


with open(
    REPORTS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        reports,
        file,
        indent=4
    )


# ---------------------------------------------------------
# Save classical model predictions
# ---------------------------------------------------------

PREDICTIONS_FILE = os.path.join(
    MODEL_DIR,
    "classical_predictions.csv"
)


prediction_results.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print()
print("=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)
print()

print(
    "Permanent test set used:"
)

print(
    TEST_FILE
)

print()

print(
    "Source overlap: 0"
)

print()

print("Saved:")

print(
    "models/tfidf_vectorizer.joblib"
)

print(
    "models/logistic_regression.joblib"
)

print(
    "models/svm.joblib"
)

print(
    "models/metrics.json"
)

print(
    "models/classification_reports.json"
)

print(
    "models/classical_predictions.csv"
)

print()

print(
    "Logistic Regression and SVM "
    "training complete."
)

print()