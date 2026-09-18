import os
import json

import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

CLASSICAL_FILE = os.path.join(
    MODEL_DIR,
    "classical_predictions.csv"
)

CLINICALBERT_FILE = os.path.join(
    MODEL_DIR,
    "clinicalbert_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    MODEL_DIR,
    "model_predictions.csv"
)

CONFUSION_FILE = os.path.join(
    MODEL_DIR,
    "confusion_matrices.json"
)


# =========================================================
# DOCUMENTATION CATEGORIES
# =========================================================

LABELS = [
    "assessment",
    "chief_complaint",
    "history_present_illness",
    "medications",
    "plan"
]


# =========================================================
# MODEL PREDICTION COLUMNS
# =========================================================

MODEL_COLUMNS = {
    "Logistic Regression":
        "logistic_prediction",

    "SVM":
        "svm_prediction",

    "ClinicalBERT":
        "clinicalbert_prediction"
}


# =========================================================
# START
# =========================================================

print()
print("=" * 65)
print("ScribeDr AI - Evaluation Builder")
print("=" * 65)
print()


# =========================================================
# CHECK REQUIRED FILES
# =========================================================

if not os.path.exists(CLASSICAL_FILE):

    print(
        "ERROR: classical_predictions.csv "
        "was not found."
    )

    print(CLASSICAL_FILE)

    raise SystemExit(1)


if not os.path.exists(CLINICALBERT_FILE):

    print(
        "ERROR: clinicalbert_predictions.csv "
        "was not found."
    )

    print(CLINICALBERT_FILE)

    raise SystemExit(1)


# =========================================================
# LOAD PREDICTIONS
# =========================================================

print("Loading prediction files...")

classical_df = pd.read_csv(
    CLASSICAL_FILE
)

bert_df = pd.read_csv(
    CLINICALBERT_FILE
)


print(
    f"Classical prediction rows: "
    f"{len(classical_df):,}"
)

print(
    f"ClinicalBERT prediction rows: "
    f"{len(bert_df):,}"
)

print()


# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

classical_required = {
    "source_id",
    "text",
    "true_label",
    "logistic_prediction",
    "svm_prediction"
}

bert_required = {
    "source_id",
    "text",
    "true_label",
    "clinicalbert_prediction"
}


if not classical_required.issubset(
    classical_df.columns
):

    print(
        "ERROR: classical_predictions.csv "
        "is missing required columns."
    )

    print(
        "Available columns:"
    )

    print(
        classical_df.columns.tolist()
    )

    raise SystemExit(1)


if not bert_required.issubset(
    bert_df.columns
):

    print(
        "ERROR: clinicalbert_predictions.csv "
        "is missing required columns."
    )

    print(
        "Available columns:"
    )

    print(
        bert_df.columns.tolist()
    )

    raise SystemExit(1)


# =========================================================
# RESET ROW INDEX
# =========================================================

classical_df = (
    classical_df
    .reset_index(drop=True)
)

bert_df = (
    bert_df
    .reset_index(drop=True)
)


classical_df[
    "test_row_id"
] = classical_df.index


bert_df[
    "test_row_id"
] = bert_df.index


# =========================================================
# VERIFY SAME TEST SET
# =========================================================

if len(classical_df) != len(bert_df):

    raise RuntimeError(
        "Prediction files contain different "
        "numbers of test examples."
    )


source_match = (
    classical_df["source_id"]
    .astype(str)
    .reset_index(drop=True)
    .equals(
        bert_df["source_id"]
        .astype(str)
        .reset_index(drop=True)
    )
)


label_match = (
    classical_df["true_label"]
    .astype(str)
    .reset_index(drop=True)
    .equals(
        bert_df["true_label"]
        .astype(str)
        .reset_index(drop=True)
    )
)


text_match = (
    classical_df["text"]
    .astype(str)
    .reset_index(drop=True)
    .equals(
        bert_df["text"]
        .astype(str)
        .reset_index(drop=True)
    )
)


print("=" * 65)
print("TEST SET VALIDATION")
print("=" * 65)
print()

print(
    "Same number of rows:",
    len(classical_df) == len(bert_df)
)

print(
    "Source IDs match:",
    source_match
)

print(
    "True labels match:",
    label_match
)

print(
    "Clinical text matches:",
    text_match
)

print()


if not (
    source_match
    and label_match
    and text_match
):

    raise RuntimeError(
        "The prediction files do not represent "
        "the exact same ordered test set."
    )


# =========================================================
# BUILD MASTER PREDICTION DATASET
# =========================================================

master_df = classical_df[
    [
        "test_row_id",
        "source_id",
        "text",
        "true_label",
        "logistic_prediction",
        "svm_prediction"
    ]
].copy()


master_df[
    "clinicalbert_prediction"
] = bert_df[
    "clinicalbert_prediction"
].values


# =========================================================
# ADD CORRECT / INCORRECT FLAGS
# =========================================================

master_df[
    "logistic_correct"
] = (
    master_df["true_label"]
    ==
    master_df["logistic_prediction"]
)


master_df[
    "svm_correct"
] = (
    master_df["true_label"]
    ==
    master_df["svm_prediction"]
)


master_df[
    "clinicalbert_correct"
] = (
    master_df["true_label"]
    ==
    master_df["clinicalbert_prediction"]
)


# =========================================================
# SAVE MASTER PREDICTION FILE
# =========================================================

master_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Master prediction file created:"
)

print(
    OUTPUT_FILE
)

print()


# =========================================================
# GENERATE CONFUSION MATRICES
# =========================================================

confusion_data = {}


print("=" * 65)
print("CONFUSION MATRICES")
print("=" * 65)
print()


for model_name, prediction_column in MODEL_COLUMNS.items():

    # -----------------------------------------------------
    # Generate matrix
    # -----------------------------------------------------

    matrix = confusion_matrix(
        master_df["true_label"],
        master_df[prediction_column],
        labels=LABELS
    )


    # -----------------------------------------------------
    # Calculate accuracy
    # -----------------------------------------------------

    accuracy = accuracy_score(
        master_df["true_label"],
        master_df[prediction_column]
    )


    # -----------------------------------------------------
    # Correct and incorrect predictions
    # -----------------------------------------------------

    correct = int(
        (
            master_df["true_label"]
            ==
            master_df[prediction_column]
        ).sum()
    )


    incorrect = int(
        len(master_df)
        -
        correct
    )


    # -----------------------------------------------------
    # Find classification errors
    # -----------------------------------------------------

    errors = []


    for true_index, true_label in enumerate(
        LABELS
    ):

        for predicted_index, predicted_label in enumerate(
            LABELS
        ):

            # Skip diagonal:
            # diagonal values are correct predictions.
            if true_index == predicted_index:
                continue


            count = int(
                matrix[
                    true_index,
                    predicted_index
                ]
            )


            if count > 0:

                errors.append(
                    {
                        "true_label":
                            true_label,

                        "predicted_label":
                            predicted_label,

                        "count":
                            count
                    }
                )


    # -----------------------------------------------------
    # Sort errors from most frequent to least frequent
    # -----------------------------------------------------

    errors = sorted(
        errors,
        key=lambda item: item["count"],
        reverse=True
    )


    top_errors = errors[:5]


    # -----------------------------------------------------
    # Store matrix data
    # -----------------------------------------------------

    confusion_data[
        model_name
    ] = {

        "labels":
            LABELS,

        "matrix":
            matrix.tolist(),

        "correct_predictions":
            correct,

        "incorrect_predictions":
            incorrect,

        "total_predictions":
            int(len(master_df)),

        "accuracy":
            float(accuracy),

        "top_errors":
            top_errors
    }


    # -----------------------------------------------------
    # Display matrix in terminal
    # -----------------------------------------------------

    print(model_name)
    print("-" * 65)

    print(
        pd.DataFrame(
            matrix,
            index=LABELS,
            columns=LABELS
        )
    )

    print()

    print(
        f"Correct:   {correct}"
    )

    print(
        f"Incorrect: {incorrect}"
    )

    print(
        f"Accuracy:  {accuracy:.3f}"
    )

    print()

    print(
        "Top classification errors:"
    )


    if top_errors:

        for error in top_errors:

            print(
                f"  {error['true_label']} "
                f"-> "
                f"{error['predicted_label']}: "
                f"{error['count']}"
            )

    else:

        print(
            "  No classification errors."
        )

    print()


# =========================================================
# SAVE CONFUSION MATRIX DATA
# =========================================================

with open(
    CONFUSION_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        confusion_data,
        file,
        indent=4
    )


# =========================================================
# MASTER DATASET VALIDATION
# =========================================================

print("=" * 65)
print("MASTER DATASET VALIDATION")
print("=" * 65)
print()


print(
    f"Master test rows: "
    f"{len(master_df):,}"
)


print(
    f"Unique source records: "
    f"{master_df['source_id'].nunique():,}"
)

print()


print(
    "True-label distribution:"
)


print(
    master_df[
        "true_label"
    ]
    .value_counts()
    .sort_index()
)

print()


# =========================================================
# FINAL SUMMARY
# =========================================================

print("=" * 65)
print("EVALUATION BUILD COMPLETE")
print("=" * 65)
print()


print(
    "Created:"
)

print(
    OUTPUT_FILE
)

print(
    CONFUSION_FILE
)

print()


print(
    "The confusion matrix file now contains:"
)

print(
    "- Matrix values"
)

print(
    "- Accuracy"
)

print(
    "- Correct prediction count"
)

print(
    "- Incorrect prediction count"
)

print(
    "- Five most common errors per model"
)

print()


print(
    "All three models were evaluated "
    "against the same permanent test set."
)

print()