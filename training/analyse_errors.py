import os
import json

import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support
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

PREDICTIONS_FILE = os.path.join(
    MODEL_DIR,
    "model_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    MODEL_DIR,
    "error_analysis.json"
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
# HELPER FUNCTIONS
# =========================================================

def readable_label(label):
    """
    Convert internal labels into readable names.
    """

    names = {
        "assessment":
            "Assessment",

        "chief_complaint":
            "Chief Complaint",

        "history_present_illness":
            "History of Present Illness",

        "medications":
            "Medications",

        "plan":
            "Plan"
    }

    return names.get(
        label,
        label.replace("_", " ").title()
    )


def safe_percentage(
    numerator,
    denominator
):
    """
    Return a percentage while avoiding division by zero.
    """

    if denominator == 0:
        return 0.0

    return (
        numerator
        /
        denominator
    ) * 100


# =========================================================
# START
# =========================================================

print()
print("=" * 65)
print("ScribeDr AI - Error Analysis")
print("=" * 65)
print()


# =========================================================
# CHECK MASTER PREDICTION FILE
# =========================================================

if not os.path.exists(
    PREDICTIONS_FILE
):

    print(
        "ERROR: model_predictions.csv "
        "was not found."
    )

    print()
    print(
        "Expected location:"
    )

    print(
        PREDICTIONS_FILE
    )

    print()
    print(
        "Run training\\build_evaluation.py first."
    )

    raise SystemExit(1)


# =========================================================
# LOAD MASTER PREDICTIONS
# =========================================================

print(
    "Loading master prediction dataset..."
)

df = pd.read_csv(
    PREDICTIONS_FILE
)

print(
    f"Test examples loaded: "
    f"{len(df):,}"
)

print()


# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

required_columns = {
    "source_id",
    "text",
    "true_label",
    "logistic_prediction",
    "svm_prediction",
    "clinicalbert_prediction"
}


if not required_columns.issubset(
    df.columns
):

    print(
        "ERROR: model_predictions.csv "
        "is missing required columns."
    )

    print()

    print(
        "Available columns:"
    )

    print(
        df.columns.tolist()
    )

    raise SystemExit(1)


# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    subset=list(
        required_columns
    )
).copy()


df["text"] = (
    df["text"]
    .astype(str)
)


for column in [
    "true_label",
    "logistic_prediction",
    "svm_prediction",
    "clinicalbert_prediction"
]:

    df[column] = (
        df[column]
        .astype(str)
    )


# =========================================================
# VERIFY LABELS
# =========================================================

unexpected_true_labels = (
    set(df["true_label"])
    -
    set(LABELS)
)


if unexpected_true_labels:

    raise RuntimeError(
        "Unexpected true labels found: "
        f"{sorted(unexpected_true_labels)}"
    )


for model_name, column in MODEL_COLUMNS.items():

    unexpected_predictions = (
        set(df[column])
        -
        set(LABELS)
    )

    if unexpected_predictions:

        raise RuntimeError(
            f"Unexpected predictions for "
            f"{model_name}: "
            f"{sorted(unexpected_predictions)}"
        )


# =========================================================
# BASIC DATASET INFORMATION
# =========================================================

dataset_summary = {

    "test_examples":
        int(len(df)),

    "unique_source_records":
        int(
            df["source_id"].nunique()
        ),

    "categories":
        int(len(LABELS)),

    "labels":
        LABELS
}


print("=" * 65)
print("TEST DATASET")
print("=" * 65)
print()

print(
    f"Test examples: "
    f"{dataset_summary['test_examples']:,}"
)

print(
    f"Unique source records: "
    f"{dataset_summary['unique_source_records']:,}"
)

print(
    f"Documentation categories: "
    f"{dataset_summary['categories']}"
)

print()


# =========================================================
# PER-MODEL ERROR ANALYSIS
# =========================================================

model_analysis = {}


print("=" * 65)
print("MODEL ERROR ANALYSIS")
print("=" * 65)
print()


for model_name, prediction_column in MODEL_COLUMNS.items():

    print(
        model_name
    )

    print(
        "-" * 65
    )


    # -----------------------------------------------------
    # Correct / incorrect predictions
    # -----------------------------------------------------

    correct_mask = (
        df["true_label"]
        ==
        df[prediction_column]
    )


    correct_count = int(
        correct_mask.sum()
    )


    error_count = int(
        (~correct_mask).sum()
    )


    accuracy = (
        correct_count
        /
        len(df)
    )


    # -----------------------------------------------------
    # Per-category metrics
    # -----------------------------------------------------

    (
        precision_values,
        recall_values,
        f1_values,
        support_values
    ) = precision_recall_fscore_support(
        df["true_label"],
        df[prediction_column],
        labels=LABELS,
        average=None,
        zero_division=0
    )


    category_performance = {}


    for index, label in enumerate(
        LABELS
    ):

        category_performance[
            label
        ] = {

            "display_name":
                readable_label(label),

            "precision":
                float(
                    precision_values[index]
                ),

            "recall":
                float(
                    recall_values[index]
                ),

            "f1":
                float(
                    f1_values[index]
                ),

            "support":
                int(
                    support_values[index]
                )
        }


    # -----------------------------------------------------
    # Hardest category
    #
    # We define "hardest" as the category with the
    # lowest recall. This means the model missed the
    # greatest proportion of true examples in that class.
    # -----------------------------------------------------

    hardest_label = min(
        LABELS,
        key=lambda label:
            category_performance[
                label
            ]["recall"]
    )


    hardest_category = {

        "label":
            hardest_label,

        "display_name":
            readable_label(
                hardest_label
            ),

        "recall":
            category_performance[
                hardest_label
            ]["recall"],

        "f1":
            category_performance[
                hardest_label
            ]["f1"],

        "support":
            category_performance[
                hardest_label
            ]["support"]
    }


    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    matrix = confusion_matrix(
        df["true_label"],
        df[prediction_column],
        labels=LABELS
    )


    # -----------------------------------------------------
    # Find all off-diagonal errors
    # -----------------------------------------------------

    confusion_errors = []


    for true_index, true_label in enumerate(
        LABELS
    ):

        for predicted_index, predicted_label in enumerate(
            LABELS
        ):

            if true_index == predicted_index:
                continue


            count = int(
                matrix[
                    true_index,
                    predicted_index
                ]
            )


            if count > 0:

                true_support = int(
                    (
                        df["true_label"]
                        ==
                        true_label
                    ).sum()
                )


                confusion_errors.append(
                    {
                        "true_label":
                            true_label,

                        "true_display_name":
                            readable_label(
                                true_label
                            ),

                        "predicted_label":
                            predicted_label,

                        "predicted_display_name":
                            readable_label(
                                predicted_label
                            ),

                        "count":
                            count,

                        "percentage_of_true_category":
                            float(
                                safe_percentage(
                                    count,
                                    true_support
                                )
                            )
                    }
                )


    confusion_errors = sorted(
        confusion_errors,
        key=lambda item:
            item["count"],
        reverse=True
    )


    top_confusions = (
        confusion_errors[:5]
    )


    if confusion_errors:

        largest_confusion = (
            confusion_errors[0]
        )

    else:

        largest_confusion = None


    # -----------------------------------------------------
    # Store model analysis
    # -----------------------------------------------------

    model_analysis[
        model_name
    ] = {

        "correct_predictions":
            correct_count,

        "error_count":
            error_count,

        "accuracy":
            float(accuracy),

        "hardest_category":
            hardest_category,

        "largest_confusion":
            largest_confusion,

        "top_confusions":
            top_confusions,

        "category_performance":
            category_performance
    }


    # -----------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------

    print(
        f"Correct predictions: "
        f"{correct_count}"
    )

    print(
        f"Errors: "
        f"{error_count}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.3f}"
    )

    print()

    print(
        "Hardest category:"
    )

    print(
        f"  "
        f"{hardest_category['display_name']}"
    )

    print(
        f"  Recall: "
        f"{hardest_category['recall']:.3f}"
    )

    print(
        f"  F1-score: "
        f"{hardest_category['f1']:.3f}"
    )

    print()


    if largest_confusion:

        print(
            "Largest confusion:"
        )

        print(
            f"  "
            f"{largest_confusion['true_display_name']}"
            f" -> "
            f"{largest_confusion['predicted_display_name']}"
            f": "
            f"{largest_confusion['count']}"
        )

    else:

        print(
            "Largest confusion: none"
        )


    print()
    print(
        "Top five confusions:"
    )


    if top_confusions:

        for error in top_confusions:

            print(
                f"  "
                f"{error['true_display_name']}"
                f" -> "
                f"{error['predicted_display_name']}"
                f": "
                f"{error['count']}"
            )

    else:

        print(
            "  No classification errors."
        )


    print()


# =========================================================
# MODEL AGREEMENT ANALYSIS
# =========================================================

print("=" * 65)
print("MODEL AGREEMENT ANALYSIS")
print("=" * 65)
print()


logistic_correct = (
    df["logistic_prediction"]
    ==
    df["true_label"]
)


svm_correct = (
    df["svm_prediction"]
    ==
    df["true_label"]
)


clinicalbert_correct = (
    df["clinicalbert_prediction"]
    ==
    df["true_label"]
)


# ---------------------------------------------------------
# All three predictions agree with one another
# ---------------------------------------------------------

all_three_agree = (
    (
        df["logistic_prediction"]
        ==
        df["svm_prediction"]
    )
    &
    (
        df["svm_prediction"]
        ==
        df["clinicalbert_prediction"]
    )
)


all_three_agree_count = int(
    all_three_agree.sum()
)


# ---------------------------------------------------------
# All three models correct
# ---------------------------------------------------------

all_three_correct = (
    logistic_correct
    &
    svm_correct
    &
    clinicalbert_correct
)


all_three_correct_count = int(
    all_three_correct.sum()
)


# ---------------------------------------------------------
# All three models incorrect
# ---------------------------------------------------------

all_three_incorrect = (
    (~logistic_correct)
    &
    (~svm_correct)
    &
    (~clinicalbert_correct)
)


all_three_incorrect_count = int(
    all_three_incorrect.sum()
)


# ---------------------------------------------------------
# Only one model correct
# ---------------------------------------------------------

logistic_only_correct = (
    logistic_correct
    &
    (~svm_correct)
    &
    (~clinicalbert_correct)
)


svm_only_correct = (
    (~logistic_correct)
    &
    svm_correct
    &
    (~clinicalbert_correct)
)


clinicalbert_only_correct = (
    (~logistic_correct)
    &
    (~svm_correct)
    &
    clinicalbert_correct
)


logistic_only_correct_count = int(
    logistic_only_correct.sum()
)


svm_only_correct_count = int(
    svm_only_correct.sum()
)


clinicalbert_only_correct_count = int(
    clinicalbert_only_correct.sum()
)


# ---------------------------------------------------------
# Exactly two models correct
# ---------------------------------------------------------

logistic_svm_only_correct = (
    logistic_correct
    &
    svm_correct
    &
    (~clinicalbert_correct)
)


logistic_bert_only_correct = (
    logistic_correct
    &
    (~svm_correct)
    &
    clinicalbert_correct
)


svm_bert_only_correct = (
    (~logistic_correct)
    &
    svm_correct
    &
    clinicalbert_correct
)


logistic_svm_only_correct_count = int(
    logistic_svm_only_correct.sum()
)


logistic_bert_only_correct_count = int(
    logistic_bert_only_correct.sum()
)


svm_bert_only_correct_count = int(
    svm_bert_only_correct.sum()
)


# ---------------------------------------------------------
# Exactly two correct total
# ---------------------------------------------------------

exactly_two_correct_count = int(
    logistic_svm_only_correct_count
    +
    logistic_bert_only_correct_count
    +
    svm_bert_only_correct_count
)


# ---------------------------------------------------------
# Exactly one correct total
# ---------------------------------------------------------

exactly_one_correct_count = int(
    logistic_only_correct_count
    +
    svm_only_correct_count
    +
    clinicalbert_only_correct_count
)


# ---------------------------------------------------------
# At least one model correct
# ---------------------------------------------------------

at_least_one_correct = (
    logistic_correct
    |
    svm_correct
    |
    clinicalbert_correct
)


at_least_one_correct_count = int(
    at_least_one_correct.sum()
)


# ---------------------------------------------------------
# Agreement object
# ---------------------------------------------------------

agreement_analysis = {

    "total_examples":
        int(len(df)),

    "all_three_predictions_agree":
        all_three_agree_count,

    "all_three_correct":
        all_three_correct_count,

    "all_three_incorrect":
        all_three_incorrect_count,

    "exactly_two_correct":
        exactly_two_correct_count,

    "exactly_one_correct":
        exactly_one_correct_count,

    "at_least_one_correct":
        at_least_one_correct_count,

    "logistic_only_correct":
        logistic_only_correct_count,

    "svm_only_correct":
        svm_only_correct_count,

    "clinicalbert_only_correct":
        clinicalbert_only_correct_count,

    "logistic_and_svm_only_correct":
        logistic_svm_only_correct_count,

    "logistic_and_clinicalbert_only_correct":
        logistic_bert_only_correct_count,

    "svm_and_clinicalbert_only_correct":
        svm_bert_only_correct_count
}


# ---------------------------------------------------------
# Print agreement results
# ---------------------------------------------------------

print(
    "All three predictions agree:",
    all_three_agree_count
)

print(
    "All three correct:",
    all_three_correct_count
)

print(
    "All three incorrect:",
    all_three_incorrect_count
)

print()

print(
    "Exactly two models correct:",
    exactly_two_correct_count
)

print(
    "Exactly one model correct:",
    exactly_one_correct_count
)

print()

print(
    "Logistic Regression only correct:",
    logistic_only_correct_count
)

print(
    "SVM only correct:",
    svm_only_correct_count
)

print(
    "ClinicalBERT only correct:",
    clinicalbert_only_correct_count
)

print()

print(
    "Logistic + SVM correct, "
    "ClinicalBERT wrong:",
    logistic_svm_only_correct_count
)

print(
    "Logistic + ClinicalBERT correct, "
    "SVM wrong:",
    logistic_bert_only_correct_count
)

print(
    "SVM + ClinicalBERT correct, "
    "Logistic wrong:",
    svm_bert_only_correct_count
)

print()

print(
    "At least one model correct:",
    at_least_one_correct_count
)

print()


# =========================================================
# IDENTIFY INTERESTING EXAMPLES
# =========================================================

interesting_examples = {

    "all_three_incorrect":
        [],

    "clinicalbert_only_correct":
        [],

    "logistic_only_correct":
        [],

    "svm_only_correct":
        []
}


# ---------------------------------------------------------
# Helper to extract examples
# ---------------------------------------------------------

def extract_examples(
    dataframe,
    mask,
    limit=10
):

    subset = dataframe[
        mask
    ].head(
        limit
    )


    examples = []


    for _, row in subset.iterrows():

        examples.append(
            {
                "source_id":
                    str(
                        row["source_id"]
                    ),

                "text":
                    row["text"],

                "true_label":
                    row["true_label"],

                "true_display_name":
                    readable_label(
                        row["true_label"]
                    ),

                "logistic_prediction":
                    row[
                        "logistic_prediction"
                    ],

                "logistic_display_name":
                    readable_label(
                        row[
                            "logistic_prediction"
                        ]
                    ),

                "svm_prediction":
                    row[
                        "svm_prediction"
                    ],

                "svm_display_name":
                    readable_label(
                        row[
                            "svm_prediction"
                        ]
                    ),

                "clinicalbert_prediction":
                    row[
                        "clinicalbert_prediction"
                    ],

                "clinicalbert_display_name":
                    readable_label(
                        row[
                            "clinicalbert_prediction"
                        ]
                    )
            }
        )


    return examples


interesting_examples[
    "all_three_incorrect"
] = extract_examples(
    df,
    all_three_incorrect
)


interesting_examples[
    "clinicalbert_only_correct"
] = extract_examples(
    df,
    clinicalbert_only_correct
)


interesting_examples[
    "logistic_only_correct"
] = extract_examples(
    df,
    logistic_only_correct
)


interesting_examples[
    "svm_only_correct"
] = extract_examples(
    df,
    svm_only_correct
)


# =========================================================
# BUILD FINAL JSON STRUCTURE
# =========================================================

analysis = {

    "dataset":
        dataset_summary,

    "models":
        model_analysis,

    "agreement":
        agreement_analysis,

    "interesting_examples":
        interesting_examples
}


# =========================================================
# SAVE ANALYSIS
# =========================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        analysis,
        file,
        indent=4,
        ensure_ascii=False
    )


# =========================================================
# FINAL VALIDATION
# =========================================================

agreement_total = (
    all_three_correct_count
    +
    exactly_two_correct_count
    +
    exactly_one_correct_count
    +
    all_three_incorrect_count
)


print("=" * 65)
print("VALIDATION")
print("=" * 65)
print()

print(
    "Total test examples:",
    len(df)
)

print(
    "Agreement groups total:",
    agreement_total
)


if agreement_total != len(df):

    raise RuntimeError(
        "Agreement groups do not sum "
        "to the complete test set."
    )


print(
    "Agreement groups account for "
    "all test examples: True"
)

print()


# =========================================================
# COMPLETE
# =========================================================

print("=" * 65)
print("ERROR ANALYSIS COMPLETE")
print("=" * 65)
print()

print(
    "Analysis saved to:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "The analysis contains:"
)

print(
    "- hardest category for each model"
)

print(
    "- largest category confusion"
)

print(
    "- five most common confusions"
)

print(
    "- per-category performance"
)

print(
    "- model agreement analysis"
)

print(
    "- examples where individual "
    "models succeed or fail"
)

print()