import os
import json
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "clinical_sections.csv"
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

SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "split_summary.json"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print()
print("=" * 65)
print("ScribeDr AI - Permanent Train/Test Split")
print("=" * 65)
print()

if not os.path.exists(DATA_FILE):

    print("ERROR: Processed dataset not found.")
    print(DATA_FILE)

    raise SystemExit(1)


df = pd.read_csv(DATA_FILE)


required_columns = {
    "source_id",
    "text",
    "label"
}


if not required_columns.issubset(df.columns):

    print(
        "ERROR: clinical_sections.csv does not "
        "contain the required columns."
    )

    raise SystemExit(1)


df = df.dropna(
    subset=[
        "source_id",
        "text",
        "label"
    ]
).copy()


df["text"] = df["text"].astype(str)


df = df[
    df["text"].str.len() >= 20
].copy()


print(
    f"Usable labelled sections: {len(df):,}"
)

print(
    f"Unique source records: "
    f"{df['source_id'].nunique():,}"
)

print()


# ---------------------------------------------------------
# Create group-aware split
# ---------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


train_indices, test_indices = next(
    splitter.split(
        df,
        groups=df["source_id"]
    )
)


train_df = df.iloc[
    train_indices
].copy()


test_df = df.iloc[
    test_indices
].copy()


# ---------------------------------------------------------
# Verify source isolation
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


print("=" * 65)
print("SPLIT VALIDATION")
print("=" * 65)
print()

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
    f"Source overlap: "
    f"{len(overlap)}"
)


if overlap:

    print()
    print(
        "ERROR: Source leakage detected."
    )

    raise SystemExit(1)


# ---------------------------------------------------------
# Verify every class is represented
# ---------------------------------------------------------

print()
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


all_labels = set(
    df["label"].unique()
)

train_labels = set(
    train_df["label"].unique()
)

test_labels = set(
    test_df["label"].unique()
)


if (
    all_labels != train_labels
    or
    all_labels != test_labels
):

    print()
    print(
        "ERROR: Not every category is represented "
        "in both train and test sets."
    )

    raise SystemExit(1)


# ---------------------------------------------------------
# Sort for reproducibility/readability
# ---------------------------------------------------------

train_df = train_df.sort_values(
    by=[
        "source_id",
        "label"
    ]
).reset_index(drop=True)


test_df = test_df.sort_values(
    by=[
        "source_id",
        "label"
    ]
).reset_index(drop=True)


# ---------------------------------------------------------
# Save permanent files
# ---------------------------------------------------------

train_df.to_csv(
    TRAIN_FILE,
    index=False
)


test_df.to_csv(
    TEST_FILE,
    index=False
)


# ---------------------------------------------------------
# Save experiment summary
# ---------------------------------------------------------

summary = {

    "random_state":
        RANDOM_STATE,

    "test_size":
        TEST_SIZE,

    "total_sections":
        int(len(df)),

    "training_sections":
        int(len(train_df)),

    "testing_sections":
        int(len(test_df)),

    "total_source_records":
        int(df["source_id"].nunique()),

    "training_source_records":
        int(train_df["source_id"].nunique()),

    "testing_source_records":
        int(test_df["source_id"].nunique()),

    "source_overlap":
        int(len(overlap)),

    "training_class_distribution":
        {
            str(key): int(value)
            for key, value
            in train_df[
                "label"
            ].value_counts().items()
        },

    "testing_class_distribution":
        {
            str(key): int(value)
            for key, value
            in test_df[
                "label"
            ].value_counts().items()
        }

}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ---------------------------------------------------------
# Complete
# ---------------------------------------------------------

print()
print("=" * 65)
print("PERMANENT SPLIT CREATED")
print("=" * 65)
print()

print("Training set:")
print(TRAIN_FILE)
print()

print("Test set:")
print(TEST_FILE)
print()

print("Split summary:")
print(SUMMARY_FILE)
print()

print(
    "All future models should use these "
    "files instead of generating a new split."
)

print()