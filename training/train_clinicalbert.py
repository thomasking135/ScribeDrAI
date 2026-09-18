import os
import json
import random

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report
)

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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

CLINICALBERT_DIR = os.path.join(
    MODEL_DIR,
    "clinicalbert"
)

METRICS_FILE = os.path.join(
    MODEL_DIR,
    "metrics.json"
)

REPORT_FILE = os.path.join(
    MODEL_DIR,
    "clinicalbert_report.json"
)

PREDICTIONS_FILE = os.path.join(
    MODEL_DIR,
    "clinicalbert_predictions.csv"
)

os.makedirs(
    CLINICALBERT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = (
    "emilyalsentzer/Bio_ClinicalBERT"
)

LABELS = [
    "assessment",
    "chief_complaint",
    "history_present_illness",
    "medications",
    "plan"
]

label_to_id = {
    label: index
    for index, label in enumerate(LABELS)
}

id_to_label = {
    index: label
    for label, index in label_to_id.items()
}

MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print()
print("=" * 65)
print("ScribeDr AI - ClinicalBERT Training")
print("=" * 65)
print()

print("Device:", DEVICE)
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

    raise RuntimeError(
        "train_split.csv does not contain "
        "the required columns."
    )


if not required_columns.issubset(
    test_df.columns
):

    raise RuntimeError(
        "test_split.csv does not contain "
        "the required columns."
    )


# ---------------------------------------------------------
# Clean and filter
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
    train_df["text"].astype(str)
)

test_df["text"] = (
    test_df["text"].astype(str)
)


train_df = train_df[
    train_df["label"].isin(LABELS)
].copy()


test_df = test_df[
    test_df["label"].isin(LABELS)
].copy()


# ---------------------------------------------------------
# Verify permanent split
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
    f"Source overlap: "
    f"{len(overlap)}"
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
# Tokenizer
# ---------------------------------------------------------

print(
    "Loading ClinicalBERT tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print(
    "Tokenizer loaded."
)

print()


# ---------------------------------------------------------
# Dataset class
# ---------------------------------------------------------

class ClinicalSectionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length
    ):

        self.texts = (
            dataframe["text"]
            .tolist()
        )

        self.labels = [
            label_to_id[label]
            for label
            in dataframe["label"].tolist()
        ]

        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):

        return len(self.texts)


    def __getitem__(
        self,
        index
    ):

        text = self.texts[index]

        label = self.labels[index]


        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )


        item = {
            key: value.squeeze(0)
            for key, value
            in encoding.items()
        }


        item["labels"] = torch.tensor(
            label,
            dtype=torch.long
        )


        return item


# ---------------------------------------------------------
# Create datasets
# ---------------------------------------------------------

train_dataset = ClinicalSectionDataset(
    train_df,
    tokenizer,
    MAX_LENGTH
)

test_dataset = ClinicalSectionDataset(
    test_df,
    tokenizer,
    MAX_LENGTH
)


# ---------------------------------------------------------
# Data loaders
# ---------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ---------------------------------------------------------
# Load pretrained ClinicalBERT
# ---------------------------------------------------------

print(
    "Loading Bio_ClinicalBERT..."
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id_to_label,
        label2id=label_to_id
    )
)

model.to(
    DEVICE
)

print(
    "ClinicalBERT loaded."
)

print()


# ---------------------------------------------------------
# Optimiser
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

print("=" * 65)
print("TRAINING")
print("=" * 65)
print()


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0


    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )


    for batch_number, batch in enumerate(
        train_loader,
        start=1
    ):

        batch = {
            key: value.to(DEVICE)
            for key, value
            in batch.items()
        }


        optimizer.zero_grad()


        outputs = model(
            **batch
        )


        loss = outputs.loss

        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
        )


        if batch_number % 50 == 0:

            print(
                f"  Batch "
                f"{batch_number}/"
                f"{len(train_loader)} "
                f"- Loss: "
                f"{loss.item():.4f}"
            )


    average_loss = (
        total_loss
        /
        len(train_loader)
    )


    print(
        f"Epoch average loss: "
        f"{average_loss:.4f}"
    )

    print()


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

print("=" * 65)
print("EVALUATION")
print("=" * 65)
print()


model.eval()


true_label_ids = []

predicted_label_ids = []


with torch.no_grad():

    for batch in test_loader:

        labels = batch[
            "labels"
        ]


        inputs = {
            key: value.to(DEVICE)
            for key, value
            in batch.items()
            if key != "labels"
        }


        outputs = model(
            **inputs
        )


        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )


        true_label_ids.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )


        predicted_label_ids.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )


# ---------------------------------------------------------
# Convert IDs back to documentation labels
# ---------------------------------------------------------

true_labels = [
    id_to_label[label_id]
    for label_id
    in true_label_ids
]


predicted_labels = [
    id_to_label[label_id]
    for label_id
    in predicted_label_ids
]


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=LABELS,
        average="macro",
        zero_division=0
    )
)


(
    weighted_precision,
    weighted_recall,
    weighted_f1,
    _
) = precision_recall_fscore_support(
    true_labels,
    predicted_labels,
    labels=LABELS,
    average="weighted",
    zero_division=0
)


accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


# ---------------------------------------------------------
# Detailed report
# ---------------------------------------------------------

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
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
# Save fine-tuned model and tokenizer
# ---------------------------------------------------------

model.save_pretrained(
    CLINICALBERT_DIR
)

tokenizer.save_pretrained(
    CLINICALBERT_DIR
)


# ---------------------------------------------------------
# ClinicalBERT metrics
# ---------------------------------------------------------

clinicalbert_metrics = {

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


# ---------------------------------------------------------
# Preserve classical model metrics
# ---------------------------------------------------------

if os.path.exists(
    METRICS_FILE
):

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metrics = json.load(
            file
        )

else:

    metrics = {}


metrics[
    "ClinicalBERT"
] = clinicalbert_metrics


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
# Save detailed ClinicalBERT report
# ---------------------------------------------------------

report = classification_report(
    true_labels,
    predicted_labels,
    labels=LABELS,
    output_dict=True,
    zero_division=0
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ---------------------------------------------------------
# Save individual ClinicalBERT predictions
# ---------------------------------------------------------

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
            "label":
                "true_label"
        }
    )
)


prediction_results[
    "clinicalbert_prediction"
] = predicted_labels


prediction_results.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print()
print("=" * 65)
print("CLINICALBERT TRAINING COMPLETE")
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


print(
    "Fine-tuned model saved to:"
)

print(
    CLINICALBERT_DIR
)

print()


print(
    "Evaluation results saved to:"
)

print(
    METRICS_FILE
)

print()


print(
    "Detailed report saved to:"
)

print(
    REPORT_FILE
)

print()


print(
    "Individual predictions saved to:"
)

print(
    PREDICTIONS_FILE
)

print()