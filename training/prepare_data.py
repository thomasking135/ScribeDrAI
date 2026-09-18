import os
import re
from collections import Counter

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "mtsamples.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "clinical_sections.csv"
)

UNRECOGNISED_FILE = os.path.join(
    OUTPUT_DIR,
    "unrecognised_headings.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Clinical section mapping
# ---------------------------------------------------------

SECTION_MAP = {

    # Chief Complaint
    "CHIEF COMPLAINT": "chief_complaint",
    "CHIEF COMPLAINTS": "chief_complaint",
    "CHIEF COMPLIANT": "chief_complaint",
    "CHIEF REASON FOR CONSULTATION": "chief_complaint",
    "CHIEF COMPLAINT - REASON FOR VISIT":
        "chief_complaint",
    "CHIEF COMPLAINT / REASON FOR THE VISIT":
        "chief_complaint",

    # History of Present Illness
    "HISTORY OF PRESENT ILLNESS":
        "history_present_illness",

    "HISTORY OF PRESENTING ILLNESS":
        "history_present_illness",

    "BRIEF HISTORY OF PRESENT ILLNESS":
        "history_present_illness",

    "CURRENT HISTORY OF PRESENT ILLNESS":
        "history_present_illness",

    "HISTORY OF PRESENT COMPLAINT":
        "history_present_illness",

    "HISTORY OF PRESENT PROBLEM":
        "history_present_illness",

    "HISTORY OF PRESENTING PROBLEM":
        "history_present_illness",

    "HISTORY OF PRESENTING COMPLAINT":
        "history_present_illness",

    "HPI":
        "history_present_illness",

    # Medications
    "MEDICATIONS": "medications",
    "MEDICATION": "medications",
    "CURRENT MEDICATIONS": "medications",
    "CURRENT MEDICATION": "medications",
    "HOME MEDICATIONS": "medications",
    "PRESENT MEDICATIONS": "medications",
    "MEDICATIONS AT HOME": "medications",
    "CURRENT MEDICATIONS AT HOME": "medications",
    "MEDICATIONS PRIOR TO ADMISSION": "medications",
    "MEDICATIONS ON ADMISSION": "medications",

    # Assessment
    "ASSESSMENT": "assessment",
    "ASSESSMENTS": "assessment",
    "IMPRESSION": "assessment",
    "DIAGNOSTIC IMPRESSION": "assessment",
    "FINAL IMPRESSION": "assessment",
    "CLINICAL IMPRESSION": "assessment",
    "INITIAL IMPRESSION": "assessment",

    # Plan
    "PLAN": "plan",
    "PLANS": "plan",
    "TREATMENT PLAN": "plan",
    "PLAN OF CARE": "plan",
    "DISCHARGE PLAN": "plan",
    "POSTOPERATIVE PLAN": "plan"
}


# ---------------------------------------------------------
# Heading detection
# ---------------------------------------------------------

HEADING_PATTERN = re.compile(
    r"(?:^|,)\s*"
    r"([A-Z][A-Z0-9 /&().\-]{1,60})"
    r":\s*,?",
    re.MULTILINE
)


def normalise_heading(heading):
    """
    Standardise spacing and remove simple numbering
    such as CHIEF COMPLAINT (1/1).
    """

    heading = " ".join(heading.split()).strip()

    heading = re.sub(
        r"\s*\(\d+/\d+\)\s*$",
        "",
        heading
    )

    return heading


def clean_section_text(text):
    """
    Clean formatting artefacts while retaining the
    clinical wording of the original transcription.
    """

    text = text.strip()

    # Remove leading/trailing commas and spaces
    text = text.strip(" ,")

    # Replace repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_sections(transcription, source_id):
    """
    Extract recognised clinical sections from a
    single MTSamples transcription.
    """

    sections = []
    unrecognised = []

    matches = list(
        HEADING_PATTERN.finditer(transcription)
    )

    if not matches:
        return sections, unrecognised

    for index, match in enumerate(matches):

        raw_heading = match.group(1)

        heading = normalise_heading(
            raw_heading
        )

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(transcription)

        section_text = transcription[start:end]

        section_text = clean_section_text(
            section_text
        )

        if heading in SECTION_MAP:

            label = SECTION_MAP[heading]

            # Ignore extremely short sections
            if len(section_text) >= 20:

                sections.append({
                    "source_id": source_id,
                    "original_heading": heading,
                    "text": section_text,
                    "label": label
                })

        else:

            unrecognised.append({
                "heading": heading
            })

    return sections, unrecognised


# ---------------------------------------------------------
# Load MTSamples
# ---------------------------------------------------------

print()
print("=" * 60)
print("ScribeDr AI - MTSamples Data Preparation")
print("=" * 60)
print()


if not os.path.exists(INPUT_FILE):

    print("ERROR: MTSamples dataset not found.")
    print()
    print("Expected location:")
    print(INPUT_FILE)

    raise SystemExit(1)


print("Loading MTSamples...")

df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df):,}")
print()


# ---------------------------------------------------------
# Validate dataset
# ---------------------------------------------------------

if "transcription" not in df.columns:

    print(
        "ERROR: The CSV does not contain "
        "a 'transcription' column."
    )

    raise SystemExit(1)


# ---------------------------------------------------------
# Extract labelled sections
# ---------------------------------------------------------

all_sections = []

unrecognised_counter = Counter()

records_with_transcription = 0

records_with_recognised_sections = 0


for row_index, row in df.iterrows():

    transcription = row["transcription"]

    if pd.isna(transcription):
        continue

    transcription = str(transcription)

    if not transcription.strip():
        continue

    records_with_transcription += 1

    sections, unrecognised = extract_sections(
        transcription,
        row_index
    )

    if sections:
        records_with_recognised_sections += 1

    all_sections.extend(sections)

    for item in unrecognised:
        unrecognised_counter[
            item["heading"]
        ] += 1


# ---------------------------------------------------------
# Create processed dataset
# ---------------------------------------------------------

processed_df = pd.DataFrame(
    all_sections
)


if processed_df.empty:

    print("No recognised clinical sections found.")

    raise SystemExit(1)


# Remove exact duplicate examples
processed_df = processed_df.drop_duplicates(
    subset=["text", "label"]
)

processed_df = processed_df.reset_index(
    drop=True
)


# ---------------------------------------------------------
# Save labelled dataset
# ---------------------------------------------------------

processed_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Save unrecognised headings for auditing
# ---------------------------------------------------------

unrecognised_df = pd.DataFrame(
    unrecognised_counter.most_common(),
    columns=[
        "heading",
        "count"
    ]
)

unrecognised_df.to_csv(
    UNRECOGNISED_FILE,
    index=False
)


# ---------------------------------------------------------
# Report results
# ---------------------------------------------------------

print("Data preparation complete.")
print()

print(
    f"Records containing transcription text: "
    f"{records_with_transcription:,}"
)

print(
    f"Records containing recognised sections: "
    f"{records_with_recognised_sections:,}"
)

print()

print("LABEL DISTRIBUTION")
print("-" * 60)


label_counts = (
    processed_df["label"]
    .value_counts()
)


for label, count in label_counts.items():

    print(
        f"{label:<30} {count:>6,}"
    )


print("-" * 60)

print(
    f"{'TOTAL':<30} "
    f"{len(processed_df):>6,}"
)

print()


print("Processed dataset saved to:")
print(OUTPUT_FILE)

print()

print("Unrecognised heading audit saved to:")
print(UNRECOGNISED_FILE)

print()

print(
    "Next step: train the Logistic Regression "
    "and SVM classifiers."
)

print()