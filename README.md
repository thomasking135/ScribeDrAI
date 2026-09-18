# ScribeDr AI

ScribeDr AI is a machine-learning research prototype developed to investigate the automatic classification of clinical documentation text.

The project compares three machine-learning approaches for classifying segments of medical transcription text into predefined clinical documentation categories.

## Research Question

How accurately can machine-learning models classify segments of clinical transcription text into predefined clinical documentation categories?

## Project Aim

The aim of ScribeDr AI is to compare classical machine-learning approaches with a transformer-based clinical language model for clinical text classification.

Three models are evaluated:

1. TF-IDF + Logistic Regression
2. TF-IDF + Support Vector Machine (SVM)
3. Fine-tuned Bio_ClinicalBERT

All three models are evaluated on the same permanent held-out test set.

## Classification Categories

Clinical transcription sections are classified into five categories:

- Chief Complaint
- History of Present Illness
- Medications
- Assessment
- Plan

These categories are derived from section headings contained within medical transcription text.

## Dataset

The project uses the public MTSamples medical transcription dataset.

The source dataset contains:

- 4,999 original records
- 4,966 records containing transcription text

During preprocessing, recognised clinical section headings are identified and their associated text is extracted.

After preprocessing and duplicate removal, the final classification dataset contains:

- 1,950 labelled clinical sections
- 975 unique source transcription records
- 5 documentation categories

### Class Distribution

| Category | Sections |
| --- | ---: |
| Assessment | 577 |
| History of Present Illness | 463 |
| Medications | 409 |
| Plan | 354 |
| Chief Complaint | 147 |
| Total | 1,950 |

The resulting dataset is imbalanced, with Chief Complaint representing the smallest category.

## Data Preparation

The preprocessing script is:

```text
training/prepare_data.py
```

It:

- loads the MTSamples CSV file
- identifies recognised clinical section headings
- maps heading variants to the five target categories
- extracts the text belonging to each section
- removes very short sections
- removes exact duplicate text/category examples
- preserves the original source record identifier
- records unrecognised section headings for inspection

Ambiguous combined headings such as `ASSESSMENT AND PLAN` are deliberately excluded rather than being arbitrarily assigned to one category.

The processed dataset is saved as:

```text
data/processed/clinical_sections.csv
```

Unrecognised headings are saved as:

```text
data/processed/unrecognised_headings.csv
```

## Train/Test Split

A permanent source-grouped train/test split is created by:

```text
training/create_test_split.py
```

The split contains:

| Split | Sections | Source Records |
| --- | ---: | ---: |
| Training | 1,581 | 780 |
| Testing | 369 | 195 |
| Total | 1,950 | 975 |

Source overlap between the training and testing sets is:

```text
0
```

This is important because multiple sections can originate from the same medical transcription.

Keeping all sections from an original transcription within a single split reduces the risk of information leakage between training and evaluation data.

The permanent split is stored as:

```text
data/processed/train_split.csv
data/processed/test_split.csv
```

All three models use this same held-out test set.

## Model 1: Logistic Regression

The first baseline model combines TF-IDF text features with Logistic Regression.

The TF-IDF vectoriser uses:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    sublinear_tf=True
)
```

The classifier uses:

```python
LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)
```

Class weighting is used because the five documentation categories are not equally represented.

## Model 2: Support Vector Machine

The second classical model uses the same TF-IDF representation with a linear Support Vector Machine.

```python
LinearSVC(
    class_weight="balanced",
    random_state=42
)
```

Using the same TF-IDF representation allows the two classical classification approaches to be compared under similar conditions.

## Model 3: Bio_ClinicalBERT

The transformer model uses the pretrained:

```text
emilyalsentzer/Bio_ClinicalBERT
```

checkpoint.

Bio_ClinicalBERT is subsequently fine-tuned on the ScribeDr AI clinical section classification dataset.

The training configuration includes:

```text
Maximum sequence length: 256
Batch size: 4
Epochs: 3
Learning rate: 2e-5
Random seed: 42
```

The model is trained using PyTorch and Hugging Face Transformers.

Training automatically uses CUDA when an appropriate GPU is available and otherwise uses the CPU.

## Final Evaluation Results

All three models were evaluated against the same permanent held-out test set containing 369 clinical sections.

Macro averaging is used for precision, recall and F1-score so that each of the five documentation categories contributes equally to these measures.

| Model | Precision | Recall | F1-score | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.861 | 0.823 | 0.832 | 0.873 |
| SVM | 0.863 | 0.801 | 0.810 | 0.875 |
| ClinicalBERT | 0.898 | 0.912 | 0.905 | 0.930 |

These values are experimental results from the permanent held-out test set and should not be interpreted as clinical performance guarantees.

## Classification Errors

The final test results produced:

| Model | Correct | Incorrect | Accuracy |
| --- | ---: | ---: | ---: |
| Logistic Regression | 322 | 47 | 0.873 |
| SVM | 323 | 46 | 0.875 |
| ClinicalBERT | 343 | 26 | 0.930 |

Error analysis also examines the documentation category with the lowest recall and the most common confusion for each model.

## Model Agreement

The three classifiers are also compared on an example-by-example basis.

Of the 369 held-out test examples:

- all three models were correct on 303 examples
- all three models were incorrect on 12 examples
- exactly two models were correct on 25 examples
- at least one model was correct on 357 examples

This analysis helps identify examples where the models behave similarly and examples where their predictions diverge.

## Misclassified Example Explorer

The evaluation dashboard includes an interactive Misclassified Example Explorer.

It allows examples to be inspected from several groups:

- All Three Models Incorrect
- ClinicalBERT Only Correct
- Logistic Regression Only Correct
- SVM Only Correct

For each saved example, the interface displays:

- the true documentation category
- the Logistic Regression prediction
- the SVM prediction
- the ClinicalBERT prediction
- the associated clinical text
- the source record identifier

Previous and Next controls allow multiple examples to be inspected.

The error-analysis process stores a limited number of examples from each group for interactive inspection.

## Evaluation Dashboard

ScribeDr AI includes a browser-based evaluation dashboard.

When the Flask application is running, it can be accessed at:

```text
http://127.0.0.1:5000/evaluation
```

The dashboard displays:

- dataset summary
- overall model comparison
- model performance chart
- class distribution
- per-category precision, recall and F1-score
- confusion matrices
- frequent classification errors
- error analysis summary
- model agreement analysis
- misclassified example explorer
- metric explanations
- experimental design information

## Classifier Interface

The application also contains an interactive classifier.

Clinical text can be entered into the browser interface and classified using the trained models.

Longer input can be divided into segments using blank lines.

Each segment is independently assigned to one of the five documentation categories.

The resulting sections are grouped into a structured display.

This application performs text classification. It does not perform clinical Named Entity Recognition and does not independently extract diagnoses, medication names, symptoms or procedures from free text.

## Project Structure

```text
ScribeDrAI/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── mtsamples.csv
│   │
│   └── processed/
│       ├── clinical_sections.csv
│       ├── unrecognised_headings.csv
│       ├── train_split.csv
│       └── test_split.csv
│
├── models/
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── metrics.json
│   ├── classification_reports.json
│   ├── classical_predictions.csv
│   ├── clinicalbert_predictions.csv
│   ├── model_predictions.csv
│   ├── confusion_matrices.json
│   ├── clinicalbert_report.json
│   ├── error_analysis.json
│   │
│   └── clinicalbert/
│
├── training/
│   ├── prepare_data.py
│   ├── create_test_split.py
│   ├── train_models.py
│   ├── train_clinicalbert.py
│   ├── build_evaluation.py
│   └── analyse_errors.py
│
├── templates/
│   ├── index.html
│   └── evaluation.html
│
└── static/
    ├── style.css
    └── script.js
```

## Installation

Python 3.11 is used for the project.

From the project directory:

```cmd
cd C:\ScribeDrAI
```

Install the required Python packages:

```cmd
py -m pip install -r requirements.txt
```

The ClinicalBERT component additionally requires PyTorch, Transformers, Accelerate and Safetensors.

If they are not already included in `requirements.txt`, they can be installed with:

```cmd
py -m pip install torch
py -m pip install transformers
py -m pip install accelerate
py -m pip install safetensors
```

## Running the Application

From:

```text
C:\ScribeDrAI
```

run:

```cmd
py app.py
```

The Flask development server can then be accessed locally at:

```text
http://127.0.0.1:5000
```

The evaluation dashboard is available at:

```text
http://127.0.0.1:5000/evaluation
```

## Experimental Pipeline

The project contains the following experimental pipeline:

```text
MTSamples
    ↓
Clinical section extraction
    ↓
Permanent source-grouped train/test split
    ↓
┌──────────────────────────────┐
│                              │
│ TF-IDF                       │
│   ├── Logistic Regression    │
│   └── Linear SVM             │
│                              │
│ Bio_ClinicalBERT             │
│   └── Fine-tuning            │
│                              │
└──────────────────────────────┘
    ↓
Common held-out test set
    ↓
Precision / Recall / F1 / Accuracy
    ↓
Confusion Matrices
    ↓
Error and Agreement Analysis
    ↓
Flask Evaluation Dashboard
```

## Reproducing the Experiment

The experimental pipeline is represented by the following scripts:

```cmd
py training\prepare_data.py
py training\create_test_split.py
py training\train_models.py
py training\train_clinicalbert.py
py training\build_evaluation.py
py training\analyse_errors.py
```

The application can then be started with:

```cmd
py app.py
```

Important: retraining the models may overwrite existing model files and experimental results.

For that reason, existing trained models and evaluation artefacts should be preserved before conducting a new experimental run.

## Evaluation Metrics

### Precision

Precision measures the proportion of sections assigned to a category that actually belong to that category.

### Recall

Recall measures the proportion of sections belonging to a category that the model successfully identifies.

### F1-score

F1-score combines precision and recall into a single measure.

### Support

Support represents the number of test examples belonging to a particular documentation category.

### Macro Averaging

Macro averaging calculates a metric independently for each category and then averages the category scores.

This is useful for this experiment because the dataset is imbalanced and prevents the largest category from completely dominating the reported precision, recall and F1-score.

## Responsible AI

ScribeDr AI is an academic machine-learning prototype.

Responsible use requires consideration of several issues.

### Privacy and Data Governance

The project should use public, de-identified or fictional clinical text.

Real identifiable patient information should not be entered into the research prototype.

Any future use involving health information would require appropriate privacy, security, consent and governance processes.

For a New Zealand context, relevant considerations include the Privacy Act 2020 and Health Information Privacy Code 2020.

### Bias and Fairness

The performance of the models is dependent on the characteristics of the training dataset.

Performance measured on MTSamples-derived text does not establish equivalent performance across different hospitals, countries, medical specialties, patient populations or documentation styles.

Per-category metrics are therefore reported rather than relying only on overall accuracy.

### Transparency

The system clearly identifies the classification task and reports the models used.

The evaluation dashboard provides:

- precision
- recall
- F1-score
- accuracy
- confusion matrices
- error analysis
- model agreement
- examples of classification behaviour

This makes model performance and limitations more visible than reporting a single accuracy value alone.

### Human Oversight

Predictions produced by ScribeDr AI should be treated as machine-generated classifications requiring human review.

The application is not designed to replace clinicians or professional clinical documentation processes.

## Limitations

Important limitations include:

- the dataset is relatively small
- the five classes are imbalanced
- Chief Complaint has substantially fewer examples than the other categories
- labels are derived from existing section headings rather than newly created expert annotations
- documentation conventions in MTSamples may not represent current clinical documentation practices everywhere
- the system performs section classification rather than medical reasoning
- blank-line separation is used by the application when classifying multiple user-entered segments
- performance on the held-out MTSamples-derived test set does not demonstrate performance in a real clinical environment
- the application has not undergone clinical validation

## Intended Use

ScribeDr AI is intended for:

- machine-learning research
- university assessment
- experimentation with clinical NLP
- comparison of classification architectures
- demonstration of model evaluation techniques

It is not intended for:

- diagnosis
- treatment recommendations
- emergency medical use
- autonomous clinical documentation
- clinical decision-making

## Status

Current prototype functionality includes:

- dataset preprocessing
- source-grouped train/test splitting
- TF-IDF feature extraction
- Logistic Regression classification
- SVM classification
- Bio_ClinicalBERT fine-tuning
- common held-out model evaluation
- confusion matrices
- error analysis
- model agreement analysis
- interactive example inspection
- Flask classifier interface
- Flask evaluation dashboard

## Author

Thomas King

University machine-learning research project

2026