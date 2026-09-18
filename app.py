from flask import Flask, render_template, request, jsonify

import os
import json
import re

import joblib
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

CLINICALBERT_DIR = os.path.join(
    MODEL_DIR,
    "clinicalbert"
)

# Hugging Face repository containing the fine-tuned model.
# This is used when the local ClinicalBERT directory is not
# available, such as during cloud deployment.
CLINICALBERT_HF_REPO = (
    "thomasking135/ScribeDrAI-ClinicalBERT"
)


# =========================================================
# GLOBAL MODEL OBJECTS
# =========================================================

MODELS = {}

VECTORIZER = None

METRICS = {}

CLASSIFICATION_REPORTS = {}

CONFUSION_MATRICES = {}

ERROR_ANALYSIS = {}

CLINICALBERT_MODEL = None

CLINICALBERT_TOKENIZER = None


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# LOAD MODELS AND EVALUATION DATA
# =========================================================

def load_models():

    global MODELS
    global VECTORIZER
    global METRICS
    global CLASSIFICATION_REPORTS
    global CONFUSION_MATRICES
    global ERROR_ANALYSIS
    global CLINICALBERT_MODEL
    global CLINICALBERT_TOKENIZER


    # -----------------------------------------------------
    # File paths
    # -----------------------------------------------------

    vectorizer_path = os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.joblib"
    )

    logistic_path = os.path.join(
        MODEL_DIR,
        "logistic_regression.joblib"
    )

    svm_path = os.path.join(
        MODEL_DIR,
        "svm.joblib"
    )

    metrics_path = os.path.join(
        MODEL_DIR,
        "metrics.json"
    )

    reports_path = os.path.join(
        MODEL_DIR,
        "classification_reports.json"
    )

    clinicalbert_report_path = os.path.join(
        MODEL_DIR,
        "clinicalbert_report.json"
    )

    confusion_path = os.path.join(
        MODEL_DIR,
        "confusion_matrices.json"
    )

    error_analysis_path = os.path.join(
        MODEL_DIR,
        "error_analysis.json"
    )


    # -----------------------------------------------------
    # TF-IDF vectorizer
    # -----------------------------------------------------

    if os.path.exists(
        vectorizer_path
    ):

        VECTORIZER = joblib.load(
            vectorizer_path
        )

        print(
            "TF-IDF vectorizer loaded successfully."
        )

    else:

        print(
            "WARNING: TF-IDF vectorizer was not found."
        )


    # -----------------------------------------------------
    # Logistic Regression
    # -----------------------------------------------------

    if os.path.exists(
        logistic_path
    ):

        MODELS["logistic"] = joblib.load(
            logistic_path
        )

        print(
            "Logistic Regression loaded successfully."
        )

    else:

        print(
            "WARNING: Logistic Regression model "
            "was not found."
        )


    # -----------------------------------------------------
    # SVM
    # -----------------------------------------------------

    if os.path.exists(
        svm_path
    ):

        MODELS["svm"] = joblib.load(
            svm_path
        )

        print(
            "SVM loaded successfully."
        )

    else:

        print(
            "WARNING: SVM model was not found."
        )


    # -----------------------------------------------------
    # Overall evaluation metrics
    # -----------------------------------------------------

    if os.path.exists(
        metrics_path
    ):

        with open(
            metrics_path,
            "r",
            encoding="utf-8"
        ) as file:

            METRICS = json.load(
                file
            )

        print(
            "Evaluation metrics loaded successfully."
        )

    else:

        print(
            "WARNING: metrics.json was not found."
        )


    # -----------------------------------------------------
    # Classical classification reports
    # -----------------------------------------------------

    if os.path.exists(
        reports_path
    ):

        with open(
            reports_path,
            "r",
            encoding="utf-8"
        ) as file:

            CLASSIFICATION_REPORTS = json.load(
                file
            )

        print(
            "Classical classification reports "
            "loaded successfully."
        )

    else:

        print(
            "WARNING: classification_reports.json "
            "was not found."
        )


    # -----------------------------------------------------
    # ClinicalBERT classification report
    # -----------------------------------------------------

    if os.path.exists(
        clinicalbert_report_path
    ):

        with open(
            clinicalbert_report_path,
            "r",
            encoding="utf-8"
        ) as file:

            clinicalbert_report = json.load(
                file
            )

        CLASSIFICATION_REPORTS[
            "ClinicalBERT"
        ] = clinicalbert_report

        print(
            "ClinicalBERT classification report "
            "loaded successfully."
        )

    else:

        print(
            "WARNING: clinicalbert_report.json "
            "was not found."
        )


    # -----------------------------------------------------
    # Confusion matrices
    # -----------------------------------------------------

    print()

    print(
        "Looking for confusion matrices at:"
    )

    print(
        confusion_path
    )


    if os.path.exists(
        confusion_path
    ):

        with open(
            confusion_path,
            "r",
            encoding="utf-8"
        ) as file:

            CONFUSION_MATRICES = json.load(
                file
            )

        print(
            "Confusion matrices loaded successfully."
        )

        print(
            "Models in confusion matrix file:",
            list(
                CONFUSION_MATRICES.keys()
            )
        )

    else:

        print(
            "WARNING: confusion_matrices.json "
            "was not found."
        )


    # -----------------------------------------------------
    # Error analysis
    # -----------------------------------------------------

    print()

    print(
        "Looking for error analysis at:"
    )

    print(
        error_analysis_path
    )


    if os.path.exists(
        error_analysis_path
    ):

        with open(
            error_analysis_path,
            "r",
            encoding="utf-8"
        ) as file:

            ERROR_ANALYSIS = json.load(
                file
            )

        print(
            "Error analysis loaded successfully."
        )

        print(
            "Error analysis models:",
            list(
                ERROR_ANALYSIS.get(
                    "models",
                    {}
                ).keys()
            )
        )

    else:

        print(
            "WARNING: error_analysis.json "
            "was not found."
        )


    # -----------------------------------------------------
    # Fine-tuned ClinicalBERT
    #
    # LOCAL DEVELOPMENT:
    # Load from models/clinicalbert when available.
    #
    # CLOUD DEPLOYMENT:
    # If the local directory does not exist, load the
    # fine-tuned model from Hugging Face.
    # -----------------------------------------------------

    print()

    if os.path.isdir(
        CLINICALBERT_DIR
    ):

        clinicalbert_source = (
            CLINICALBERT_DIR
        )

        print(
            "Loading fine-tuned ClinicalBERT "
            "from local directory..."
        )

    else:

        clinicalbert_source = (
            CLINICALBERT_HF_REPO
        )

        print(
            "Local ClinicalBERT directory "
            "was not found."
        )

        print(
            "Loading fine-tuned ClinicalBERT "
            "from Hugging Face:"
        )

        print(
            CLINICALBERT_HF_REPO
        )


    try:

        CLINICALBERT_TOKENIZER = (
            AutoTokenizer.from_pretrained(
                clinicalbert_source
            )
        )


        CLINICALBERT_MODEL = (
            AutoModelForSequenceClassification
            .from_pretrained(
                clinicalbert_source
            )
        )


        CLINICALBERT_MODEL.to(
            DEVICE
        )


        CLINICALBERT_MODEL.eval()


        MODELS[
            "clinicalbert"
        ] = "clinicalbert"


        print(
            "ClinicalBERT loaded successfully."
        )

        print(
            "ClinicalBERT source:",
            clinicalbert_source
        )


    except Exception as error:

        CLINICALBERT_MODEL = None
        CLINICALBERT_TOKENIZER = None

        print(
            "WARNING: ClinicalBERT could not "
            "be loaded."
        )

        print(
            "ClinicalBERT error:",
            str(error)
        )

        print(
            "The application will continue "
            "with the available classical models."
        )


    # -----------------------------------------------------
    # Loading summary
    # -----------------------------------------------------

    print()
    print("=" * 65)
    print("SCRIBE DR AI LOAD SUMMARY")
    print("=" * 65)


    print(
        "Available classifier models:",
        list(
            MODELS.keys()
        )
    )


    print(
        "Evaluation metric models:",
        list(
            METRICS.keys()
        )
    )


    print(
        "Classification report models:",
        list(
            CLASSIFICATION_REPORTS.keys()
        )
    )


    print(
        "Confusion matrix models:",
        list(
            CONFUSION_MATRICES.keys()
        )
    )


    print(
        "Error analysis models:",
        list(
            ERROR_ANALYSIS.get(
                "models",
                {}
            ).keys()
        )
    )


    print(
        "Device:",
        DEVICE
    )


    print("=" * 65)
    print()


# =========================================================
# LOAD EVERYTHING AT STARTUP
# =========================================================

load_models()


# =========================================================
# CLINICALBERT PREDICTION
# =========================================================

def predict_clinicalbert(
    segments
):

    encodings = (
        CLINICALBERT_TOKENIZER(
            segments,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
    )


    encodings = {

        key:
            value.to(
                DEVICE
            )

        for key, value
        in encodings.items()
    }


    with torch.no_grad():

        outputs = (
            CLINICALBERT_MODEL(
                **encodings
            )
        )


        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )


    predicted_labels = []


    for prediction in predictions:

        label_id = (
            prediction.item()
        )


        label = (
            CLINICALBERT_MODEL
            .config
            .id2label[
                label_id
            ]
        )


        predicted_labels.append(
            label
        )


    return predicted_labels


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        available_models=list(
            MODELS.keys()
        )
    )


# =========================================================
# CLASSIFICATION API
# =========================================================

@app.route(
    "/classify",
    methods=["POST"]
)
def classify():

    data = request.get_json(
        silent=True
    ) or {}


    text = data.get(
        "text",
        ""
    ).strip()


    model_name = data.get(
        "model",
        "logistic"
    )


    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not text:

        return jsonify({
            "error":
                "Please enter clinical text."
        }), 400


    if model_name not in MODELS:

        return jsonify({
            "error":
                "The selected model is not available."
        }), 400


    # -----------------------------------------------------
    # Split longer input into segments
    # -----------------------------------------------------

    segments = re.split(
        r"\n\s*\n+",
        text
    )


    segments = [

        segment.strip()

        for segment in segments

        if len(
            segment.strip()
        ) >= 10
    ]


    if not segments:

        segments = [
            text
        ]


    # -----------------------------------------------------
    # ClinicalBERT classification
    # -----------------------------------------------------

    if model_name == "clinicalbert":

        if (
            CLINICALBERT_MODEL is None
            or
            CLINICALBERT_TOKENIZER is None
        ):

            return jsonify({
                "error":
                    "ClinicalBERT is not loaded."
            }), 503


        predictions = (
            predict_clinicalbert(
                segments
            )
        )


    # -----------------------------------------------------
    # Classical classification
    # -----------------------------------------------------

    else:

        if VECTORIZER is None:

            return jsonify({
                "error":
                    "TF-IDF vectorizer is not loaded."
            }), 503


        transformed = (
            VECTORIZER.transform(
                segments
            )
        )


        predictions = (
            MODELS[
                model_name
            ].predict(
                transformed
            )
        )


    # -----------------------------------------------------
    # Build individual results
    # -----------------------------------------------------

    results = []


    for segment, prediction in zip(
        segments,
        predictions
    ):

        results.append({

            "text":
                segment,

            "category":
                str(
                    prediction
                )
        })


    # -----------------------------------------------------
    # Group sections by predicted category
    # -----------------------------------------------------

    grouped = {}


    for result in results:

        category = (
            result[
                "category"
            ]
        )


        if category not in grouped:

            grouped[
                category
            ] = []


        grouped[
            category
        ].append(
            result[
                "text"
            ]
        )


    return jsonify({

        "model":
            model_name,

        "segments":
            results,

        "grouped":
            grouped
    })


# =========================================================
# EVALUATION DASHBOARD
# =========================================================

@app.route("/evaluation")
def evaluation():

    print(
        "Evaluation page - confusion models:",
        list(
            CONFUSION_MATRICES.keys()
        )
    )


    print(
        "Evaluation page - error analysis models:",
        list(
            ERROR_ANALYSIS.get(
                "models",
                {}
            ).keys()
        )
    )


    return render_template(
        "evaluation.html",

        metrics=
            METRICS,

        reports=
            CLASSIFICATION_REPORTS,

        confusion_matrices=
            CONFUSION_MATRICES,

        error_analysis=
            ERROR_ANALYSIS
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "ok",

        "models_loaded":
            list(
                MODELS.keys()
            ),

        "evaluation_models":
            list(
                METRICS.keys()
            ),

        "confusion_matrix_models":
            list(
                CONFUSION_MATRICES.keys()
            ),

        "error_analysis_models":
            list(
                ERROR_ANALYSIS.get(
                    "models",
                    {}
                ).keys()
            ),

        "error_analysis_loaded":
            bool(
                ERROR_ANALYSIS
            ),

        "clinicalbert_loaded":
            (
                CLINICALBERT_MODEL is not None
                and
                CLINICALBERT_TOKENIZER is not None
            ),

        "clinicalbert_huggingface_repo":
            CLINICALBERT_HF_REPO,

        "device":
            str(
                DEVICE
            )
    })


# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )