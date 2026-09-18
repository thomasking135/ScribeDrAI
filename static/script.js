const button =
    document.getElementById("classifyButton");

const clinicalText =
    document.getElementById("clinicalText");

const modelSelect =
    document.getElementById("modelSelect");

const resultCard =
    document.getElementById("resultCard");

const modelUsed =
    document.getElementById("modelUsed");

const structuredResults =
    document.getElementById("structuredResults");


button.addEventListener("click", async () => {

    const text = clinicalText.value.trim();
    const model = modelSelect.value;

    if (!text) {
        alert("Please enter some clinical text.");
        return;
    }

    if (!model) {
        alert("Please select a trained model.");
        return;
    }

    button.disabled = true;
    button.textContent = "Classifying...";

    try {

        const response = await fetch(
            "/classify",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    text: text,
                    model: model
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error ||
                "Classification failed."
            );
        }

        structuredResults.innerHTML = "";

        const categoryOrder = [
            "chief_complaint",
            "history_present_illness",
            "medications",
            "assessment",
            "plan"
        ];

        categoryOrder.forEach(category => {

            if (!data.grouped[category]) {
                return;
            }

            const section =
                document.createElement("div");

            section.className =
                "structured-section";

            const heading =
                document.createElement("h3");

            heading.textContent =
                formatCategory(category);

            section.appendChild(heading);

            data.grouped[category].forEach(
                text => {

                    const paragraph =
                        document.createElement("p");

                    paragraph.textContent = text;

                    section.appendChild(
                        paragraph
                    );
                }
            );

            structuredResults.appendChild(
                section
            );
        });

        modelUsed.textContent =
            "Model: " +
            formatModel(data.model);

        resultCard.classList.remove(
            "hidden"
        );

    } catch (error) {

        alert(error.message);

    } finally {

        button.disabled = false;

        button.textContent =
            "Classify Clinical Text";
    }
});


function formatCategory(category) {

    const names = {

        chief_complaint:
            "Chief Complaint",

        history_present_illness:
            "History of Present Illness",

        medications:
            "Medications",

        assessment:
            "Assessment",

        plan:
            "Plan"
    };

    return names[category] || category;
}


function formatModel(model) {

    const names = {

        logistic:
            "TF-IDF + Logistic Regression",

        svm:
            "TF-IDF + Support Vector Machine",

        clinicalbert:
            "ClinicalBERT"
    };

    return names[model] || model;
}