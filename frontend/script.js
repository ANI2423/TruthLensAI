const BASE_URL = "http://127.0.0.1:8000";

async function checkText() {
    const text = document.getElementById("newsText").value;
    const resultEl = document.getElementById("textResult");

    if (!text) {
        resultEl.innerText = "⚠️ Please enter some text";
        return;
    }

    resultEl.innerText = "🔄 Analyzing...";

    const response = await fetch(`${BASE_URL}/predict-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });

    const data = await response.json();
    resultEl.innerText = `Result: ${data.prediction}`;
}

async function checkImage() {
    const file = document.getElementById("imageInput").files[0];
    const resultEl = document.getElementById("imageResult");

    if (!file) {
        resultEl.innerText = "⚠️ Please select an image";
        return;
    }

    resultEl.innerText = "🔄 Analyzing...";

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${BASE_URL}/predict-image`, {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    resultEl.innerText =
        `Result: ${data.prediction} (confidence: ${data.confidence.toFixed(2)})`;
}
