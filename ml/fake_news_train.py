import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Simple training data (for learning)
data = {
    "text": [
        "Government released official notice",
        "This is fake news and rumor",
        "Clickbait headline spreading fake story",
        "Election results officially declared",
        "Fake claim going viral on social media"
    ],
    "label": [0, 1, 1, 0, 1]  # 0 = REAL, 1 = FAKE
}

df = pd.DataFrame(data)

# 2. Convert text → numbers
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 3. Train model
model = LogisticRegression()
model.fit(X, y)

# 4. Save model + vectorizer
joblib.dump(model, "../backend/models/fake_news_model.pkl")
joblib.dump(vectorizer, "../backend/models/vectorizer.pkl")

print("✅ Fake News model trained and saved")
