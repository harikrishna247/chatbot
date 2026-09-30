"""
College Exam FAQ Chatbot (Flask + TF-IDF + Cosine Similarity)
Run:  pip install -r requirements.txt
      python app.py   ->   http://127.0.0.1:5000
"""

import json
import re

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

THRESHOLD = 0.25  # minimum similarity to accept a match

# Short forms students commonly type -> normal words
SYNONYMS = {
    r"\bsem\b": "semester",
    r"\badmit card\b": "hall ticket",
    r"\bhallticket\b": "hall ticket",
    r"\bmarksheet\b": "mark sheet",
    r"\breval\b": "revaluation",
    r"\brecheck(ing)?\b": "revaluation",
    r"\bsupply\b": "arrear",
    r"\bbacklogs?\b": "arrear",
    r"\btime table\b": "timetable",
    r"\bcia\b": "internal marks",
    r"\bcheating\b": "malpractice",
    r"\bphones?\b": "mobile",
}


def clean(text):
    text = text.lower()
    for pattern, word in SYNONYMS.items():
        text = re.sub(pattern, word, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------- Load FAQ data and train ----------
with open("data/faq.json", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

patterns, owners = [], []  # every pattern points to its intent
for i, intent in enumerate(intents):
    for p in intent["patterns"]:
        patterns.append(clean(p))
        owners.append(i)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
matrix = vectorizer.fit_transform(patterns)


# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    message = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not message:
        return jsonify(answer="Please type a question about exams.", confidence=0)

    scores = cosine_similarity(vectorizer.transform([clean(message)]), matrix)[0]
    best = int(scores.argmax())
    confidence = round(float(scores[best]), 2)

    if confidence < THRESHOLD:
        return jsonify(
            answer="Sorry, I don't have an answer for that. Try asking about exam dates, "
                   "hall tickets, results, attendance or revaluation.",
            confidence=confidence,
        )

    intent = intents[owners[best]]
    return jsonify(answer=intent["answer"], intent=intent["intent"], confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
