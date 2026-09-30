# College Exam FAQ Chatbot

A chatbot that answers frequently asked questions about college examinations, such as exam dates, hall tickets, attendance, results and revaluation.

## How it works
1. The student's question is cleaned and short forms (sem, reval, admit card) are expanded.
2. TF-IDF converts the question into a vector.
3. Cosine similarity finds the closest FAQ.
4. The bot returns that answer. If the match is weak, it says it doesn't know.

## Tech Stack
Python, Flask, scikit-learn, HTML/CSS/JavaScript

## Run
pip install -r requirements.txt
python app.py

Open http://127.0.0.1:5000

## Project Structure
- app.py: Flask backend and NLP logic
- data/faq.json: FAQ questions and answers
- templates/index.html: chat interface

## Add a new FAQ
Add a new entry with `patterns` and an `answer` in `data/faq.json`, then restart the app.