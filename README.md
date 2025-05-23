The Multilingual AI Translation System is a cutting-edge web-based platform designed to
facilitate seamless communication across Indian languages, focusing on Telugu, Tamil, and
Marathi. Built using Flask and powered by mBART, an advanced Neural Machine
Translation (NMT) model, the system ensures fluent, contextually relevant, and semantically
accurate translations.
The system leverages natural language processing (NLP) to provide the following key
functionalities:
Multiformat Input Support: Translates content from plain text, URLs, and PDFs,
ensuring adaptability to diverse content sources.
Automatic Language Detection: Identifies the source language without requiring
manual input, streamlining the translation process.
Context-Aware Translation: Uses mBART model to preserve meaning and improve
translation fluency.
Evaluation Metrics for Accuracy: Implements BLEU, ROUGE, METEOR, Cosine
Similarity, Precision, Recall, and F1 Score to assess translation quality.
Translation History Tracking: Saves previous translations for quick reference and
usability.


# Step 1: Create and activate a virtual environment
python -m venv myenv
myenv\Scripts\activate
# Step 2: Set Flask environment variable
set FLASK_APP=app.py
# Step 3: Run the Flask application
flask run
