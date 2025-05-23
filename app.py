from flask import Flask, request, render_template, send_file, redirect, url_for, flash
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from bs4 import BeautifulSoup
import requests
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from io import BytesIO
from xhtml2pdf import pisa
import re
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel
import PyPDF2
import random
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import logging
from xhtml2pdf.default import DEFAULT_FONT

DEFAULT_FONT['helvetica'] = 'Noto Sans'
DEFAULT_FONT['times'] = 'Noto Sans'
DEFAULT_FONT['courier'] = 'Noto Sans'

# Initialize the Flask app
app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv("key")

data = {}
history = []


# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load models and tokenizers
models = {
    "te_IN": MBartForConditionalGeneration.from_pretrained("aryaumesh/english-to-telugu"),
    "mr_IN": MBartForConditionalGeneration.from_pretrained("aryaumesh/english-to-marathi"),
    "hi_IN": MBartForConditionalGeneration.from_pretrained("aryaumesh/english-to-hindi"),
}

tokenizers = {
    "te_IN": MBart50TokenizerFast.from_pretrained("aryaumesh/english-to-telugu"),
    "mr_IN": MBart50TokenizerFast.from_pretrained("aryaumesh/english-to-marathi"),
    "hi_IN": MBart50TokenizerFast.from_pretrained("aryaumesh/english-to-hindi"),
}

target_languages = {
    "te_IN": "Telugu",
    "mr_IN": "Marathi",
    "hi_IN": "Hindi",
}

# Helper Functions
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "Unknown"

def extract_webpage_text(web_url):
    try:
        response = requests.get(web_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        return f"Error fetching URL: {e}"

def extract_pdf_text(pdf_path):
    text = ''
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text, max_length=512):
    sentences = re.split(r'(?<=\.\?|!\s)', text)
    chunks, chunk = [], []
    for sentence in sentences:
        chunk.append(sentence)
        if len(' '.join(chunk)) > max_length:
            chunks.append(' '.join(chunk[:-1]))
            chunk = [sentence]
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

def translate_text(text, target_language):
    try:
        tokenizer = tokenizers[target_language]
        model = models[target_language]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(inputs["input_ids"], max_length=512, num_beams=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return None

def generate_pdf(rendered_html):
    pdf = BytesIO()
    try:
        pisa.CreatePDF(BytesIO(rendered_html.encode('utf-8')), dest=pdf)
    except Exception as e:
        logging.error(f"Exception in PDF generation: {e}")
        return None
    pdf.seek(0)
    return pdf

# Ensure the 'temp' directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')

@app.route('/')
def language():
    return render_template('index.html', target_languages=target_languages)

@app.route('/language_selection', methods=['POST'])
def language_selection():
    selected_language = request.form.get('selected_language')
    if not selected_language:
        return "No language selected", 400
    data['target_language'] = selected_language
    return redirect('/next_input')

@app.route('/next_input')
def next_input():
    return render_template('translate.html', **data)

@app.route('/translate', methods=['POST'])
def translate():
    try:
        input_type = request.form.get('input_type')
        if input_type == 'url':
            full_text = extract_webpage_text(request.form.get('web_url'))
        elif input_type == 'pdf':
            pdf_file = request.files.get('pdf_file')
            pdf_path = os.path.join('temp', pdf_file.filename)
            pdf_file.save(pdf_path)
            full_text = extract_pdf_text(pdf_path)
            os.remove(pdf_path)
        elif input_type == 'text':
            full_text = request.form.get('text')
        else:
            flash("Invalid input type selected.", "error")
            return redirect(url_for('next_input'))

        target_language = [k for k, v in target_languages.items() if v == data.get('target_language')][0]
        chunks = split_text_into_chunks(full_text)
        translated_chunks = [translate_text(chunk, target_language) for chunk in chunks]
        translated_text = ' '.join([str(chunk) for chunk in translated_chunks if chunk is not None])

        data.update({
            'original_text': full_text,
            'translated_text': translated_text,
            'input_language': target_languages.get(detect_language(full_text), detect_language(full_text)),
            'target_language': data['target_language']
        })

        history.append({'original_text': full_text, 'translated_text': translated_text, 'target_language': target_language})
        if len(history) > 50:
            history.pop(0)
        return render_template('result.html', **data)
    except Exception as e:
        logging.error(f"An error occurred in the translate route: {e}")
        flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for('next_input'))

@app.route('/history')
def view_history():
    return render_template('history.html', history=history)

if __name__ == '__main__':
    app.run(debug=True)
