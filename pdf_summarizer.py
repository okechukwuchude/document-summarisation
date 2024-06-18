#pip install PyMuPDF

import streamlit as st
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

#pip install transformers

from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

"""### Load summarisation pipeline"""

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Create the summarization pipeline
summariser = pipeline("summarization", model=model, tokenizer=tokenizer)

def summarize_text(text):
    summary = summariser(text, do_sample=False)
    return summary[0]['summary_text']

st.title("Document Summariser")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    max_length = st.number_input("Max Length", min_value=1, value=130)
    min_length = st.number_input("Min Length", min_value=1, value=30)
    summary = summarize_text(text, max_length=max_length, min_length=min_length)
    st.write("Summary:")
    st.write(summary)
