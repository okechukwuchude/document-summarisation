import os
import fitz  # PyMuPDF
import streamlit as st
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=None, min_length=None):
    summary_args = {"text": text, "do_sample": False}
    if max_length is not None:
        summary_args["max_length"] = max_length
    if min_length is not None:
        summary_args["min_length"] = min_length
    summary = summarizer(**summary_args)
    return summary[0]['summary_text']

st.title("PDF Text Summarizer")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    max_length = st.number_input("Max Length", min_value=1, value=130)
    min_length = st.number_input("Min Length", min_value=1, value=30)
    summary = summarize_text(text, max_length=max_length, min_length=min_length)
    st.write("Summary:")
    st.write(summary)

# Get the port number from the environment variable
port = int(os.environ.get('PORT', 8501))
st._is_running_with_streamlit = False
st.run(port=port)
