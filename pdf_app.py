import fitz  # PyMuPDF for PDF handling
from transformers import pipeline
from docx import Document  # python-docx for Word document handling

# Function to extract text from PDF file
def extract_text_from_pdf(file_like_object):
    document = fitz.open("pdf", file_like_object.read())
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to extract text from Word document
def extract_text_from_docx(file_like_object):
    doc = Document(file_like_object)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Initialize the summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize text
def summarize_text(text, max_length=None, min_length=None):
    if not text.strip():  # Check if text is empty or contains only whitespace
        return ""
    
    summary_args = {"inputs": text, "do_sample": False}
    if max_length is not None:
        summary_args["max_length"] = max_length
    if min_length is not None:
        summary_args["min_length"] = min_length
    
    summary = summarizer(**summary_args)
    return summary[0]['summary_text']

