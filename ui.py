import streamlit as st
import os
from app import extract_text_from_pdf, extract_text_from_docx, summarize_text

st.title("Document Summarizer")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    # Extract text based on file type
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension == ".pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == ".docx":
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == ".txt":
        text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
        st.stop()

    # Display extracted text
    st.success(f"Text extracted successfully from {file_extension.upper()} file.")
    
    if st.button("Generate Summary"):
        with st.spinner('Generating summary...'):
            # Generate summary
            summary = summarize_text(text)
        
        # Display the summary
        st.write("Summary:")
        st.write(summary)
