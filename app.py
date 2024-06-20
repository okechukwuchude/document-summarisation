import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
import torch
import base64

#load model and tokenizer from machine
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto', torch_dtype = torch.float32)

# #load model and tokenizer from huggingface
# tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
# model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

# Check if CUDA (GPU) is available and set the device accordingly
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# base_model.to(device)
#model.to(device)

#file loader and preprocessor
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 200)
    texts = text_splitter.split_documents(pages)
    final_texts = " "
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

# model pipeling from machine
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization", 
        model = base_model, 
        tokenizer = tokenizer,
        max_length = 512,
        min_length = 50
        )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

# model pipeling from huggingface
# def llm_pipeline(filepath):
#     pipe_sum = pipeline(
#         "summarization", 
#         model = model, 
#         tokenizer = tokenizer,
#         max_length = 512,
#         min_length = 50
#         )
#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     result = result[0]['summary_text']
#     return result

@st.cache_data
# function to display pdf
def displayPDF(file):
    #opeining file from path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # embedding pdf in html
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    
    #displaying file
    st.markdown(pdf_display, unsafe_allow_html=True)


# streamlit code
st.set_page_config(layout="wide", page_title="Document Summarizer")

def main():
    
    st.title("Document Summarizer")

    upload_file = st.file_uploader("Choose a file", type=["pdf"])

    if upload_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "data/" + upload_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(upload_file.read())
            with col1:
                st.info("upload file")
                pdf_viewer = displayPDF(filepath)
            with col2:
                st.info("summarisation is below")
                summary = llm_pipeline(filepath)
                st.success("Summary generated successfully")
                st.write(summary)

            

if __name__ == "__main__":
    main()