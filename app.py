import streamlit as st
import pdfplumber  # Assuming resumes are in PDF

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        full_text = ''
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

st.title('Job Recommendation System')
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    # Assuming the file is a PDF for simplicity
    text = extract_text_from_pdf(uploaded_file)
    st.write("Extracted Text:", text)
    # Add your recommendation logic here based on the extracted text
