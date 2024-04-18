import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import os

# Function to load and combine split data files
@st.cache_data
def load_data(vectors_folder, dataframe_folder):
    # Load and concatenate all CSV files
    csv_files = [f for f in os.listdir(dataframe_folder) if f.endswith('.csv')]
    frames = [pd.read_csv(os.path.join(dataframe_folder, f)) for f in csv_files]
    jobs_df = pd.concat(frames, ignore_index=True)
    
    # Load and concatenate all NPY files
    npy_files = sorted([f for f in os.listdir(vectors_folder) if f.endswith('.npy')])
    arrays = [np.load(os.path.join(vectors_folder, f)) for f in npy_files]
    job_vectors = np.concatenate(arrays, axis=0)
    
    return job_vectors, jobs_df

# Specify the folders where the split files are stored
vectors_folder = 'splits'
dataframe_folder = 'splits'
job_vectors, jobs_df = load_data(vectors_folder, dataframe_folder)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page_number in range(len(doc)):
        page = doc[page_number]
        full_text += page.get_text()
    return full_text

def find_similar_jobs(resume_text, top_k=5):
    resume_vector = model.encode(resume_text)
    similarities = util.cos_sim(resume_vector, job_vectors)
    top_results = np.argsort(-similarities[0])[:top_k]
    return jobs_df.iloc[top_results]

st.title('Job Recommendation System')
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf"])

if uploaded_file is not None:
    temp_pdf_path = "temp_uploaded_resume.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    extracted_text = extract_text_from_pdf(temp_pdf_path)
    st.write("Extracted Text:")
    st.text_area("Text", value=extracted_text, height=300)
    
    if st.button('Find Relevant Jobs'):
        similar_jobs = find_similar_jobs(extracted_text)
        st.write("Top Job Recommendations:")
        st.dataframe(similar_jobs[['job_title', 'function', 'Profil recherché']])