import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from huggingface_hub import InferenceClient
import json

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the model and job offer embeddings
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name)

# Load precomputed job offer embeddings from the compressed .npz file
job_vectors = np.load('job_vectors_compressed.npz')['job_vectors']

# Load job offer data
data = pd.read_csv('processed_jobs.csv')

# Initialize the Hugging Face Inference API
API_TOKEN = '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' 
client = InferenceClient(token=API_TOKEN)

# Text preprocessing function
def preprocess_text(text):
    text = text.replace("'", " ")  # Replace apostrophes with spaces
    text = text.replace("’", " ")  # Replace apostrophes with spaces
    text = text.replace("`", " ")  # Replace apostrophes with spaces
    text = re.sub(r'\b(nan)\b', ' ', text)  # Remove occurrences of the word "nan"
    text = re.sub(r'\d+', ' ', text)  # Remove digits
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('french')]  # Remove stopwords
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def find_top_20_job_offers(resume):
    resume_embedding = model.encode(resume, convert_to_tensor=True)
    similarities = util.cos_sim(resume_embedding, job_vectors)
    similarities = similarities.numpy().flatten()
    top_20_indices = np.argsort(similarities)[-20:][::-1]
    top_20_job_offers = data.iloc[top_20_indices]
    return top_20_job_offers['Original_Job_Offer'].tolist(), top_20_indices

def truncate_text(text, max_length=380):
    """Truncate the text to a maximum length, ensuring it doesn't cut off in the middle of a word."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    return truncated.rsplit(' ', 1)[0]  # Ensure it doesn't cut off in the middle of a word

def generate_prompt(resume_text, top_20_job_offers):
    truncated_job_offers = [truncate_text(offer) for offer in top_20_job_offers]
    truncated_job_offers_text = "\n\n".join(truncated_job_offers)

    prompt = (
        f"En vous basant sur le CV suivant:\n{resume_text}\n\n"
        f"Voici 20 offres d'emploi pertinentes:\n{truncated_job_offers_text}\n\n"
        "Tu dois recommander les 5 offres d'emploi les plus pertinentes pour le CV fourni en se basant sur les compétences (Suis ce format : Titre du poste - Ville - Résumé du rôle)"
    )
    return prompt

def get_recommendations(prompt):
    response = client.post(
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 700,
                    "temperature": 0.01
                }
            },
            model="mistralai/Mistral-Nemo-Instruct-2407"
        )
    generated_text = json.loads(response.decode())[0]["generated_text"]

    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    cleaned_text = "\n".join(line.strip() for line in generated_text.splitlines() if line.strip())
    return cleaned_text.strip()

# Streamlit UI
st.title('Job Recommendation System')
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf"])

if uploaded_file is not None:
    temp_pdf_path = "temp_uploaded_resume.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    extracted_text = extract_text_from_pdf(temp_pdf_path)
    
    if st.button('Find Relevant Jobs'):
        preprocessed_resume = preprocess_text(extracted_text)
        top_20_job_offers, top_20_indices = find_top_20_job_offers(preprocessed_resume)
        prompt = generate_prompt(preprocessed_resume, top_20_job_offers)
        recommendations = get_recommendations(prompt)
        st.write("Top 5 Job Recommendations:")
        st.text_area("Recommendations", value=recommendations, height=300)
