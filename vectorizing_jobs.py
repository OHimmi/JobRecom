import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
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

# Function to preprocess and encode job offers
def preprocess_and_encode_jobs(csv_file, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    df = pd.read_csv(csv_file)
    
    # Keep a copy of the original job offers
    df['Original_Job_Offer'] = df['Job_Offer']

    # Apply preprocessing to the 'Job_Offer' column
    df['Job_Offer'] = df['Job_Offer'].apply(preprocess_text)

    # Load the sentence-transformers model
    model = SentenceTransformer(model_name)

    # Encode the job offers text
    job_vectors = model.encode(df['Job_Offer'].tolist(), show_progress_bar=True)

    # Save the embeddings as a compressed .npz file
    np.savez_compressed('job_vectors_compressed.npz', job_vectors=job_vectors)

    # Save the DataFrame as a CSV file (optional, for reference)
    df.to_csv('processed_jobs.csv', index=False)

    print("Job embeddings and data saved successfully.")

csv_file = 'jobs.csv'  # Replace with your actual CSV file path
preprocess_and_encode_jobs(csv_file)
