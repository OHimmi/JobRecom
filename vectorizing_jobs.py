import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

def vectorize_jobs(file_path, output_path, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    df = pd.read_csv(file_path)

    # Ensure text columns are of type string and handle missing values
    text_columns = ['job_title', 'function', 'Profil recherché']  # specify your text columns
    for column in text_columns:
        df[column] = df[column].fillna('')  # fill missing values with empty string
        df[column] = df[column].astype(str)  # ensure data type is string

    # Combine columns into a single text column
    df['combined_text'] = df['job_title'] + ' ' + df['function'] + ' ' + df['Profil recherché']

    # Initialize the model
    model = SentenceTransformer(model_name)

    # Vectorize the combined text
    job_vectors = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

    # Save the vectors to a file
    np.save(output_path, job_vectors)

# Run the function
file_path = 'all_job_offers.csv'  # Your CSV file path
output_path = 'job_vectors.npy'  # Path to save the numpy array of job vectors
vectorize_jobs(file_path, output_path)