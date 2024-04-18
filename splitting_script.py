import numpy as np
import pandas as pd
import os

def split_csv_equally(file_path, num_parts=4, output_folder='splits'):
    """ Split a large CSV file into exactly num_parts equal parts. """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    df = pd.read_csv(file_path)
    rows_per_part = len(df) // num_parts
    remainder = len(df) % num_parts

    start_idx = 0
    for part in range(1, num_parts + 1):
        if part <= remainder:  # Handle remainder by adding 1 extra row to some parts
            end_idx = start_idx + rows_per_part + 1
        else:
            end_idx = start_idx + rows_per_part

        df_part = df.iloc[start_idx:end_idx]
        chunk_file_path = os.path.join(output_folder, f'part_{part}.csv')
        df_part.to_csv(chunk_file_path, index=False)
        print(f'Part {part} saved: {chunk_file_path}')
        start_idx = end_idx

def split_npy_equally(file_path, num_parts=2, output_folder='splits'):
    """ Split a large NPY file into exactly num_parts equal parts. """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    data = np.load(file_path, allow_pickle=True)
    elements_per_part = len(data) // num_parts
    remainder = len(data) % num_parts

    start_idx = 0
    for part in range(1, num_parts + 1):
        if part <= remainder:  # Handle remainder by adding 1 extra element to some parts
            end_idx = start_idx + elements_per_part + 1
        else:
            end_idx = start_idx + elements_per_part

        part_data = data[start_idx:end_idx]
        chunk_file_path = os.path.join(output_folder, f'part_{part}.npy')
        np.save(chunk_file_path, part_data)
        print(f'Part {part} saved: {chunk_file_path}')
        start_idx = end_idx

# Example usage:
csv_file = 'all_job_offers.csv'  # Your large CSV file path
npy_file = 'job_vectors.npy'    # Your large NPY file path

split_csv_equally(csv_file)
split_npy_equally(npy_file)
