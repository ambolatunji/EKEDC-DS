import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
import argparse
import sys

def get_file_size(file_path):
    """Get file size in MB"""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)

def optimize_dtypes(df):
    """Optimize data types to reduce memory usage"""
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype in ['int64', 'int32']:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object':
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
    return df

def compress_csv(input_file, output_format='gzip', chunk_size=50000):
    """Compress large CSV file to various formats"""
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return

    original_size = get_file_size(input_file)
    base_name = os.path.splitext(input_file)[0]
    
    print(f"\nProcessing {input_file}")
    print(f"Original size: {original_size:.2f} MB")

    # Get total rows for progress bar
    total_rows = sum(1 for _ in open(input_file)) - 1
    chunks = pd.read_csv(input_file, chunksize=chunk_size)
    
    if output_format == 'gzip':
        output_file = f"{base_name}_compressed.csv.gz"
        first_chunk = True
        with tqdm(total=total_rows, desc="Compressing to GZIP") as pbar:
            for chunk in chunks:
                chunk = optimize_dtypes(chunk)
                chunk.to_csv(output_file, 
                           mode='w' if first_chunk else 'a',
                           header=first_chunk,
                           compression={'method': 'gzip', 'compresslevel': 1},
                           index=False)
                first_chunk = False
                pbar.update(len(chunk))
                del chunk
                gc.collect()

    elif output_format == 'parquet':
        output_file = f"{base_name}_compressed.parquet"
        dfs = []
        total_size = 0
        max_memory = 500  # MB
        
        with tqdm(total=total_rows, desc="Compressing to Parquet") as pbar:
            for chunk in chunks:
                chunk = optimize_dtypes(chunk)
                chunk_size = chunk.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                
                if total_size + chunk_size > max_memory:
                    # Write accumulated chunks to parquet
                    combined_df = pd.concat(dfs, ignore_index=True)
                    if os.path.exists(output_file):
                        combined_df.to_parquet(output_file, engine='pyarrow', append=True, index=False)
                    else:
                        combined_df.to_parquet(output_file, engine='pyarrow', index=False)
                    del combined_df
                    dfs = []
                    total_size = 0
                    gc.collect()
                
                dfs.append(chunk)
                total_size += chunk_size
                pbar.update(len(chunk))
                
            # Write remaining chunks
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                if os.path.exists(output_file):
                    combined_df.to_parquet(output_file, engine='pyarrow', append=True, index=False)
                else:
                    combined_df.to_parquet(output_file, engine='pyarrow', index=False)

    elif output_format == 'pickle':
        output_file = f"{base_name}_compressed.pkl"
        first_chunk = True
        with tqdm(total=total_rows, desc="Compressing to Pickle") as pbar:
            for chunk in chunks:
                chunk = optimize_dtypes(chunk)
                chunk.to_pickle(f"{base_name}_temp_{pbar.n}.pkl", compression='gzip')
                pbar.update(len(chunk))
                del chunk
                gc.collect()
            
        # Merge temp pickle files
        print("\nMerging temporary files...")
        temp_files = [f for f in os.listdir() if f.startswith(f"{os.path.basename(base_name)}_temp_") and f.endswith('.pkl')]
        pd.concat([pd.read_pickle(f) for f in temp_files]).to_pickle(output_file, compression='gzip')
        
        # Cleanup temp files
        for f in temp_files:
            os.remove(f)

    # Calculate compression results
    compressed_size = get_file_size(output_file)
    reduction = ((original_size - compressed_size) / original_size) * 100

    print("\nCompression Results:")
    print(f"Original size: {original_size:.2f} MB")
    print(f"Compressed size: {compressed_size:.2f} MB")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress large CSV files')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--format', choices=['gzip', 'parquet', 'pickle'], 
                        default='gzip', help='Output format (default: gzip)')
    parser.add_argument('--chunk-size', type=int, default=50000,
                        help='Processing chunk size (default: 50000)')
    
    args = parser.parse_args()
    compress_csv(args.input_file, args.format, args.chunk_size)
