import os
import pandas as pd
from typing import Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

def ingest_data(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Reads data from CSV, performs basic schema alignment and downcasts types to save memory.
    """
    logger.info(f"Starting ingestion for file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file {file_path} not found.")

    # Read data
    df = pd.read_csv(file_path)
    
    # Drop index col if exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Clean missing values and duplicates
    initial_rows = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    logger.info(f"Dropped {initial_rows - len(df)} missing/duplicate rows.")

    # Downcast datatypes to save RAM
    logger.info("Downcasting numeric datatypes...")
    df['cc_num'] = df['cc_num'].astype('int64')
    df['zip'] = df['zip'].astype('int32')
    df['city_pop'] = df['city_pop'].astype('int32')
    df['unix_time'] = df['unix_time'].astype('int64')
    df['amt'] = df['amt'].astype('float32')
    df['lat'] = df['lat'].astype('float32')
    df['long'] = df['long'].astype('float32')
    df['merch_lat'] = df['merch_lat'].astype('float32')
    df['merch_long'] = df['merch_long'].astype('float32')
    
    if 'is_fraud' in df.columns:
        df['is_fraud'] = df['is_fraud'].astype('int8')

    # Convert timestamps
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    # Optional Sampling (Stratified to keep fraud proportions)
    if sample_size and sample_size < len(df):
        logger.info(f"Sampling dataset to {sample_size} records...")
        if 'is_fraud' in df.columns:
            # Stratified sample
            fraud_subset = df[df['is_fraud'] == 1]
            legit_subset = df[df['is_fraud'] == 0]
            
            fraud_ratio = len(fraud_subset) / len(df)
            n_fraud = int(sample_size * fraud_ratio)
            n_legit = sample_size - n_fraud
            
            # Bound sampling sizes to available data
            n_fraud = min(n_fraud, len(fraud_subset))
            n_legit = min(n_legit, len(legit_subset))
            
            sampled_fraud = fraud_subset.sample(n=n_fraud, random_state=42)
            sampled_legit = legit_subset.sample(n=n_legit, random_state=42)
            df = pd.concat([sampled_fraud, sampled_legit]).sort_index()
        else:
            df = df.sample(n=sample_size, random_state=42)

    logger.info(f"Ingestion complete. Total records: {len(df)}")
    return df
