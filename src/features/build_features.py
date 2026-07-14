import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from src.utils.logger import get_logger
from src.features.graph_features import build_network_features

logger = get_logger(__name__)

def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Computes geographical distance in kilometers between coordinates using Haversine formula.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km

class FeaturePipeline:
    def __init__(self):
        self.merchant_fraud_rates = {}
        self.category_fraud_rates = {}
        self.state_fraud_rates = {}
        self.job_fraud_rates = {}
        
        self.customer_pr = {}
        self.merchant_pr = {}
        self.global_fraud_rate = 0.0

    def fit(self, train_df: pd.DataFrame):
        logger.info("Fitting Feature Engineering Pipeline...")
        self.global_fraud_rate = train_df['is_fraud'].mean() if 'is_fraud' in train_df.columns else 0.0
        
        self.merchant_fraud_rates = train_df.groupby('merchant')['is_fraud'].mean().to_dict() if 'is_fraud' in train_df.columns else {}
        self.category_fraud_rates = train_df.groupby('category')['is_fraud'].mean().to_dict() if 'is_fraud' in train_df.columns else {}
        self.state_fraud_rates = train_df.groupby('state')['is_fraud'].mean().to_dict() if 'is_fraud' in train_df.columns else {}
        self.job_fraud_rates = train_df.groupby('job')['is_fraud'].mean().to_dict() if 'is_fraud' in train_df.columns else {}
        
        self.customer_pr, self.merchant_pr = build_network_features(train_df)
        logger.info("Feature Engineering Pipeline fitted successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming dataset and engineering features...")
        df = df.copy()
        
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'])

        # 1. Geographic Features
        df['distance_km'] = haversine_distance(df['lat'].values, df['long'].values, 
                                               df['merch_lat'].values, df['merch_long'].values)
        
        # 2. Demographic Features
        df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days / 365.25

        # 3. Temporal Features
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['trans_date_trans_time'].dt.month
        
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

        # 4. Target / Risk Encodings
        df['merchant_fraud_rate'] = df['merchant'].map(self.merchant_fraud_rates).fillna(self.global_fraud_rate)
        df['category_fraud_rate'] = df['category'].map(self.category_fraud_rates).fillna(self.global_fraud_rate)
        df['state_fraud_rate'] = df['state'].map(self.state_fraud_rates).fillna(self.global_fraud_rate)
        df['job_fraud_rate'] = df['job'].map(self.job_fraud_rates).fillna(self.global_fraud_rate)

        # 5. Graph Features Lookup
        df['customer_pagerank'] = df['cc_num'].map(self.customer_pr).fillna(0.0)
        df['merchant_pagerank'] = df['merchant'].map(self.merchant_pr).fillna(0.0)

        # 6. Customer History & Velocity
        logger.info("Computing customer transaction velocities and rolling stats...")
        df = df.sort_values(by=['cc_num', 'trans_date_trans_time']).reset_index(drop=True)
        
        df['prev_trans_time'] = df.groupby('cc_num')['trans_date_trans_time'].shift(1)
        df['time_since_prev_trans_min'] = (df['trans_date_trans_time'] - df['prev_trans_time']).dt.total_seconds() / 60.0
        df['time_since_prev_trans_min'] = df['time_since_prev_trans_min'].fillna(-1.0)
        df = df.drop(columns=['prev_trans_time'])

        # Compute rolling window features correctly mapping multi-index back to df
        rolling_1h = df.groupby('cc_num').rolling('1h', on='trans_date_trans_time')['amt']
        df['cum_count_1h'] = rolling_1h.count().reset_index(level=0, drop=True)
        df['cum_sum_1h'] = rolling_1h.sum().reset_index(level=0, drop=True)
        df['cum_mean_1h'] = df['cum_sum_1h'] / df['cum_count_1h']
        df['cum_std_1h'] = rolling_1h.std().reset_index(level=0, drop=True).fillna(0.0)
        
        rolling_24h = df.groupby('cc_num').rolling('24h', on='trans_date_trans_time')['amt']
        df['cum_count_24h'] = rolling_24h.count().reset_index(level=0, drop=True)
        df['cum_sum_24h'] = rolling_24h.sum().reset_index(level=0, drop=True)
        df['cum_mean_24h'] = df['cum_sum_24h'] / df['cum_count_24h']
        df['cum_std_24h'] = rolling_24h.std().reset_index(level=0, drop=True).fillna(0.0)

        df['amt_to_mean_ratio_1h'] = df['amt'] / (df['cum_mean_1h'] + 1e-5)
        df['amt_to_mean_ratio_24h'] = df['amt'] / (df['cum_mean_24h'] + 1e-5)
        df['amt_diff_mean_24h'] = df['amt'] - df['cum_mean_24h']
        df['amt_z_score_24h'] = (df['amt'] - df['cum_mean_24h']) / (df['cum_std_24h'] + 1e-5)

        df['gender_code'] = df['gender'].map({'M': 1, 'F': 0}).fillna(-1)
        df['category_code'] = df['category'].astype('category').cat.codes

        cols_to_drop = [
            'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'first', 'last', 
            'gender', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num'
        ]
        features_df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Fill any remaining NaNs globally to ensure robust modeling pipelines
        features_df = features_df.fillna(0.0)
        
        logger.info(f"Feature engineering complete. Generated {features_df.shape[1]} features.")
        return features_df
