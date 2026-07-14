import numpy as np
import pandas as pd
from src.features.build_features import haversine_distance, FeaturePipeline

def test_haversine_distance():
    # NYC to LA distance approx 3940 km
    lat1, lon1 = np.array([40.7128]), np.array([-74.0060])
    lat2, lon2 = np.array([34.0522]), np.array([-118.2437])
    
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    assert np.abs(dist[0] - 3940) < 100

def test_feature_pipeline_fit_transform():
    # Make small dummy dataset
    dummy_data = pd.DataFrame({
        "trans_date_trans_time": ["2019-01-01 10:00:00", "2019-01-01 11:00:00"],
        "cc_num": [123456, 123456],
        "merchant": ["merch_1", "merch_2"],
        "category": ["shopping_net", "grocery_pos"],
        "amt": [100.0, 50.0],
        "first": ["John", "John"],
        "last": ["Doe", "John"],
        "gender": ["M", "M"],
        "street": ["Main St", "Main St"],
        "city": ["New York", "New York"],
        "state": ["NY", "NY"],
        "zip": [10001, 10001],
        "lat": [40.7, 40.7],
        "long": [-74.0, -74.0],
        "city_pop": [8000000, 8000000],
        "job": ["Developer", "Developer"],
        "dob": ["1980-01-01", "1980-01-01"],
        "trans_num": ["tx1", "tx2"],
        "unix_time": [1546336800, 1546340400],
        "merch_lat": [40.72, 40.68],
        "merch_long": [-73.98, -74.02],
        "is_fraud": [0, 0]
    })
    
    pipeline = FeaturePipeline()
    pipeline.fit(dummy_data)
    transformed = pipeline.transform(dummy_data)
    
    # Check that required features are present
    assert "distance_km" in transformed.columns
    assert "age" in transformed.columns
    assert "hour" in transformed.columns
    assert "time_since_prev_trans_min" in transformed.columns
    assert "customer_pagerank" in transformed.columns
    assert "merchant_pagerank" in transformed.columns
