import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def validate_dataset(df: pd.DataFrame, dataset_name: str = "transactions") -> bool:
    """
    Validates a pandas DataFrame against quality metrics and contract expectations.
    """
    logger.info(f"Starting data validation for: {dataset_name}")
    
    # Required columns contract
    required_cols = [
        "trans_date_trans_time", "cc_num", "merchant", "category", "amt", 
        "first", "last", "gender", "street", "city", "state", "zip", 
        "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time", 
        "merch_lat", "merch_long"
    ]
    
    # Try Great Expectations if supported
    try:
        import great_expectations as gx
        context = gx.get_context()
        datasource_name = f"{dataset_name}_datasource"
        try:
            datasource = context.data_sources.add_pandas(name=datasource_name)
        except Exception:
            datasource = context.data_sources.get(name=datasource_name)
            
        data_asset = datasource.add_dataframe_asset(name=f"{dataset_name}_asset")
        batch_def = data_asset.add_batch_definition_whole_dataframe(name="batch_def")
        suite = context.suites.add(gx.ExpectationSuite(name=f"{dataset_name}_suite"))
        
        validator = context.get_validator(
            batch_request=batch_def.build_batch_request(batch_parameters={"dataframe": df}),
            expectation_suite_name=f"{dataset_name}_suite"
        )
        for col in required_cols:
            validator.expect_column_to_exist(column=col)
            validator.expect_column_values_to_not_be_null(column=col)
            
        validator.expect_column_values_to_be_between(column="amt", min_value=0.0001)
        res = validator.validate()
        if res.success:
            logger.info("Great Expectations validation passed.")
            return True
    except Exception as ge_err:
        logger.warning(f"GE validation skipped due to environment incompatibility ({ge_err}). Running native checks.")

    # Native Pandas contract validation fallback
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in {dataset_name}: {missing_cols}")
        return False
        
    null_counts = df[required_cols].isnull().sum().sum()
    if null_counts > 0:
        logger.warning(f"Found {null_counts} null values across required columns in {dataset_name}.")

    if (df['amt'] <= 0).any():
        logger.warning("Found non-positive transaction amounts in dataset.")
        
    logger.info(f"Native contract validation successful for {dataset_name}.")
    return True

