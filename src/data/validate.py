import pandas as pd
import great_expectations as gx
from src.utils.logger import get_logger

logger = get_logger(__name__)

def validate_dataset(df: pd.DataFrame, dataset_name: str = "transactions") -> bool:
    """
    Validates a pandas DataFrame against quality metrics using Great Expectations.
    Returns True if all critical checks pass, raises an exception or returns False otherwise.
    """
    logger.info(f"Starting Great Expectations validation for: {dataset_name}")
    
    # 1. Get ephemeral data context
    context = gx.get_context()
    
    # 2. Add pandas datasource
    datasource_name = f"{dataset_name}_datasource"
    try:
        datasource = context.data_sources.add_pandas(name=datasource_name)
    except Exception:
        datasource = context.data_sources.get(name=datasource_name)
    
    # 3. Add data asset
    asset_name = f"{dataset_name}_asset"
    try:
        data_asset = datasource.add_dataframe_asset(name=asset_name)
    except Exception:
        data_asset = datasource.get_asset(name=asset_name)
    
    # 4. Get batch definition
    try:
        batch_definition = data_asset.add_batch_definition_whole_dataframe(name="batch_def")
    except Exception:
        batch_definition = data_asset.get_batch_definition(name="batch_def")
    
    # 5. Create expectation suite
    suite_name = f"{dataset_name}_suite"
    try:
        context.suites.add(gx.ExpectationSuite(name=suite_name))
    except Exception:
        pass
    
    # 6. Retrieve validator
    batch_parameters = {"dataframe": df}
    batch_request = batch_definition.build_batch_request(batch_parameters=batch_parameters)
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )
    
    # 7. Configure expectations on the validator
    required_cols = [
        "trans_date_trans_time", "cc_num", "merchant", "category", "amt", 
        "first", "last", "gender", "street", "city", "state", "zip", 
        "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time", 
        "merch_lat", "merch_long"
    ]
    
    for col in required_cols:
        validator.expect_column_to_exist(column=col)
        validator.expect_column_values_to_not_be_null(column=col)
        
    # Transaction amount must be positive (expressed as between 0.0001 and infinity to support all versions of GE)
    validator.expect_column_values_to_be_between(column="amt", min_value=0.0001, max_value=None)
    
    # Coordinates constraints
    validator.expect_column_values_to_be_between(column="lat", min_value=-90.0, max_value=90.0)
    validator.expect_column_values_to_be_between(column="long", min_value=-180.0, max_value=180.0)
    validator.expect_column_values_to_be_between(column="merch_lat", min_value=-90.0, max_value=90.0)
    validator.expect_column_values_to_be_between(column="merch_long", min_value=-180.0, max_value=180.0)
    
    # Gender values should be M or F
    validator.expect_column_values_to_be_in_set(column="gender", value_set=["M", "F"])
    
    if "is_fraud" in df.columns:
        validator.expect_column_values_to_be_in_set(column="is_fraud", value_set=[0, 1])

    # 8. Run validation
    validation_results = validator.validate()
    
    success = validation_results.success
    if success:
        logger.info("Great Expectations validation passed successfully!")
    else:
        logger.warning("Great Expectations validation FAILED! Inspecting errors...")
        for result in validation_results.results:
            if not result.success:
                logger.warning(f"Expectation failed for: {result.expectation_config.kwargs.get('column')} - {result.expectation_config.expectation_type}")
                
    return success
