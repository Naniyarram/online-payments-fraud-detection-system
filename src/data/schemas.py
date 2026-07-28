from pydantic import BaseModel, Field, ConfigDict

class TransactionSchema(BaseModel):
    # Transaction payload validation schema
    model_config = ConfigDict(protected_namespaces=())

    trans_date_trans_time: str = Field(..., description="Timestamp of transaction in YYYY-MM-DD HH:MM:SS format")
    cc_num: int = Field(..., description="Credit card number")
    merchant: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Transaction category")
    amt: float = Field(..., gt=0, description="Transaction amount")
    first: str = Field(..., description="First name of cardholder")
    last: str = Field(..., description="Last name of cardholder")
    gender: str = Field(..., description="Gender of cardholder (M/F)")
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City of cardholder")
    state: str = Field(..., description="State of cardholder")
    zip: int = Field(..., description="Zip code")
    lat: float = Field(..., ge=-90, le=90, description="Latitude of cardholder")
    long: float = Field(..., ge=-180, le=180, description="Longitude of cardholder")
    city_pop: int = Field(..., ge=0, description="Population of cardholder's city")
    job: str = Field(..., description="Job of cardholder")
    dob: str = Field(..., description="Date of birth of cardholder (YYYY-MM-DD)")
    trans_num: str = Field(..., description="Unique transaction ID")
    unix_time: int = Field(..., description="Unix epoch timestamp")
    merch_lat: float = Field(..., ge=-90, le=90, description="Latitude of merchant")
    merch_long: float = Field(..., ge=-180, le=180, description="Longitude of merchant")

