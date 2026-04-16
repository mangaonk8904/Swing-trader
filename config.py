from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # API Keys
    fintel_api_key: str = ""
    seeking_alpha_rapidapi_key: str = ""

    # Scoring Weights (must sum to 1.0)
    technical_weight: float = 0.35
    fundamental_weight: float = 0.35
    institutional_weight: float = 0.30

    # Signal Thresholds
    strong_buy_threshold: float = 75.0
    buy_threshold: float = 55.0
    neutral_threshold: float = 40.0

    # Technical Indicator Parameters
    rsi_period: int = 14
    sma_short: int = 50
    sma_long: int = 200
    volume_avg_period: int = 20
    atr_period: int = 14

    # Risk Management
    stop_loss_atr_mult: float = 1.5
    target_atr_mult: float = 3.0

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
