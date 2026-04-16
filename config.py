from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


def _get_streamlit_secret(key: str) -> str:
    """Try to read a key from Streamlit secrets (available on Streamlit Cloud)."""
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        return ""


class Settings(BaseSettings):
    # API Keys
    fintel_api_key: str = ""
    seeking_alpha_rapidapi_key: str = ""

    def model_post_init(self, __context) -> None:
        # Fall back to Streamlit secrets if env vars are empty
        if not self.fintel_api_key:
            self.fintel_api_key = _get_streamlit_secret("FINTEL_API_KEY")
        if not self.seeking_alpha_rapidapi_key:
            self.seeking_alpha_rapidapi_key = _get_streamlit_secret("SEEKING_ALPHA_RAPIDAPI_KEY")

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
