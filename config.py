from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


def _get_streamlit_secret(key: str) -> str:
    """Try to read a key from Streamlit secrets (available on Streamlit Cloud)."""
    try:
        import streamlit as st
        # Access via dict-style to support both nested and flat secrets
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
        return ""
    except Exception:
        return ""


class Settings(BaseSettings):
    # API Keys
    fintel_api_key: str = ""
    seeking_alpha_rapidapi_key: str = ""
    groq_api_key: str = ""

    def model_post_init(self, __context) -> None:
        # Fall back to Streamlit secrets if env vars are empty
        if not self.fintel_api_key:
            self.fintel_api_key = _get_streamlit_secret("FINTEL_API_KEY")
        if not self.seeking_alpha_rapidapi_key:
            self.seeking_alpha_rapidapi_key = _get_streamlit_secret("SEEKING_ALPHA_RAPIDAPI_KEY")
        if not self.groq_api_key:
            self.groq_api_key = _get_streamlit_secret("GROQ_API_KEY")

    # Scoring Weights (must sum to 1.0)
    technical_weight: float = 0.30
    fundamental_weight: float = 0.25
    institutional_weight: float = 0.20
    sa_weight: float = 0.25

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
