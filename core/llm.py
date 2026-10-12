"""Chat model factory. Keys come from .env; nothing is hard-coded here."""

from core import config


def get_llm():
    """Return the configured chat model, or raise if its API key is missing."""
    provider = config.LLM_PROVIDER

    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env, or run with --deterministic.")
        from langchain_groq import ChatGroq

        return ChatGroq(model=config.GROQ_MODEL, temperature=config.LLM_TEMPERATURE, api_key=config.GROQ_API_KEY)

    if provider == "google":
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env, or run with --deterministic.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.GOOGLE_MODEL, temperature=config.LLM_TEMPERATURE, google_api_key=config.GOOGLE_API_KEY
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' or 'google'.")
