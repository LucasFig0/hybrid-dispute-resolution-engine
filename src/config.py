"""
Modulo de configuracao centralizada da aplicacao via variaveis de ambiente (Pydantic Settings).

Responsabilidades:
1. Carregar chaves de API opcionais (Groq ou Gemini) a partir do ambiente ou arquivo .env.
2. Definir parametros de custo de inferencia por milhao de tokens para calculo de Unit Economics.
3. Permitir alternancia transparente entre modo LLM em nuvem (Groq/Gemini) e modo deterministico
   guiado por RAG (offline_rag_engine) para execucao reprodutivel em testes automatizados e CI/CD.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppSettings(BaseSettings):
    """
    Configuracoes globais do Motor Hibrido de Resolucao de Disputas.
    """
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "iFood GenAI Hybrid Dispute Engine"
    app_version: str = "1.0.0"

    # Provedor de inferencia do Tier 2: "offline_rag_engine" (padrao seguro para testes) ou "groq"
    llm_provider: str = "offline_rag_engine"
    groq_api_key: Optional[str] = None
    groq_model_name: str = "llama-3.3-70b-versatile"

    # Parametros de preco de mercado (referencia Llama 3.3 70B / GPT-4o-mini) para telemetria financeira
    llm_input_cost_per_1m_usd: float = 0.59
    llm_output_cost_per_1m_usd: float = 0.79
    llm_max_retries: int = 2
    llm_timeout_seconds: float = 15.0


def get_settings() -> AppSettings:
    """
    Retorna uma instancia validada das configuracoes da aplicacao.
    """
    return AppSettings()
