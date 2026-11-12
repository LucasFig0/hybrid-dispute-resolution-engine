"""
Modulo de inferencia em tempo real do modelo tabular calibrado (Tier 1 - LightGBM).

Responsabilidades:
1. Carregar o artefato serializado (artifacts/lgbm_tier1_calibrated.joblib) uma unica vez em memoria (Singleton).
2. Garantir o alinhamento estrito da ordem das 15 features exigidas pelo modelo.
3. Executar a predicao de probabilidade calibrada com latencia alvo abaixo de 35ms.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "lgbm_tier1_calibrated.joblib"


@dataclass(frozen=True)
class Tier1PredictionResult:
    """
    Objeto imutavel de retorno da inferencia do modelo Tier 1.
    """
    calibrated_probability: float
    raw_probability: float
    inference_latency_ms: float


class Tier1TabularPredictor:
    """
    Encapsula o modelo LightGBM calibrado para inferencia online de baixa latencia.
    """

    def __init__(self, artifact_path: Optional[Path] = None) -> None:
        self.artifact_path = artifact_path or DEFAULT_ARTIFACT_PATH
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Artefato do modelo Tier 1 nao encontrado em: {self.artifact_path}. "
                "Execute train_tier1_model.py antes de iniciar o preditor."
            )

        payload: Dict[str, Any] = joblib.load(self.artifact_path)
        self.calibrated_model = payload["model"]
        self.base_model = payload["base_model"]
        self.feature_names: List[str] = payload["features"]
        self.training_metrics: Dict[str, float] = payload["metrics"]

    def predict_single(self, order_context: Dict[str, Any]) -> Tier1PredictionResult:
        """
        Recebe um dicionario com os dados do pedido (vindos da camada Gold do DuckDB),
        valida a presenca de todas as features obrigatorias e retorna a probabilidade calibrada.
        """
        start_ns = time.perf_counter_ns()

        # Verifica se alguma feature obrigatoria esta ausente no payload de entrada
        missing_features = [f for f in self.feature_names if f not in order_context]
        if missing_features:
            raise ValueError(f"Features obrigatorias ausentes para o Tier 1: {missing_features}")

        # Constroi DataFrame de 1 linha respeitando estritamente a ordem de colunas do treinamento
        row_values = [[float(order_context[col] or 0.0) for col in self.feature_names]]
        X_input = pd.DataFrame(row_values, columns=self.feature_names)

        # Calcula probabilidade calibrada (Regressao Isotonica) e probabilidade bruta da arvore
        calib_prob = float(self.calibrated_model.predict_proba(X_input)[0, 1])
        raw_prob = float(self.base_model.predict_proba(X_input)[0, 1])

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        return Tier1PredictionResult(
            calibrated_probability=round(calib_prob, 4),
            raw_probability=round(raw_prob, 4),
            inference_latency_ms=round(elapsed_ms, 3),
        )
