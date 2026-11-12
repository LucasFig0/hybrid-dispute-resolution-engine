"""
Suite de testes automatizados para o modelo calibrado e preditor online do Tier 1 (LightGBM).

Objetivo:
Validar integridade do artefato serializado, cumprimento das metas de ROC-AUC e Brier Score,
respeito ao SLA de latencia (< 35ms), coerencia direcional de risco e validacao de contrato.
"""

from pathlib import Path
import duckdb
import pytest
from src.router.tabular_predictor import Tier1TabularPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "dispute_warehouse.duckdb"


@pytest.fixture(scope="module")
def predictor() -> Tier1TabularPredictor:
    """
    Inicializa o preditor carregando o artefato real uma unica vez para a suite de testes.
    """
    return Tier1TabularPredictor()


def test_artifact_structure_and_quality_thresholds(predictor: Tier1TabularPredictor) -> None:
    """
    Verifica se o modelo treinado possui exatamente as 15 features da camada Gold
    e se as metricas Out-of-Time superam os limiares minimos de qualidade.
    """
    assert len(predictor.feature_names) == 15, "O modelo deve utilizar exatamente 15 features"
    assert predictor.training_metrics["roc_auc"] >= 0.70, "ROC-AUC OOT abaixo do limiar minimo de 0.70"
    assert predictor.training_metrics["brier_score_calibrated"] <= 0.07, (
        "Brier Score calibrado acima do limite toleravel de 0.07"
    )
    assert predictor.training_metrics["brier_score_calibrated"] <= predictor.training_metrics["brier_score_raw"], (
        "A calibracao isotonica nao deve piorar o Brier Score original no conjunto OOT"
    )


def test_predictor_probability_bounds_and_real_order(predictor: Tier1TabularPredictor) -> None:
    """
    Carrega um registro real da camada Gold do DuckDB e valida se a inferencia
    retorna probabilidades validas no intervalo fechado [0.0, 1.0].
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        row_df = con.execute("SELECT * FROM gold.order_dispute_context LIMIT 1").df()
    finally:
        con.close()

    order_dict = row_df.iloc[0].to_dict()
    result = predictor.predict_single(order_dict)

    assert 0.0 <= result.calibrated_probability <= 1.0
    assert 0.0 <= result.raw_probability <= 1.0


def test_predictor_sla_latency_under_35ms(predictor: Tier1TabularPredictor) -> None:
    """
    Garante que a inferencia unitaria em memoria respeita o SLA da RFC-001 (< 35ms)
    apos o aquecimento inicial (warm-up).
    """
    sample_payload = {feat: 1.0 for feat in predictor.feature_names}

    # Chamada de warm-up
    predictor.predict_single(sample_payload)

    # Medicao real
    result = predictor.predict_single(sample_payload)
    assert result.inference_latency_ms < 35.0, (
        f"Violacao de SLA no Tier 1: latencia de {result.inference_latency_ms}ms excedeu 35ms"
    )


def test_predictor_directional_sanity_monotony(predictor: Tier1TabularPredictor) -> None:
    """
    Teste comportamental (Directional Expectation Test):
    Um pedido com atraso critico (+120 horas) e vendedor com alto historico de defeito (0.65)
    deve obrigatoriamente receber uma probabilidade de falha severa maior do que um pedido
    entregue adiantado (-48 horas) por um vendedor confiavel (0.03).
    """
    healthy_order = {
        "item_count": 1,
        "distinct_products_count": 1,
        "distinct_sellers_count": 1,
        "is_multi_seller_order": 0,
        "total_items_price": 80.0,
        "total_freight_value": 12.0,
        "payment_value": 92.0,
        "max_installments": 1,
        "delivery_delay_hours": -48.0,
        "total_delivery_duration_hours": 72.0,
        "freight_to_payment_ratio": 0.1304,
        "seller_prior_orders_count": 150,
        "seller_historical_defect_rate": 0.0300,
        "customer_prior_orders_count": 2,
        "customer_prior_disputes_count": 0,
    }

    critical_failed_order = {
        **healthy_order,
        "is_multi_seller_order": 1,
        "distinct_sellers_count": 2,
        "delivery_delay_hours": 120.0,
        "total_delivery_duration_hours": 480.0,
        "seller_historical_defect_rate": 0.6500,
    }

    res_healthy = predictor.predict_single(healthy_order)
    res_critical = predictor.predict_single(critical_failed_order)

    assert res_critical.calibrated_probability > res_healthy.calibrated_probability, (
        f"Incoerencia direcional: pedido critico ({res_critical.calibrated_probability}) "
        f"nao superou pedido saudavel ({res_healthy.calibrated_probability})"
    )
    assert res_critical.calibrated_probability >= 0.40, (
        f"Probabilidade calibrada para caso extremo de atraso foi muito baixa: {res_critical.calibrated_probability}"
    )


def test_predictor_raises_error_on_missing_features(predictor: Tier1TabularPredictor) -> None:
    """
    Verifica se o preditor rejeita payloads incompletos lancando ValueError explicito.
    """
    incomplete_payload = {"item_count": 1, "payment_value": 50.0}
    with pytest.raises(ValueError, match="Features obrigatorias ausentes"):
        predictor.predict_single(incomplete_payload)
