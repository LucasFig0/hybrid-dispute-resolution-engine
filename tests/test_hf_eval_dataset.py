"""
Suite de testes automatizados para validar o Dataset de Benchmark de Disputas (Hugging Face).

Objetivo:
1. Garantir volume e balanceamento exato dos 3 splits (100 casos cada = 300 casos).
2. Provar ausencia de vazamento de dados (Zero Data Leakage) em relacao ao split de treino (70% iniciais).
3. Validar que o HybridCascadeRouter roteia os 300 casos com precisao nos tiers esperados.
"""

import json
from pathlib import Path
import duckdb
import pytest
from src.router.cascade_router import HybridCascadeRouter
from src.schemas.dispute import DisputeRequest, OrderGoldContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "dispute_warehouse.duckdb"
BENCHMARK_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "dispute_eval_benchmark.jsonl"


@pytest.fixture(scope="module")
def benchmark_data() -> list[dict]:
    """
    Carrega os registros do arquivo JSONL gerado para o benchmark.
    """
    if not BENCHMARK_JSONL_PATH.exists():
        pytest.fail(f"Arquivo de benchmark nao encontrado em {BENCHMARK_JSONL_PATH}")

    with open(BENCHMARK_JSONL_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_benchmark_splits_balance_and_schema(benchmark_data: list[dict]) -> None:
    """
    Verifica se o dataset contem exatamente 300 exemplos (100 por split) e se todos os contextos
    de pedido satisfazem o contrato Pydantic OrderGoldContext.
    """
    assert len(benchmark_data) == 300, f"Esperado 300 casos, encontrado {len(benchmark_data)}"

    counts = {"real_standard": 0, "real_ambiguous": 0, "adversarial_injection": 0}
    for item in benchmark_data:
        counts[item["split"]] += 1
        # Valida se o dicionario aninhado respeita estritamente o contrato OrderGoldContext
        OrderGoldContext(**item["order_context"])

    assert counts["real_standard"] == 100
    assert counts["real_ambiguous"] == 100
    assert counts["adversarial_injection"] == 100


def test_benchmark_has_zero_training_data_leakage(benchmark_data: list[dict]) -> None:
    """
    Prova matematica de que nenhum pedido incluido no benchmark pertenceu aos primeiros 70%
    usados no treinamento do modelo base LightGBM.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        all_ids = con.execute("""
            SELECT order_id
            FROM gold.order_dispute_context
            ORDER BY purchase_ts ASC, order_id ASC;
        """).fetchall()
    finally:
        con.close()

    train_cutoff = int(len(all_ids) * 0.70)
    training_order_ids = {row[0] for row in all_ids[:train_cutoff]}

    benchmark_order_ids = {item["order_id"] for item in benchmark_data}
    overlap = training_order_ids.intersection(benchmark_order_ids)

    assert len(overlap) == 0, f"Detectado vazamento de {len(overlap)} pedidos do treino no benchmark!"


def test_router_accuracy_across_all_300_benchmark_cases(benchmark_data: list[dict]) -> None:
    """
    Executa o HybridCascadeRouter sobre todos os 300 casos do benchmark e verifica se o
    roteamento entre Guardrail, Tier 1 e Tier 2 atinge concordancia com o esperado.
    """
    router = HybridCascadeRouter()

    for item in benchmark_data:
        req = DisputeRequest(
            order_id=item["order_id"],
            customer_claim_text=item["customer_claim_text"]
        )
        ctx = OrderGoldContext(**item["order_context"])
        outcome = router.route_and_evaluate(req, ctx)

        assert outcome.routing_tier.value == item["expected_routing_tier"], (
            f"Falha de roteamento no caso {item['eval_id']} ({item['scenario_tag']}): "
            f"esperado {item['expected_routing_tier']}, obtido {outcome.routing_tier.value}"
        )
