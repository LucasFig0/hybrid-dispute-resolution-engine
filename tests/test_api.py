"""
Suite de testes de integracao ponta a ponta para a API FastAPI (src/api/main.py).

Objetivo:
Validar os contratos HTTP, codigos de status (200, 404, 422), roteamento hibrido completo
com dados reais do DuckDB e contabilizacao de telemetria no endpoint /v1/metrics.
"""

import json
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from src.api.main import app, telemetry
from src.schemas.dispute import PolicyClauseId, ResolutionAction, RoutingTier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "dispute_eval_benchmark.jsonl"

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_benchmark_cases() -> dict[str, dict]:
    """
    Seleciona um exemplo real de cada split do benchmark para testar a API ponta a ponta.
    """
    with open(BENCHMARK_JSONL_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    return {
        "standard": next(r for r in rows if r["split"] == "real_standard"),
        "ambiguous": next(r for r in rows if r["split"] == "real_ambiguous"),
        "adversarial": next(r for r in rows if r["split"] == "adversarial_injection"),
    }


def test_api_health_check_endpoint() -> None:
    """
    Verifica se o endpoint /health retorna HTTP 200 com metadados do modelo Tier 1 e RAG.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["tier1_features_count"] == 15
    assert data["rag_clauses_indexed"] == 7


def test_api_returns_404_for_non_existent_order_id() -> None:
    """
    Garante que o envio de um order_id inexistente na camada Gold retorna HTTP 404 tratavel.
    """
    payload = {
        "order_id": "non_existent_order_id_99999",
        "customer_claim_text": "Meu pedido atrasou muito e quero solicitar o reembolso integral."
    }
    response = client.post("/v1/disputes/evaluate", json=payload)
    assert response.status_code == 404
    assert "nao encontrado" in response.json()["detail"]


def test_api_blocks_adversarial_injection_with_zero_cost(sample_benchmark_cases: dict[str, dict]) -> None:
    """
    Testa via HTTP um caso adversarial real do benchmark, validando bloqueio pelo Guardrail
    com custo de LLM igual a zero.
    """
    adv_case = sample_benchmark_cases["adversarial"]
    response = client.post(
        "/v1/disputes/evaluate",
        json={
            "order_id": adv_case["order_id"],
            "customer_claim_text": adv_case["customer_claim_text"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["routing_tier"] == RoutingTier.BLOCKED_SECURITY_GUARDRAIL.value
    assert body["decision"]["action"] == ResolutionAction.REJECTED_POLICY_VIOLATION.value
    assert body["decision"]["policy_clause_cited"] == PolicyClauseId.POL_SEC_01.value
    assert body["estimated_cost_usd"] == 0.0


def test_api_end_to_end_tier1_and_tier2_with_metrics_tracking(sample_benchmark_cases: dict[str, dict]) -> None:
    """
    Executa requisicoes reais para o Tier 1 e para o Tier 2 e verifica se o endpoint
    GET /v1/metrics contabiliza corretamente o Offload Rate e o custo acumulado.
    """
    # Reinicia contadores de telemetria para validar medicao exata
    telemetry.__init__()

    # 1. Requisicao resolvida no Tier 1 (Custo $0)
    std_case = sample_benchmark_cases["standard"]
    resp_t1 = client.post(
        "/v1/disputes/evaluate",
        json={
            "order_id": std_case["order_id"],
            "customer_claim_text": std_case["customer_claim_text"],
        },
    )
    assert resp_t1.status_code == 200
    body_t1 = resp_t1.json()
    assert body_t1["routing_tier"] == RoutingTier.TIER_1_DETERMINISTIC_ML.value
    assert body_t1["estimated_cost_usd"] == 0.0

    # 2. Requisicao escalada para o Agente Tier 2 (Custo > $0)
    amb_case = sample_benchmark_cases["ambiguous"]
    resp_t2 = client.post(
        "/v1/disputes/evaluate",
        json={
            "order_id": amb_case["order_id"],
            "customer_claim_text": amb_case["customer_claim_text"],
        },
    )
    assert resp_t2.status_code == 200
    body_t2 = resp_t2.json()
    assert body_t2["routing_tier"] == RoutingTier.TIER_2_LLM_AGENT.value
    assert body_t2["estimated_cost_usd"] > 0.0

    # 3. Verifica metricas consolidadas no endpoint GET /v1/metrics
    metrics_resp = client.get("/v1/metrics")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["total_requests"] == 2
    assert metrics["tier1_resolved_count"] == 1
    assert metrics["tier2_agent_count"] == 1
    assert metrics["llm_offload_rate"] == 0.50
    assert metrics["total_estimated_cost_usd"] > 0.0
