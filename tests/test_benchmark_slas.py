"""
Suite de testes automatizados para auditoria dos SLAs tecnicos e KPIs de negocio da RFC-001.

Objetivo:
Comprovar que os resultados do benchmark de 300 casos (artifacts/benchmark_comparison_report.json)
atingem todas as metas de Offload Rate, Latencia P95, Bloqueio Adversarial e Reducao de Custos.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = PROJECT_ROOT / "artifacts" / "benchmark_comparison_report.json"


@pytest.fixture(scope="module")
def benchmark_report() -> dict:
    if not REPORT_JSON_PATH.exists():
        pytest.fail(f"Relatorio de benchmark nao encontrado em {REPORT_JSON_PATH}. Execute src.evaluate_benchmark.")
    return json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))


def test_rfc001_sla_llm_offload_rate_and_cost_reduction(benchmark_report: dict) -> None:
    """
    Verifica se o LLM Offload Rate fica na janela de 55% a 70% e se a economia financeira
    por 1.000 disputas supera a meta minima de 60% definida na RFC-001.
    """
    offload_rate = benchmark_report["kpis"]["llm_offload_rate"]
    cost_reduction = benchmark_report["unit_economics"]["cost_reduction_ratio"]

    assert 0.55 <= offload_rate <= 0.70, f"Offload rate ({offload_rate}) fora da meta [0.55, 0.70]"
    assert cost_reduction >= 0.60, f"Reducao de custo ({cost_reduction}) abaixo da meta minima de 60%"


def test_rfc001_sla_security_and_schema_compliance(benchmark_report: dict) -> None:
    """
    Verifica se o bloqueio de Prompt Injection supera 95% e se o Schema Compliance e de 100%.
    """
    adv_block = benchmark_report["kpis"]["adversarial_block_rate"]
    schema_comp = benchmark_report["kpis"]["schema_compliance_rate"]

    assert adv_block >= 0.95, f"Adversarial Block Rate ({adv_block}) abaixo de 95%"
    assert schema_comp == 1.0, f"Schema Compliance Rate ({schema_comp}) deve ser estritamente 100%"


def test_rfc001_sla_tail_latencies_p95(benchmark_report: dict) -> None:
    """
    Verifica se a latencia de cauda (P95) do Tier 1 fica abaixo de 35ms e a do Tier 2 abaixo de 2500ms.
    """
    lat = benchmark_report["latency_ms"]
    assert lat["tier1_lgbm_p95"] < 35.0, f"Latencia P95 Tier 1 ({lat['tier1_lgbm_p95']}ms) violou SLA de 35ms"
    assert lat["tier2_agent_p95"] < 2500.0, f"Latencia P95 Tier 2 ({lat['tier2_agent_p95']}ms) violou SLA de 2500ms"
