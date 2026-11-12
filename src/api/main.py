"""
Microsservico FastAPI do Motor Hibrido de Resolucao de Disputas (Tier 1 LightGBM + Tier 2 Agent RAG).

Responsabilidades:
1. Expor o endpoint principal POST /v1/disputes/evaluate com validacao estrita Pydantic.
2. Integrar consulta a camada Gold (DuckDB), Roteador Hibrido em Cascata e Agente Tier 2.
3. Registrar metricas operacionais e financeiras em tempo real (Offload Rate, Latencia e Unit Economics).
4. Expor endpoints de observabilidade (GET /v1/metrics) e saude do servico (GET /health).
"""

import time
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, status
from src.agent.dispute_agent import Tier2DisputeAgent
from src.config import get_settings
from src.database.duckdb_client import DuckDBDisputeRepository
from src.router.cascade_router import HybridCascadeRouter
from src.schemas.dispute import (
    DisputeRequest,
    DisputeResponse,
    RoutingTier,
)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Motor de decisao em cascata para resolucao de disputas e reembolsos combinando "
        "inferencia tabular calibrada (Tier 1 LightGBM) e Agente GenAI com Contextual RAG (Tier 2)."
    ),
)

# Inicializacao Singleton dos componentes para baixa latencia em producao
repository = DuckDBDisputeRepository()
cascade_router = HybridCascadeRouter()
tier2_agent = Tier2DisputeAgent(settings=settings)


class TelemetryTracker:
    """
    Acumulador em memoria para observabilidade de SLAs tecnicos e KPIs financeiros da API.
    """

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.tier1_resolved_count: int = 0
        self.guardrail_blocked_count: int = 0
        self.tier2_agent_count: int = 0
        self.total_cost_usd: float = 0.0
        self.total_latency_ms: float = 0.0

    def record(self, tier: RoutingTier, latency_ms: float, cost_usd: float) -> None:
        """
        Registra os indicadores de uma requisicao processada.
        """
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.total_cost_usd += cost_usd

        if tier == RoutingTier.TIER_1_DETERMINISTIC_ML:
            self.tier1_resolved_count += 1
        elif tier == RoutingTier.BLOCKED_SECURITY_GUARDRAIL:
            self.guardrail_blocked_count += 1
        elif tier == RoutingTier.TIER_2_LLM_AGENT:
            self.tier2_agent_count += 1

    def snapshot(self) -> Dict[str, Any]:
        """
        Retorna o resumo consolidado das metricas de operacao e Unit Economics.
        """
        if self.total_requests == 0:
            return {
                "total_requests": 0,
                "llm_offload_rate": 0.0,
                "tier1_resolved_count": 0,
                "guardrail_blocked_count": 0,
                "tier2_agent_count": 0,
                "avg_latency_ms": 0.0,
                "total_estimated_cost_usd": 0.0,
            }

        offloaded = self.tier1_resolved_count + self.guardrail_blocked_count
        return {
            "total_requests": self.total_requests,
            "llm_offload_rate": round(offloaded / self.total_requests, 4),
            "tier1_resolved_count": self.tier1_resolved_count,
            "guardrail_blocked_count": self.guardrail_blocked_count,
            "tier2_agent_count": self.tier2_agent_count,
            "avg_latency_ms": round(self.total_latency_ms / self.total_requests, 3),
            "total_estimated_cost_usd": round(self.total_cost_usd, 6),
        }


telemetry = TelemetryTracker()


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """
    Verifica a disponibilidade do servico, modelo calibrado Tier 1 e repositorio DuckDB.
    """
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "llm_provider": settings.llm_provider,
        "tier1_features_count": len(cascade_router.predictor.feature_names),
        "rag_clauses_indexed": len(tier2_agent.retriever.chunks),
    }


@app.get("/v1/metrics", status_code=status.HTTP_200_OK)
def get_operational_metrics() -> Dict[str, Any]:
    """
    Retorna os indicadores em tempo real de Offload Rate, latencia media e custo acumulado.
    """
    return telemetry.snapshot()


@app.post(
    "/v1/disputes/evaluate",
    response_model=DisputeResponse,
    status_code=status.HTTP_200_OK,
)
def evaluate_dispute_endpoint(payload: DisputeRequest) -> DisputeResponse:
    """
    Endpoint principal de resolucao de disputas:
    1. Recupera o contexto imutavel do pedido na camada Gold do DuckDB.
    2. Executa o Roteador Hibrido (Guardrail Anti-Injection + LightGBM Calibrado Tier 1).
    3. Caso necessario, escala para o Agente GenAI com Contextual RAG no Tier 2.
    """
    start_ns = time.perf_counter_ns()

    # 1. Busca de contexto transacional e historico na camada Gold
    order_ctx = repository.get_order_context(payload.order_id)
    if order_ctx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pedido '{payload.order_id}' nao encontrado na camada Gold do Data Warehouse.",
        )

    # 2. Avaliacao pelo Roteador em Cascata (Guardrail + Tier 1 LightGBM)
    router_outcome = cascade_router.route_and_evaluate(payload, order_ctx)

    if not router_outcome.requires_tier2_agent and router_outcome.decision is not None:
        total_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)
        telemetry.record(router_outcome.routing_tier, total_ms, cost_usd=0.0)
        return DisputeResponse(
            order_id=payload.order_id,
            routing_tier=router_outcome.routing_tier,
            tabular_risk_score=router_outcome.tabular_risk_score,
            decision=router_outcome.decision,
            latency_ms=total_ms,
            estimated_cost_usd=0.0,
        )

    # 3. Escalonamento para Zona Cinzenta: Agente GenAI + Contextual RAG (Tier 2)
    agent_outcome = tier2_agent.evaluate_dispute(
        request=payload,
        order_ctx=order_ctx,
        tabular_risk_score=router_outcome.tabular_risk_score,
    )
    total_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)
    telemetry.record(
        RoutingTier.TIER_2_LLM_AGENT,
        total_ms,
        cost_usd=agent_outcome.estimated_cost_usd,
    )

    return DisputeResponse(
        order_id=payload.order_id,
        routing_tier=RoutingTier.TIER_2_LLM_AGENT,
        tabular_risk_score=router_outcome.tabular_risk_score,
        decision=agent_outcome.decision,
        latency_ms=total_ms,
        estimated_cost_usd=agent_outcome.estimated_cost_usd,
    )
