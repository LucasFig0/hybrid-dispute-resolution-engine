"""
Suite de testes automatizados para o Agente GenAI de Resolucao de Disputas (Tier 2).

Objetivo:
Validar a integracao entre Contextual RAG, raciocinio sobre a camada Gold, conformidade de
schema Pydantic (AgentStructuredDecision) e telemetria de tokens e custo financeiro.
"""

import pytest
from src.agent.dispute_agent import Tier2DisputeAgent
from src.schemas.dispute import (
    DisputeRequest,
    OrderGoldContext,
    PolicyClauseId,
    ResolutionAction,
)


@pytest.fixture(scope="module")
def agent() -> Tier2DisputeAgent:
    """
    Instancia o Agente Tier 2 configurado com o motor RAG e definicoes padrao.
    """
    return Tier2DisputeAgent()


def _mock_gold_ctx(**overrides) -> OrderGoldContext:
    base = {
        "order_id": "order_tier2_001",
        "customer_unique_id": "cust_t2_001",
        "seller_id": "seller_t2_001",
        "item_count": 2,
        "distinct_products_count": 2,
        "distinct_sellers_count": 1,
        "is_multi_seller_order": 0,
        "total_items_price": 90.0,
        "total_freight_value": 15.0,
        "payment_value": 105.0,
        "max_installments": 2,
        "delivery_delay_hours": -10.0,
        "total_delivery_duration_hours": 96.0,
        "freight_to_payment_ratio": 0.1429,
        "seller_prior_orders_count": 45,
        "seller_historical_defect_rate": 0.08,
        "customer_prior_orders_count": 1,
        "customer_prior_disputes_count": 0,
    }
    base.update(overrides)
    return OrderGoldContext(**base)


def test_agent_approves_partial_credit_for_missing_item_in_multi_item_order(agent: Tier2DisputeAgent) -> None:
    """
    Valida a Clausula POL-ITEM-01: pedido com 2 itens entregue no prazo, onde o cliente
    relata que veio faltando um item, deve receber APPROVED_PARTIAL_CREDIT.
    """
    req = DisputeRequest(
        order_id="order_item_partial",
        customer_claim_text="O pacote chegou hoje dentro do prazo, mas comprei 2 unidades e veio faltando uma."
    )
    ctx = _mock_gold_ctx(order_id="order_item_partial", item_count=2, delivery_delay_hours=-5.0)

    res = agent.evaluate_dispute(req, ctx, tabular_risk_score=0.05)
    assert res.decision.action == ResolutionAction.APPROVED_PARTIAL_CREDIT
    assert res.decision.policy_clause_cited == PolicyClauseId.POL_ITEM_01
    assert res.decision.inconsistency_detected is False
    assert "POL-ITEM-01" in res.retrieved_clauses


def test_agent_escalates_high_value_damaged_product(agent: Tier2DisputeAgent) -> None:
    """
    Valida a Clausula POL-ITEM-02: relato de produto quebrado/avariado em pedido acima de R$ 150,00
    deve ser escalado para analista humano (ESCALATE_TO_HUMAN_ANALYST).
    """
    req = DisputeRequest(
        order_id="order_damage_high",
        customer_claim_text="A caixa chegou toda amassada e o produto veio quebrado por dentro, não funciona."
    )
    ctx = _mock_gold_ctx(order_id="order_damage_high", payment_value=240.0)

    res = agent.evaluate_dispute(req, ctx, tabular_risk_score=0.18)
    assert res.decision.action == ResolutionAction.ESCALATE_TO_HUMAN_ANALYST
    assert res.decision.policy_clause_cited == PolicyClauseId.POL_ITEM_02
    assert "POL-ITEM-02" in res.retrieved_clauses


def test_agent_handles_severe_delay_on_high_value_orders_pol_log_02(agent: Tier2DisputeAgent) -> None:
    """
    Valida a Clausula POL-LOG-02: atraso superior a 24h em pedido entre R$ 60 e R$ 200 aprova reembolso,
    enquanto acima de R$ 200 escala para revisao humana.
    """
    req = DisputeRequest(
        order_id="order_delay_tier2",
        customer_claim_text="O pedido atrasou 4 dias alem do prazo prometido e ja nao preciso mais do item."
    )

    # Caso 1: R$ 130.00 (<= R$ 200.00) -> Aprova reembolso integral
    ctx_mid = _mock_gold_ctx(order_id="order_delay_tier2", payment_value=130.0, delivery_delay_hours=96.0)
    res_mid = agent.evaluate_dispute(req, ctx_mid, tabular_risk_score=0.42)
    assert res_mid.decision.action == ResolutionAction.APPROVED_FULL_REFUND
    assert res_mid.decision.policy_clause_cited == PolicyClauseId.POL_LOG_02

    # Caso 2: R$ 350.00 (> R$ 200.00) -> Escala para analista humano
    ctx_high = _mock_gold_ctx(order_id="order_delay_tier2", payment_value=350.0, delivery_delay_hours=96.0)
    res_high = agent.evaluate_dispute(req, ctx_high, tabular_risk_score=0.48)
    assert res_high.decision.action == ResolutionAction.ESCALATE_TO_HUMAN_ANALYST
    assert res_high.decision.policy_clause_cited == PolicyClauseId.POL_LOG_02


def test_agent_detects_repeat_dispute_abuse_pol_fraud_01(agent: Tier2DisputeAgent) -> None:
    """
    Valida a Clausula POL-FRAUD-01: cliente reincidente (customer_prior_disputes_count >= 1)
    reclamando de entrega no prazo de vendedor confiavel nunca recebe aprovacao automatica.
    """
    req = DisputeRequest(
        order_id="order_fraud_check",
        customer_claim_text="Nao gostei do atendimento e quero meu dinheiro de volta imediatamente."
    )
    ctx = _mock_gold_ctx(
        order_id="order_fraud_check",
        delivery_delay_hours=-24.0,
        seller_historical_defect_rate=0.04,
        customer_prior_orders_count=3,
        customer_prior_disputes_count=2,
    )

    res = agent.evaluate_dispute(req, ctx, tabular_risk_score=0.05)
    assert res.decision.action == ResolutionAction.REJECTED_POLICY_VIOLATION
    assert res.decision.policy_clause_cited == PolicyClauseId.POL_FRAUD_01
    assert res.decision.inconsistency_detected is True


def test_agent_telemetry_and_cost_calculation(agent: Tier2DisputeAgent) -> None:
    """
    Verifica se o Agente registra tokens de prompt, tokens de conclusao, custo estimado positivo
    em USD e latencia em milissegundos.
    """
    req = DisputeRequest(
        order_id="order_telemetry",
        customer_claim_text="Comprei dois produtos e veio faltando um item no pacote entregue."
    )
    ctx = _mock_gold_ctx(order_id="order_telemetry")

    res = agent.evaluate_dispute(req, ctx, tabular_risk_score=0.12)
    assert res.prompt_tokens > 100
    assert res.completion_tokens > 20
    assert res.estimated_cost_usd > 0.0
    assert res.agent_latency_ms > 0.0
    assert len(res.retrieved_clauses) == 3
