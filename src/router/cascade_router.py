"""
Modulo de Roteamento Hibrido em Cascata (Tier 1 Deterministico + ML vs. Tier 2 Agente LLM).

Responsabilidades:
1. Executar o Guardrail de Seguranca (POL-SEC-01) bloqueando ataques de Prompt Injection.
2. Executar a inferencia tabular calibrada do LightGBM (Tier 1).
3. Resolver casos de alta confianca e dentro do teto de alcada financeira diretamente no Tier 1 (custo LLM $0).
4. Encaminhar casos de zona cinzenta, alto valor (> R$ 120) ou divergencia de conteudo fisico para o Tier 2.
"""

import re
import time
from dataclasses import dataclass
from typing import Optional
from src.router.guardrails import SecurityGuardrail
from src.router.tabular_predictor import Tier1TabularPredictor
from src.schemas.dispute import (
    AgentStructuredDecision,
    DisputeRequest,
    OrderGoldContext,
    PolicyClauseId,
    ResolutionAction,
    RoutingTier,
)


@dataclass(frozen=True)
class RouterEvaluationOutcome:
    """
    Resultado do processamento do Roteador em Cascata.
    Se requires_tier2_agent=False, o campo decision ja contem a decisao final auditavel.
    Se requires_tier2_agent=True, o orquestrador deve acionar o Agente LLM no Tier 2.
    """
    routing_tier: RoutingTier
    requires_tier2_agent: bool
    tabular_risk_score: float
    decision: Optional[AgentStructuredDecision]
    router_latency_ms: float


class HybridCascadeRouter:
    """
    Implementa a arquitetura de decisao em cascata para otimizacao de Unit Economics e latencia.
    """

    # Palavras-chave que indicam problema fisico com o conteudo/produto (invisivel a telemetria de entrega)
    CONTENT_ISSUE_REGEX = re.compile(
        r"\b(faltando|faltou|incompleto|errado|trocado|quebrado|defeito|avariado|violado|falsificado|estragado|diferente)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        predictor: Optional[Tier1TabularPredictor] = None,
        guardrail: Optional[SecurityGuardrail] = None,
    ) -> None:
        self.predictor = predictor or Tier1TabularPredictor()
        self.guardrail = guardrail or SecurityGuardrail()

    def _has_physical_content_claim(self, text: str) -> bool:
        """
        Verifica se o texto relata avaria fisica, item faltante ou produto trocado.
        Esses casos exigem leitura semantica no Tier 2 mesmo que a entrega tenha ocorrido no prazo.
        """
        unaccented = SecurityGuardrail._strip_accents(text)
        return bool(self.CONTENT_ISSUE_REGEX.search(unaccented))

    def route_and_evaluate(
        self,
        request: DisputeRequest,
        order_ctx: OrderGoldContext,
    ) -> RouterEvaluationOutcome:
        """
        Avalia a disputa aplicando o Guardrail e as regras de negocio + ML calibrado do Tier 1.
        """
        start_ns = time.perf_counter_ns()

        # 1. Barreira de Seguranca: Inspecao Anti-Prompt Injection (POL-SEC-01)
        guard_res = self.guardrail.inspect(request.customer_claim_text)
        if guard_res.is_blocked:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            blocked_decision = AgentStructuredDecision(
                action=ResolutionAction.REJECTED_POLICY_VIOLATION,
                confidence_score=1.0,
                policy_clause_cited=PolicyClauseId.POL_SEC_01,
                inconsistency_detected=True,
                audit_justification_ptbr=(
                    f"Bloqueio deterministico de seguranca (Guardrail: {guard_res.matched_rule_name}). "
                    "Detectada tentativa de manipulacao de instrucoes de sistema no relato."
                ),
            )
            return RouterEvaluationOutcome(
                routing_tier=RoutingTier.BLOCKED_SECURITY_GUARDRAIL,
                requires_tier2_agent=False,
                tabular_risk_score=0.0,
                decision=blocked_decision,
                router_latency_ms=round(elapsed_ms, 3),
            )

        # 2. Inferencia do Modelo Calibrado LightGBM (Tier 1)
        pred = self.predictor.predict_single(order_ctx.model_dump())
        risk_prob = pred.calibrated_probability

        # 3. Regra Tier 1 A (POL-LOG-01): Auto-Approve para atraso critico comprovado em baixo valor (<= R$ 60)
        if (
            order_ctx.delivery_delay_hours > 24.0
            and order_ctx.payment_value <= 60.0
            and risk_prob >= 0.35
        ):
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            approve_decision = AgentStructuredDecision(
                action=ResolutionAction.APPROVED_FULL_REFUND,
                confidence_score=max(risk_prob, 0.85),
                policy_clause_cited=PolicyClauseId.POL_LOG_01,
                inconsistency_detected=False,
                audit_justification_ptbr=(
                    f"Reembolso integral aprovado no Tier 1: atraso logistico comprovado de "
                    f"{order_ctx.delivery_delay_hours:.1f}h em pedido de baixo valor "
                    f"(R$ {order_ctx.payment_value:.2f}) com score de falha calibrado de {risk_prob:.4f}."
                ),
            )
            return RouterEvaluationOutcome(
                routing_tier=RoutingTier.TIER_1_DETERMINISTIC_ML,
                requires_tier2_agent=False,
                tabular_risk_score=risk_prob,
                decision=approve_decision,
                router_latency_ms=round(elapsed_ms, 3),
            )

        # 4. Regra Tier 1 B (POL-LOG-03): Fast-Track Reject para alegacao puramente logistica
        # Exige entrega com >12h de antecedencia, alcada financeira <= R$ 120.00, risco <= 0.06,
        # zero disputas previas do cliente e ausencia de reclamacao de item fisico.
        has_content_issue = self._has_physical_content_claim(request.customer_claim_text)
        if (
            order_ctx.delivery_delay_hours < -12.0
            and order_ctx.payment_value <= 120.0
            and risk_prob <= 0.06
            and order_ctx.customer_prior_disputes_count == 0
            and not has_content_issue
        ):
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            reject_decision = AgentStructuredDecision(
                action=ResolutionAction.REJECTED_POLICY_VIOLATION,
                confidence_score=round(1.0 - risk_prob, 4),
                policy_clause_cited=PolicyClauseId.POL_LOG_03,
                inconsistency_detected=True,
                audit_justification_ptbr=(
                    f"Solicitacao negada no Tier 1: telemetria comprova entrega realizada com "
                    f"{abs(order_ctx.delivery_delay_hours):.1f}h de antecedencia em pedido dentro da alcada "
                    f"(R$ {order_ctx.payment_value:.2f}, risco {risk_prob:.4f}) e sem relato de avaria fisica."
                ),
            )
            return RouterEvaluationOutcome(
                routing_tier=RoutingTier.TIER_1_DETERMINISTIC_ML,
                requires_tier2_agent=False,
                tabular_risk_score=risk_prob,
                decision=reject_decision,
                router_latency_ms=round(elapsed_ms, 3),
            )

        # 5. Zona Cinzenta / Alto Valor (> R$ 120) / Divergencia de Conteudo -> Escalar para Agente LLM (Tier 2)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return RouterEvaluationOutcome(
            routing_tier=RoutingTier.TIER_2_LLM_AGENT,
            requires_tier2_agent=True,
            tabular_risk_score=risk_prob,
            decision=None,
            router_latency_ms=round(elapsed_ms, 3),
        )
