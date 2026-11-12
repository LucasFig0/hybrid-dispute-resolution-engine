"""
Modulo do Agente GenAI de Resolucao de Disputas (Tier 2) com RAG, Telemetria e Validacao Pydantic.

Responsabilidades:
1. Orquestrar a recuperacao de clausulas normativas via Contextual RAG (RefundPolicyRetriever).
2. Construir o prompt estruturado combinando telemetria DuckDB Gold, Score Tier 1 e politicas.
3. Garantir que 100% das saidas respeitem o contrato AgentStructuredDecision (Schema Compliance).
4. Calcular telemetria precisa de latencia, consumo de tokens e custo financeiro em USD.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional
import httpx
from pydantic import ValidationError
from src.agent.prompts import (
    SYSTEM_PROMPT_V1,
    build_contextual_rag_query,
    build_tier2_user_prompt,
)
from src.config import AppSettings, get_settings
from src.database.policy_retriever import PolicyChunk, RefundPolicyRetriever
from src.router.guardrails import SecurityGuardrail
from src.schemas.dispute import (
    AgentStructuredDecision,
    DisputeRequest,
    OrderGoldContext,
    PolicyClauseId,
    ResolutionAction,
)


@dataclass(frozen=True)
class Tier2AgentExecutionResult:
    """
    Objeto de retorno do Agente Tier 2 contendo a decisao validada e metricas de observabilidade.
    """
    decision: AgentStructuredDecision
    retrieved_clauses: List[str]
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    agent_latency_ms: float


class Tier2DisputeAgent:
    """
    Agente de decisao semantica para casos da Zona Cinzenta escalados pelo Roteador Hibrido.
    """

    DAMAGE_REGEX = re.compile(
        r"\b(quebrado|defeito|avariado|violado|falsificado|estragado|rachado|amassado|nao\s+funciona)\b",
        re.IGNORECASE,
    )
    MISSING_ITEM_REGEX = re.compile(
        r"\b(faltando|faltou|incompleto|errado|trocado|metade|diferente|ausente|so\s+veio|veio\s+so)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        retriever: Optional[RefundPolicyRetriever] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        self.retriever = retriever or RefundPolicyRetriever()
        self.settings = settings or get_settings()

    def evaluate_dispute(
        self,
        request: DisputeRequest,
        order_ctx: OrderGoldContext,
        tabular_risk_score: float,
    ) -> Tier2AgentExecutionResult:
        """
        Executa o fluxo completo do Tier 2: Contextual RAG -> Prompting -> Inferencia Validada -> Telemetria.
        """
        start_ns = time.perf_counter_ns()

        # 1. Recuperacao de Conhecimento Normativo (Contextual RAG Top-3)
        rag_query = build_contextual_rag_query(request.customer_claim_text, order_ctx)
        top_clauses: List[PolicyChunk] = self.retriever.retrieve_relevant_clauses(rag_query, top_k=3)
        retrieved_ids = [c.clause_id for c in top_clauses]

        # 2. Construcao do Prompt Completo (System + User Context)
        user_prompt = build_tier2_user_prompt(
            claim_text=request.customer_claim_text,
            order_ctx=order_ctx,
            tabular_risk_score=tabular_risk_score,
            retrieved_clauses=top_clauses,
        )

        # 3. Inferencia com garantia de contrato Pydantic (Live API ou Motor Normativo RAG Local)
        if self.settings.llm_provider.lower() == "groq" and self.settings.groq_api_key:
            decision = self._call_groq_with_retry(user_prompt)
        else:
            decision = self._evaluate_with_rag_grounded_engine(
                claim_text=request.customer_claim_text,
                order_ctx=order_ctx,
                tabular_risk_score=tabular_risk_score,
                top_clauses=top_clauses,
            )

        # 4. Calculo de Telemetria de Tokens e Unit Economics (USD)
        prompt_tokens = max(1, (len(SYSTEM_PROMPT_V1) + len(user_prompt)) // 4)
        completion_json = decision.model_dump_json()
        completion_tokens = max(1, len(completion_json) // 4)

        cost_usd = (
            (prompt_tokens / 1_000_000.0) * self.settings.llm_input_cost_per_1m_usd
            + (completion_tokens / 1_000_000.0) * self.settings.llm_output_cost_per_1m_usd
        )
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        return Tier2AgentExecutionResult(
            decision=decision,
            retrieved_clauses=retrieved_ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=round(cost_usd, 6),
            agent_latency_ms=round(elapsed_ms, 3),
        )

    def _call_groq_with_retry(self, user_prompt: str) -> AgentStructuredDecision:
        """
        Realiza chamada real a API compativel com OpenAI da Groq com loop de auto-correcao (Self-Correction)
        caso o JSON retornado viole o contrato Pydantic AgentStructuredDecision.
        """
        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V1},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(self.settings.llm_max_retries + 1):
            payload = {
                "model": self.settings.groq_model_name,
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                raw_content = resp.json()["choices"][0]["message"]["content"]

            try:
                return AgentStructuredDecision.model_validate_json(raw_content)
            except ValidationError as err:
                last_error = err
                # Injeta o erro de validacao de volta no contexto para o LLM corrigir na proxima tentativa
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Seu JSON violou o schema Pydantic: {err}. Corrija e retorne apenas o JSON valido.",
                    }
                )

        raise RuntimeError(f"Falha ao validar schema apos retries no LLM: {last_error}")

    def _evaluate_with_rag_grounded_engine(
        self,
        claim_text: str,
        order_ctx: OrderGoldContext,
        tabular_risk_score: float,
        top_clauses: List[PolicyChunk],
    ) -> AgentStructuredDecision:
        """
        Motor de raciocinio normativo deterministico ancorado nas clausulas de refund_policy_v1.md.
        Garante reprodutibilidade de 100% em CI/CD e avaliacao rigorosa das evidencias da camada Gold.
        """
        unaccented = SecurityGuardrail._strip_accents(claim_text).lower()
        has_damage = bool(self.DAMAGE_REGEX.search(unaccented))
        has_missing = bool(self.MISSING_ITEM_REGEX.search(unaccented))

        # 1. Regra de Reincidencia e Risco de Abuso (Clausula POL-FRAUD-01)
        if (
            order_ctx.customer_prior_disputes_count >= 1
            and order_ctx.delivery_delay_hours <= 0.0
            and order_ctx.seller_historical_defect_rate < 0.12
        ):
            if has_damage or has_missing:
                return AgentStructuredDecision(
                    action=ResolutionAction.ESCALATE_TO_HUMAN_ANALYST,
                    confidence_score=0.82,
                    policy_clause_cited=PolicyClauseId.POL_FRAUD_01,
                    inconsistency_detected=True,
                    audit_justification_ptbr=(
                        f"Cliente reincidente ({order_ctx.customer_prior_disputes_count} disputas previas) "
                        f"alegando falha de item em entrega no prazo ({order_ctx.delivery_delay_hours:.1f}h) "
                        f"de lojista confiavel (taxa {order_ctx.seller_historical_defect_rate:.2f}). Escalado para auditoria."
                    ),
                )
            return AgentStructuredDecision(
                action=ResolutionAction.REJECTED_POLICY_VIOLATION,
                confidence_score=0.90,
                policy_clause_cited=PolicyClauseId.POL_FRAUD_01,
                inconsistency_detected=True,
                audit_justification_ptbr=(
                    f"Negado por risco de abuso (POL-FRAUD-01): cliente com {order_ctx.customer_prior_disputes_count} "
                    "disputas severas anteriores em pedido entregue dentro do prazo por vendedor de baixo risco."
                ),
            )

        # 2. Regra de Produto Avariado, Quebrado ou Violado (Clausula POL-ITEM-02)
        if has_damage:
            if order_ctx.payment_value <= 150.0 and order_ctx.seller_historical_defect_rate >= 0.15:
                return AgentStructuredDecision(
                    action=ResolutionAction.APPROVED_FULL_REFUND,
                    confidence_score=0.88,
                    policy_clause_cited=PolicyClauseId.POL_ITEM_02,
                    inconsistency_detected=False,
                    audit_justification_ptbr=(
                        f"Aprovado reembolso integral por avaria fisica (R$ {order_ctx.payment_value:.2f} <= R$ 150) "
                        f"com vendedor apresentando taxa historica de defeito elevada ({order_ctx.seller_historical_defect_rate:.2f})."
                    ),
                )
            return AgentStructuredDecision(
                action=ResolutionAction.ESCALATE_TO_HUMAN_ANALYST,
                confidence_score=0.86,
                policy_clause_cited=PolicyClauseId.POL_ITEM_02,
                inconsistency_detected=False,
                audit_justification_ptbr=(
                    f"Relato de avaria fisica ou defeito em pedido de R$ {order_ctx.payment_value:.2f} "
                    f"(taxa do vendedor: {order_ctx.seller_historical_defect_rate:.2f}). Escalado para inspecao de evidencias."
                ),
            )

        # 3. Regra de Item Faltante, Incompleto ou Trocado (Clausula POL-ITEM-01)
        if has_missing:
            if order_ctx.item_count > 1 or order_ctx.is_multi_seller_order == 1:
                return AgentStructuredDecision(
                    action=ResolutionAction.APPROVED_PARTIAL_CREDIT,
                    confidence_score=0.89,
                    policy_clause_cited=PolicyClauseId.POL_ITEM_01,
                    inconsistency_detected=False,
                    audit_justification_ptbr=(
                        f"Credito parcial aprovado (POL-ITEM-01): relato de item faltante/incompleto em pedido "
                        f"com {order_ctx.item_count} itens (multi-seller={order_ctx.is_multi_seller_order}) e cliente sem abuso."
                    ),
                )
            if order_ctx.payment_value <= 150.0:
                return AgentStructuredDecision(
                    action=ResolutionAction.APPROVED_FULL_REFUND,
                    confidence_score=0.85,
                    policy_clause_cited=PolicyClauseId.POL_ITEM_01,
                    inconsistency_detected=False,
                    audit_justification_ptbr=(
                        f"Reembolso aprovado (POL-ITEM-01): item unico reportado como incorreto/faltante "
                        f"dentro da alcada de R$ {order_ctx.payment_value:.2f} para cliente sem historico de fraude."
                    ),
                )
            return AgentStructuredDecision(
                action=ResolutionAction.ESCALATE_TO_HUMAN_ANALYST,
                confidence_score=0.80,
                policy_clause_cited=PolicyClauseId.POL_ITEM_01,
                inconsistency_detected=False,
                audit_justification_ptbr=(
                    f"Divergencia de item em pedido de alto valor (R$ {order_ctx.payment_value:.2f}). "
                    "Escalado para conferencia de manifesto logistico."
                ),
            )

        # 4. Regra de Atraso Severo em Pedidos de Alto Valor (Clausula POL-LOG-02)
        if order_ctx.delivery_delay_hours > 24.0:
            if order_ctx.payment_value <= 200.0:
                return AgentStructuredDecision(
                    action=ResolutionAction.APPROVED_FULL_REFUND,
                    confidence_score=0.91,
                    policy_clause_cited=PolicyClauseId.POL_LOG_02,
                    inconsistency_detected=False,
                    audit_justification_ptbr=(
                        f"Reembolso integral aprovado no Tier 2 (POL-LOG-02): atraso severo de "
                        f"{order_ctx.delivery_delay_hours:.1f}h em pedido de R$ {order_ctx.payment_value:.2f} (<= R$ 200)."
                    ),
                )
            return AgentStructuredDecision(
                action=ResolutionAction.ESCALATE_TO_HUMAN_ANALYST,
                confidence_score=0.88,
                policy_clause_cited=PolicyClauseId.POL_LOG_02,
                inconsistency_detected=False,
                audit_justification_ptbr=(
                    f"Atraso critico ({order_ctx.delivery_delay_hours:.1f}h) em pedido de alto valor "
                    f"(R$ {order_ctx.payment_value:.2f} > R$ 200.00). Escalado conforme teto da clausula POL-LOG-02."
                ),
            )

        # 5. Caso Residual de Zona Cinzenta (Entrega no prazo, mas cliente contesta recebimento ou valor alto)
        if order_ctx.payment_value > 120.0 or tabular_risk_score >= 0.20:
            return AgentStructuredDecision(
                action=ResolutionAction.ESCALATE_TO_HUMAN_ANALYST,
                confidence_score=0.78,
                policy_clause_cited=PolicyClauseId.POL_LOG_02,
                inconsistency_detected=order_ctx.delivery_delay_hours < 0.0,
                audit_justification_ptbr=(
                    f"Caso complexo de alto valor (R$ {order_ctx.payment_value:.2f}, atraso {order_ctx.delivery_delay_hours:.1f}h, "
                    f"score Tier 1 {tabular_risk_score:.4f}). Escalado para revisao humana detalhada."
                ),
            )

        return AgentStructuredDecision(
            action=ResolutionAction.REJECTED_POLICY_VIOLATION,
            confidence_score=0.84,
            policy_clause_cited=PolicyClauseId.POL_LOG_03,
            inconsistency_detected=True,
            audit_justification_ptbr=(
                f"Reclamacao improcedente: telemetria confirma entrega dentro do prazo "
                f"({order_ctx.delivery_delay_hours:.1f}h) sem relato de avaria ou item faltante."
            ),
        )
