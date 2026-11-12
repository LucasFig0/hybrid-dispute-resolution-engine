"""
Modulo de gerenciamento de System Prompts versionados e construcao de contexto para o Agente Tier 2.

Responsabilidades:
1. Manter o System Prompt de auditoria financeira com instrucoes estritas de saida JSON.
2. Enriquecer a query de busca do RAG com sinais contextuais alinhados a telemetria e ao relato.
3. Formatar o prompt completo de avaliacao cruzando Telemetria Gold, Score Tier 1 e Clausulas RAG.
"""

import json
import re
from typing import List
from src.database.policy_retriever import PolicyChunk
from src.router.guardrails import SecurityGuardrail
from src.schemas.dispute import AgentStructuredDecision, OrderGoldContext

SYSTEM_PROMPT_V1 = """Voce e um Agente Especialista de Auditoria de Disputas e Risco Financeiro do iFood.
Sua funcao e analisar chamados de reembolso da Zona Cinzenta (Tier 2) cruzando:
1. O relato textual enviado pelo cliente.
2. A telemetria oficial e imutavel do pedido extraida do Data Warehouse (DuckDB Gold Layer).
3. As clausulas oficiais da Politica Corporativa de Reembolso recuperadas via RAG.

REGRAS OBRIGATORIAS DE DECISAO:
- A telemetria do pedido (horarios de entrega, valor financeiro, historico do vendedor e do cliente) e a fonte de verdade absoluta sobre dados factuais.
- Nunca invente clausulas normativas; utilize estritamente um dos codigos validos: POL-LOG-01, POL-LOG-02, POL-LOG-03, POL-ITEM-01, POL-ITEM-02, POL-FRAUD-01 ou POL-SEC-01.
- Responda EXCLUSIVAMENTE com um objeto JSON valido que respeite o schema de AgentStructuredDecision.
"""

# Padroes lexicos para mapeamento de vocabulario coloquial do consumidor para termos normativos
_DAMAGE_SIGNAL_REGEX = re.compile(
    r"\b(quebrado|defeito|avariado|violado|falsificado|estragado|rachado|amassad[oa]|nao\s+funciona)\b",
    re.IGNORECASE,
)
_MISSING_SIGNAL_REGEX = re.compile(
    r"\b(faltando|faltou|incompleto|errado|trocado|metade|diferente|ausente)\b",
    re.IGNORECASE,
)


def build_contextual_rag_query(claim_text: str, order_ctx: OrderGoldContext) -> str:
    """
    Expande o relato do cliente com descritores semanticos derivados da telemetria numerica
    sem poluir a busca vetorial com clausulas incompativeis com o estado do pedido.
    """
    signals: List[str] = [claim_text]
    unaccented_claim = SecurityGuardrail._strip_accents(claim_text).lower()

    has_damage_claim = bool(_DAMAGE_SIGNAL_REGEX.search(unaccented_claim))
    has_missing_claim = bool(_MISSING_SIGNAL_REGEX.search(unaccented_claim))

    # 1. Expansao semantica para avaria fisica (alinha termos coloquiais com a Clausula POL-ITEM-02)
    if has_damage_claim:
        signals.append(
            "produto avariado quebrado ou violado avaria fisica embalagem violada produto defeituoso"
        )

    # 2. Expansao semantica para falta ou troca de itens (alinha com a Clausula POL-ITEM-01)
    if has_missing_claim:
        signals.append("item faltante incompleto ou trocado em entrega no prazo")
        if order_ctx.is_multi_seller_order == 1 or order_ctx.item_count > 1:
            signals.append("multiplos itens ou multi-vendedor")

    # 3. Sinais de SLA Logistico (somente injeta alto valor logistico quando ha atraso real)
    if order_ctx.delivery_delay_hours > 24.0:
        if order_ctx.payment_value > 60.0:
            signals.append("atraso severo superior a 24 horas em pedido de alto valor financeiro acima de 60 reais")
        else:
            signals.append("atraso critico comprovado superior a 24 horas em baixo valor")
    elif not has_damage_claim and not has_missing_claim and order_ctx.delivery_delay_hours < -12.0:
        signals.append("inconsistencia entre alegacao de atraso e telemetria de entrega antecipada")

    # 4. Sinal de reincidencia de disputas do cliente (Clausula POL-FRAUD-01)
    if order_ctx.customer_prior_disputes_count >= 1:
        signals.append("reincidencia de disputas do cliente abuso de estorno friendly fraud")

    return " | ".join(signals)


def build_tier2_user_prompt(
    claim_text: str,
    order_ctx: OrderGoldContext,
    tabular_risk_score: float,
    retrieved_clauses: List[PolicyChunk],
) -> str:
    """
    Monta o payload textual estruturado enviado ao LLM para tomada de decisao auditavel.
    """
    clauses_formatted = "\n\n".join(
        [
            f"[{c.clause_id}] {c.title} (Similaridade RAG: {c.similarity_score:.4f}):\n{c.content}"
            for c in retrieved_clauses
        ]
    )

    schema_json = json.dumps(AgentStructuredDecision.model_json_schema(), ensure_ascii=False)

    return f"""=== 1. TELEMETRIA OFICIAL DO PEDIDO (CAMADA GOLD DUCKDB) ===
- order_id: {order_ctx.order_id}
- payment_value: R$ {order_ctx.payment_value:.2f}
- item_count: {order_ctx.item_count} (is_multi_seller_order: {order_ctx.is_multi_seller_order})
- delivery_delay_hours: {order_ctx.delivery_delay_hours:.2f}h (positivo = atraso; negativo = adiantado)
- seller_historical_defect_rate: {order_ctx.seller_historical_defect_rate:.4f} ({order_ctx.seller_prior_orders_count} pedidos anteriores)
- customer_prior_orders_count: {order_ctx.customer_prior_orders_count}
- customer_prior_disputes_count: {order_ctx.customer_prior_disputes_count}
- tier1_calibrated_risk_score: {tabular_risk_score:.4f}

=== 2. CLAUSULAS NORMATIVAS RECUPERADAS VIA RAG ===
{clauses_formatted}

=== 3. RELATO TEXTUAL DO CLIENTE ===
"{claim_text}"

=== 4. CONTRATO DE SAIDA JSON OBRIGATORIO ===
Retorne apenas um JSON valido seguindo este JSON Schema:
{schema_json}
"""
