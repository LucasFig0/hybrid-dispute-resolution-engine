"""
Definicao dos contratos de dados estritos (Pydantic v2 Schemas) do Motor de Disputas.

Responsabilidades:
1. Padronizar os Enums de acao de resolucao (ResolutionAction) e camada de decisao (RoutingTier).
2. Validar o payload de entrada da requisicao de disputa (DisputeRequest).
3. Tipar o contexto enriquecido extraido da camada Gold do DuckDB (OrderGoldContext).
4. Garantir o contrato de saida estruturada do Agente e do Roteador (AgentStructuredDecision e DisputeResponse).
"""

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResolutionAction(str, Enum):
    """
    Acoes finais permitidas para uma disputa financeira/operacional.
    """
    APPROVED_FULL_REFUND = "APPROVED_FULL_REFUND"
    APPROVED_PARTIAL_CREDIT = "APPROVED_PARTIAL_CREDIT"
    REJECTED_POLICY_VIOLATION = "REJECTED_POLICY_VIOLATION"
    ESCALATE_TO_HUMAN_ANALYST = "ESCALATE_TO_HUMAN_ANALYST"


class RoutingTier(str, Enum):
    """
    Camada arquitetural responsavel pela tomada de decisao final.
    """
    TIER_1_DETERMINISTIC_ML = "TIER_1_DETERMINISTIC_ML"
    TIER_2_LLM_AGENT = "TIER_2_LLM_AGENT"
    BLOCKED_SECURITY_GUARDRAIL = "BLOCKED_SECURITY_GUARDRAIL"


class PolicyClauseId(str, Enum):
    """
    Identificadores oficiais das clausulas da Politica de Reembolso (refund_policy_v1.md).
    Garante que nem o Roteador nem o LLM inventem clausulas inexistentes (Zero Hallucination).
    """
    POL_LOG_01 = "POL-LOG-01"
    POL_LOG_02 = "POL-LOG-02"
    POL_LOG_03 = "POL-LOG-03"
    POL_ITEM_01 = "POL-ITEM-01"
    POL_ITEM_02 = "POL-ITEM-02"
    POL_FRAUD_01 = "POL-FRAUD-01"
    POL_SEC_01 = "POL-SEC-01"


class DisputeRequest(BaseModel):
    """
    Contrato de entrada da API ao receber uma abertura de disputa.
    """
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    order_id: str = Field(
        ...,
        min_length=8,
        max_length=64,
        description="Identificador unico do pedido na camada Gold do DuckDB"
    )
    customer_claim_text: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Relato textual livre enviado pelo cliente na abertura da disputa"
    )


class OrderGoldContext(BaseModel):
    """
    Representacao tipada dos dados reais do pedido recuperados da tabela gold.order_dispute_context.
    """
    model_config = ConfigDict(extra="ignore")

    order_id: str
    customer_unique_id: str
    seller_id: str
    item_count: int = Field(..., ge=1)
    distinct_products_count: int = Field(..., ge=1)
    distinct_sellers_count: int = Field(..., ge=1)
    is_multi_seller_order: int = Field(..., ge=0, le=1)
    total_items_price: float = Field(..., gt=0.0)
    total_freight_value: float = Field(..., ge=0.0)
    payment_value: float = Field(..., gt=0.0)
    max_installments: int = Field(..., ge=1)
    delivery_delay_hours: float
    total_delivery_duration_hours: float = Field(..., ge=0.0)
    freight_to_payment_ratio: float = Field(..., ge=0.0)
    seller_prior_orders_count: int = Field(..., ge=0)
    seller_historical_defect_rate: float = Field(..., ge=0.0, le=1.0)
    customer_prior_orders_count: int = Field(..., ge=0)
    customer_prior_disputes_count: int = Field(..., ge=0)


class AgentStructuredDecision(BaseModel):
    """
    Contrato estrito de decisao exigido tanto do Roteador Tier 1 quanto do Agente GenAI Tier 2.
    """
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    action: ResolutionAction = Field(
        ...,
        description="Decisao final tomada sobre o pedido de reembolso"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Grau de confianca na decisao no intervalo [0.0, 1.0]"
    )
    policy_clause_cited: PolicyClauseId = Field(
        ...,
        description="Codigo exato da clausula da politica corporativa aplicada"
    )
    inconsistency_detected: bool = Field(
        ...,
        description="True se o relato do cliente contradiz a telemetria ou viola seguranca"
    )
    audit_justification_ptbr: str = Field(
        ...,
        min_length=15,
        max_length=600,
        description="Justificativa tecnica concisa em portugues para trilha de auditoria financeira"
    )

    @field_validator("confidence_score")
    @classmethod
    def round_confidence(cls, value: float) -> float:
        """
        Padroniza o score de confianca em 4 casas decimais.
        """
        return round(float(value), 4)


class DisputeResponse(BaseModel):
    """
    Resposta completa retornada pela API com telemetria de latencia, custo e decisao auditavel.
    """
    model_config = ConfigDict(extra="forbid")

    order_id: str
    routing_tier: RoutingTier
    tabular_risk_score: float = Field(..., ge=0.0, le=1.0)
    decision: AgentStructuredDecision
    latency_ms: float = Field(..., ge=0.0)
    estimated_cost_usd: float = Field(..., ge=0.0)
