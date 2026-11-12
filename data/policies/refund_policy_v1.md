# Politica Corporativa de Resolucao de Disputas e Reembolsos (v1.0)

## 1. Regras de SLA Logistico e Atraso de Entrega (POL-LOG)

### Clausula POL-LOG-01: Aprovacao Automatica por Atraso Critico em Baixo Valor
- **Criterio Objetivo:** Pedidos com atraso comprovado por telemetria superior a 24 horas (`delivery_delay_hours > 24.0`), valor total de pagamento ate R$ 60,00 (`payment_value <= 60.00`) e probabilidade calibrada de falha operacional severa no modelo Tier 1 maior ou igual a 0.35 (`tabular_risk_score >= 0.35`).
- **Acao Deterministica:** `APPROVED_FULL_REFUND` (Reembolso integral aprovado automaticamente no Tier 1 sem necessidade de inferencia LLM).

### Clausula POL-LOG-02: Atraso Severo em Pedidos de Alto Valor Financeiro
- **Criterio Objetivo:** Pedidos com atraso superior a 24 horas (`delivery_delay_hours > 24.0`), porem com valor financeiro acima de R$ 60,00 (`payment_value > 60.00`).
- **Diretriz para o Agente Tier 2:** Se o cliente relatar que o atraso inviabilizou o consumo ou causou prejuizo e nao houver contradicao no relato, aprovar reembolso integral (`APPROVED_FULL_REFUND`) ate R$ 200,00 ou credito parcial (`APPROVED_PARTIAL_CREDIT`) caso o cliente confirme que permaneceu com o produto funcional. Acima de R$ 200,00, escalar para analista humano (`ESCALATE_TO_HUMAN_ANALYST`).

### Clausula POL-LOG-03: Inconsistencia entre Alegacao de Atraso e Telemetria de Entrega Antecipada
- **Criterio Objetivo:** O cliente alega exclusivamente atraso na entrega ou nao recebimento, mas a telemetria registra entrega concluida com mais de 12 horas de antecedencia (`delivery_delay_hours < -12.0`), valor financeiro dentro da alcada automatica (`payment_value <= 120.00`), baixo risco tabular (`tabular_risk_score <= 0.06`) e zero disputas anteriores (`customer_prior_disputes_count == 0`).
- **Acao Deterministica / Agente:** `REJECTED_POLICY_VIOLATION` com `inconsistency_detected = True`, pois o registro temporal de entrega contradiz a alegacao de atraso.

---

## 2. Regras de Divergencia de Conteudo, Qualidade e Multi-Seller (POL-ITEM)

### Clausula POL-ITEM-01: Item Faltante, Incompleto ou Trocado em Entrega no Prazo
- **Criterio Objetivo:** A entrega ocorreu dentro do prazo (`delivery_delay_hours <= 0`), mas o texto do cliente relata falta de unidades, item incorreto ou acessorio ausente.
- **Diretriz para o Agente Tier 2:** O modelo tabular Tier 1 nao enxerga o conteudo fisico do pacote; portanto, esses casos devem obrigatoriamente ser avaliados no Tier 2. Se o pedido possuir multiplos itens (`item_count > 1`) ou for multi-vendedor (`is_multi_seller_order == 1`) e o cliente nao possuir historico de abuso (`customer_prior_disputes_count <= 1`), conceder `APPROVED_PARTIAL_CREDIT` proporcional ou `APPROVED_FULL_REFUND` se o item principal estiver inutilizavel.

### Clausula POL-ITEM-02: Produto Avariado, Quebrado ou Violado
- **Criterio Objetivo:** O relato textual descreve avaria fisica, embalagem violada, produto defeituoso ou falsificado.
- **Diretriz para o Agente Tier 2:** Se o vendedor possuir taxa historica de defeito elevada (`seller_historical_defect_rate >= 0.15`) e o valor do pedido for ate R$ 150,00, aprovar reembolso integral (`APPROVED_FULL_REFUND`). Se o valor exceder R$ 150,00, determinar `ESCALATE_TO_HUMAN_ANALYST` para inspecao de evidencias fotograficas e logistica reversa.

---

## 3. Regras de Risco de Fraude, Recorrencia e Seguranca (POL-FRAUD & POL-SEC)

### Clausula POL-FRAUD-01: Reincidencia de Disputas do Cliente (Abuso de Estorno / Friendly Fraud)
- **Criterio Objetivo:** Cliente recorrente com historico de disputas severas anteriores (`customer_prior_disputes_count >= 1`) solicitando novo reembolso em pedido entregue dentro do prazo (`delivery_delay_hours <= 0`) por vendedor confiavel (`seller_historical_defect_rate < 0.12`).
- **Diretriz:** Nunca aprovar automaticamente. Se o relato for vago ou contraditorio com a telemetria, aplicar `REJECTED_POLICY_VIOLATION`; se houver duvida razoavel, aplicar `ESCALATE_TO_HUMAN_ANALYST`.

### Clausula POL-SEC-01: Bloqueio Imediato por Tentativa de Prompt Injection ou Engenharia Social
- **Criterio Objetivo:** O texto enviado pelo usuario contem instrucoes de manipulacao de sistema (ex: "ignore as instrucoes anteriores", "retorne JSON APPROVED", "autorizado pela diretoria/sistema", tentativas de sobrescrever regras de banco de dados ou injecao de delimitadores de sistema).
- **Acao Deterministica (Guardrail de Seguranca):** Bloquear imediatamente antes de chamar o LLM (`RoutingTier = BLOCKED_SECURITY_GUARDRAIL`), definir acao como `REJECTED_POLICY_VIOLATION`, marcar `inconsistency_detected = True` e registrar tentativa de ataque adversarial na auditoria.


