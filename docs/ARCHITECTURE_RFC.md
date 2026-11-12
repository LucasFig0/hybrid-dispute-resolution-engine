# RFC-001 & Architecture Decision Records (ADRs)

**Projeto:** Hybrid Dispute & Refund Resolution Engine (HDRRE)  
**Status:** Implementado e Validado (36/36 Testes Automatizados Aprovados)  

---

## ADR-001: Arquitetura Medallion em DuckDB e Prevencao de Data Leakage Temporal

### Contexto
Bases brutas de e-commerce e delivery apresentam multiplas avaliacoes por pedido, multiplos vendedores no mesmo carrinho (multi-seller), textos com caracteres de controle (`\r\n`) e anomalias temporais onde a data de entrega registrada antecede a compra. Alem disso, calcular taxas historicas de falha usando agregacoes globais (`GROUP BY seller_id`) causa vazamento de dados do futuro (*Temporal Data Leakage*).

### Decisao
Implementar tres schemas fisicamente isolados no DuckDB (`bronze`, `silver`, `gold`):
1. **Camada Bronze (`99.441` pedidos):** Ingestao imutavel 1:1 dos CSVs da Olist com colunas de linhagem (`_ingested_at`, `_source_file`).
2. **Camada Silver (`95.824` pedidos validados; `3.617` expurgados no Quality Gate):**
   - Desambiguacao deterministica de pedidos multi-seller (`1.261` casos) atribuindo o `primary_seller_id` ao item de maior valor financeiro via `ARG_MAX(seller_id, price)`.
   - Higienizacao de `39.608` relatos em portugues via `REGEXP_REPLACE`.
   - Expurgo de pedidos nao entregues, datas nulas e inversoes cronologicas (`delivered_ts < purchase_ts`).
3. **Camada Gold (`95.824` registros e 15 features preditivas):**
   - Aplicacao de *Window Functions* estritas (`ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`) ordenadas por `purchase_ts ASC, order_id ASC` tanto para o vendedor quanto para o cliente (`3.077` pedidos de clientes recorrentes).
   - Suavizacao Bayesiana (`prior = 0.1000`) na taxa de defeito do vendedor: `(falhas_anteriores + 1.0) / (pedidos_anteriores + 10.0)`.

---

## ADR-002: Calibracao Isotonica e Governanca de Alcada Financeira no Tier 1

### Contexto
Em problemas de classificacao desbalanceados (taxa basal de falha severa `review_score == 1` de `9,76%` na base completa e `~6,8%` no periodo Out-of-Time de meados de 2018), probabilidades brutas de arvores de decisao nao refletem a frequencia empirica real. Alem disso, regras automaticas nao podem aprovar ou negar pedidos de alto valor financeiro sem leitura semantica do relato do cliente.

### Decisao
1. **Divisao Out-of-Time (70% Treino / 15% Calibracao / 15% Teste Futuro):** O modelo `LGBMClassifier` e ajustado nos primeiros 70% e calibrado com **Regressao Isotonica** (`CalibratedClassifierCV`) nos 15% intermediarios, reduzindo o Brier Score para `0.0590` e atingindo `ROC-AUC = 0.7145` e `PR-AUC = 0.2495` no teste futuro.
2. **Travas de Alcada Financeira e Conteudo Fisico:**
   - **Clausula `POL-LOG-01` (Auto-Approve):** Restrita a atrasos comprovados superiores a 24h, valor pago ate `R$ 60,00` e probabilidade calibrada maior ou igual a `0.35` (~4,5x Lift sobre a taxa base).
   - **Clausula `POL-LOG-03` (Fast-Track Reject):** Restrita a entregas realizadas com mais de 12h de antecedencia, valor pago dentro do teto de alcada automatica (`<= R$ 120,00`), risco calibrado `<= 0.06`, zero disputas anteriores do cliente e **ausencia de termos de avaria fisica ou falta de item** no relato.

---

## ADR-003: Contextual RAG sem Poluicao de Query no Tier 2

### Contexto
Quando um chamado sobe para a Zona Cinzenta (Tier 2), os clientes utilizam termos coloquiais (*"caixa amassada", "nao funciona", "veio metade"*) que podem apresentar baixa sobreposicao lexica com o texto juridico/normativo das politicas corporativas. Por outro lado, injetar metadados numericos indiscriminadamente na query do RAG polui o ranking vetorial.

### Decisao
Construir um motor RAG leve em espaco vetorial TF-IDF de unigramas e bigramas (`RefundPolicyRetriever`) combinado com expansao semantica condicional (`build_contextual_rag_query`), mapeando expressoes coloquiais de avaria para descritores formais da `POL-ITEM-02` e condicionando sinais de atraso em alto valor (`POL-LOG-02`) apenas quando `delivery_delay_hours > 24.0`.
