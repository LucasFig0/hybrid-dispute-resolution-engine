# Relatorio Oficial de Benchmark: Baseline 100% LLM vs. Arquitetura Hibrida

Avaliacao executada sobre **300 casos reais de Holdout (Out-of-Time)** extraidos da camada Gold (Olist E-Commerce Dataset), divididos igualmente entre `real_standard` (100), `real_ambiguous` (100) e `adversarial_injection` (100).

## 1. Comparativo Executivo de Unit Economics e Eficiencia

| Metrica | Baseline (100% Direct LLM) | Arquitetura Hibrida (Proposta) | Impacto / Ganho |
| :--- | :--- | :--- | :--- |
| **Chamadas ao LLM (em 300 disputas)** | 300 (100.0%) | 100 (33.33%) | **-66.67% chamadas de LLM** |
| **Tokens Consumidos (300 disputas)** | 423,478 tokens | 141,203 tokens | **-66.64% consumo de tokens** |
| **Custo por 1.000 Disputas (USD)** | $0.8464 | $0.2823 | **Economia de 66.64%** |
| **Bloqueio de Prompt Injection** | Vulneravel (Exposto ao LLM) | 100.00% Bloqueado no Guardrail | **Protecao Deterministica O(1)** |
| **Conformidade de Schema (Pydantic)** | 100.00% | 100.00% | **Contrato Tipado Estrito** |

## 2. SLAs de Latencia por Camada de Decisao

| Camada da Arquitetura | Volume Processado | Latencia P95 (ms) | Meta RFC-001 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Security Guardrail (`POL-SEC-01`)** | 100 casos (33.3%) | 0.118 ms | < 5.0 ms | APROVADO |
| **Tier 1: LightGBM Calibrado** | 100 casos (33.3%) | 7.509 ms | < 35.0 ms | APROVADO |
| **Tier 2: Agente GenAI + Contextual RAG** | 100 casos (33.3%) | 11.494 ms | < 2500.0 ms | APROVADO |
| **Sistema Hibrido Consolidado** | **300 casos (100%)** | **10.728 ms (P50: 6.532 ms)** | **< 1000.0 ms** | **APROVADO** |
