# Hybrid Dispute & Refund Resolution Engine (iFood GenAI Portfolio)

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark%20Dataset%20(300%20Cases)-yellow)](https://huggingface.co/datasets/LucasFig0/brazilian-ecommerce-dispute-evals)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Medallion%20%26%20LightGBM%20Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/figlucasazevedo/hybrid-dispute-engine-tier-1)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Pytest-36%20Passed-brightgreen)]()

Motor de decisao em cascata (**Tiered Hybrid Routing Architecture**) projetado para resolver disputas financeiras e operacionais de pos-entrega em marketplaces de alta volumetria. A arquitetura combina **barreira deterministica Anti-Prompt Injection**, **inferencia tabular calibrada de ultrabaixa latencia (Tier 1 - LightGBM)** e **Agente GenAI tipado com Contextual RAG (Tier 2 - Pydantic + FastAPI)**.

---

## 1. Problema de Negocio e Solucao de Engenharia (Unit Economics)

Roteamento ingenuo de 100% dos tickets de suporte para LLMs de grande porte inviabiliza a margem operacional (*Unit Economics*) em pedidos de ticket baixo, aumenta a latencia de cauda (P95/P99) e expoe o sistema de reembolsos a engenharia social (*Prompt Injection*).

Este sistema resolve o trade-off entre **custo de inferencia**, **latencia** e **qualidade de auditoria** dividindo a decisao em tres barreiras:
1. **Security Guardrail (`POL-SEC-01`):** Bloqueia tentativas de *Prompt Injection*, *Jailbreak* e falsificacao de alcada administrativa em menos de `1 ms` a custo zero.
2. **Tier 1 — Classificador LightGBM Calibrado + Regras de Alcada (`POL-LOG-01` / `POL-LOG-03`):** Consome 15 features da camada Gold do DuckDB e resolve disputas logisticas claras dentro do teto financeiro em `~7,5 ms` (P95) sem acionar o LLM.
3. **Tier 2 — Agente GenAI + Contextual RAG (`POL-LOG-02` / `POL-ITEM-01` / `POL-ITEM-02` / `POL-FRAUD-01`):** Acionado exclusivamente para casos da Zona Cinzenta (divergencia de conteudo fisico como item faltante/avariado, alto valor financeiro acima de R$ 120,00 ou reincidencia de disputas), retornando saida estritamente validada via Pydantic (`AgentStructuredDecision`).

---

## 2. Resultados Empiricos no Benchmark Out-of-Time (300 Casos Holdout)

Avaliacao reproduzivel executada pelo script `src/evaluate_benchmark.py` sobre 300 casos reais de Holdout (posteriores ao corte de treinamento do modelo), divididos entre `real_standard` (100), `real_ambiguous` (100) e `adversarial_injection` (100):

| Metrica / SLA | Baseline (100% Direct LLM) | Arquitetura Hibrida (Proposta) | Meta RFC-001 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **LLM Offload Rate** | 0,00% (300 chamadas LLM) | **66,67%** (Apenas 100 chamadas LLM) | 55% a 70% | APROVADO |
| **Custo por 1.000 Disputas (USD)** | $0,8464 USD | **$0,2823 USD** | Reducao >= 60% | **-66,64% Custo** |
| **Adversarial Block Rate** | 0,00% (Exposto ao LLM) | **100,00%** (100/100 bloqueados) | > 95% | APROVADO |
| **Schema Compliance Rate** | 100,00% | **100,00%** (Contrato Pydantic) | 100% | APROVADO |
| **Latencia P95 Tier 1 (LightGBM)** | N/A | **7,509 ms** | < 35,0 ms | APROVADO |

### Metricas do Modelo Tabular Tier 1 (Conjunto Out-of-Time: 14.374 Pedidos)
- **Dataset Base:** *Brazilian E-Commerce Public Dataset by Olist* (99.441 pedidos brutos -> 95.824 validados na Silver/Gold).
- **Split Temporal Estrito:** 70% Treino (67.076) | 15% Calibracao Isotonica (14.374) | 15% Teste Futuro OOT (14.374).
- **ROC-AUC (OOT):** `0.7145` | **PR-AUC (OOT):** `0.2495` | **Brier Score Calibrado:** `0.0590`.
- **Top 5 Features (Split Importance):** `delivery_delay_hours` (1268), `seller_historical_defect_rate` (1245), `total_delivery_duration_hours` (1235), `seller_prior_orders_count` (1134), `freight_to_payment_ratio` (1050).

---

## 3. Arquitetura do Repositorio

```text
ifood-genai-dispute-engine/
├── artifacts/
│   ├── lgbm_tier1_calibrated.joblib         # Modelo LightGBM + Calibrador Isotonico serializado
│   └── benchmark_comparison_report.json     # Telemetria auditavel dos 300 casos de Holdout
├── data/
│   ├── policies/refund_policy_v1.md         # Base normativa (7 clausulas POL-*) indexada pelo RAG
│   └── processed/
│       ├── dispute_warehouse.duckdb         # Data Warehouse Medallion (Schemas: bronze, silver, gold)
│       ├── dispute_eval_benchmark.jsonl     # Benchmark com 300 casos (Zero Training Leakage)
│       └── HF_DATASET_README.md             # Dataset Card para publicacao no Hugging Face Hub
├── docs/
│   ├── ARCHITECTURE_RFC.md                  # Especificacao tecnica e ADRs de engenharia
│   └── BENCHMARK_RESULTS.md                 # Tabelas comparativas geradas automaticamente
├── notebooks/
│   └── 01_olist_medallion_and_calibrated_lgbm_tier1.ipynb  # Notebook publico (Kaggle)
├── src/
│   ├── config.py                            # Configuracoes tipadas via pydantic-settings
│   ├── evaluate_benchmark.py                # Motor de benchmark comparativo (Baseline vs Hibrido)
│   ├── schemas/dispute.py                   # Contratos Pydantic v2 (Entrada, Contexto Gold e Saida)
│   ├── database/
│   │   ├── build_warehouse.py               # Pipeline ETL Medallion (Bronze -> Silver -> Gold)
│   │   ├── build_hf_eval_dataset.py         # Construtor e uploader do benchmark para o Hugging Face
│   │   ├── duckdb_client.py                 # Repositorio de leitura de baixa latencia na camada Gold
│   │   └── policy_retriever.py              # Motor RAG vetorial TF-IDF bigram sobre clausulas POL-*
│   ├── router/
│   │   ├── guardrails.py                    # Barreira heuristica O(1) contra Prompt Injection
│   │   ├── train_tier1_model.py             # Treinamento Out-of-Time e calibracao isotonica LightGBM
│   │   ├── tabular_predictor.py             # Wrapper Singleton de inferencia online (< 35ms)
│   │   └── cascade_router.py                # Orquestrador de decisao e alcada financeira Tier 1 vs Tier 2
│   ├── agent/
│   │   ├── prompts.py                       # System prompts e enriquecimento semantico Contextual RAG
│   │   └── dispute_agent.py                 # Agente Tier 2 (suporte Groq API + motor normativo RAG local)
│   └── api/main.py                          # Microsservico FastAPI (/v1/disputes/evaluate, /v1/metrics, /health)
├── tests/                                   # Suite com 36 testes automatizados (pytest)
├── Dockerfile                               # Imagem de producao (python:3.11-slim, non-root)
└── requirements.txt
```

---

## 4. Como Executar Localmente (Reprodutibilidade Completa)

```powershell
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Construir o Data Warehouse Medallion (Bronze, Silver e Gold) no DuckDB
.\.venv\Scripts\python.exe -m src.database.build_warehouse

# 3. Treinar e calibrar o modelo LightGBM do Tier 1 (Out-of-Time Split)
.\.venv\Scripts\python.exe -m src.router.train_tier1_model

# 4. Gerar o dataset de benchmark (300 casos) e rodar a avaliacao comparativa
.\.venv\Scripts\python.exe -m src.database.build_hf_eval_dataset
.\.venv\Scripts\python.exe -m src.evaluate_benchmark

# 5. Executar a suite completa de 36 testes automatizados (Medallion, ML, Router, RAG, API e SLAs)
.\.venv\Scripts\python.exe -m pytest -v

# 6. Iniciar o microsservico FastAPI na porta 8000
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --port 8000
```

