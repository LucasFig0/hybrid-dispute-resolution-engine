---
language:
- pt
license: mit
task_categories:
- text-classification
- tabular-classification
tags:
- ecommerce
- dispute-resolution
- prompt-injection
- brazilian-portuguese
dataset_info:
  features:
  - name: eval_id
    dtype: large_string
  - name: split
    dtype: large_string
  - name: scenario_tag
    dtype: large_string
  - name: order_id
    dtype: large_string
  - name: purchase_ts
    dtype: large_string
  - name: customer_claim_text
    dtype: large_string
  - name: expected_routing_tier
    dtype: large_string
  - name: expected_policy_clause
    dtype: large_string
  - name: tabular_risk_score
    dtype: float64
  - name: order_context
    struct:
    - name: order_id
      dtype: string
    - name: customer_unique_id
      dtype: string
    - name: seller_id
      dtype: string
    - name: item_count
      dtype: int64
    - name: distinct_products_count
      dtype: int64
    - name: distinct_sellers_count
      dtype: int64
    - name: is_multi_seller_order
      dtype: int64
    - name: total_items_price
      dtype: float64
    - name: total_freight_value
      dtype: float64
    - name: payment_value
      dtype: float64
    - name: max_installments
      dtype: int64
    - name: delivery_delay_hours
      dtype: float64
    - name: total_delivery_duration_hours
      dtype: float64
    - name: freight_to_payment_ratio
      dtype: float64
    - name: seller_prior_orders_count
      dtype: int64
    - name: seller_historical_defect_rate
      dtype: float64
    - name: customer_prior_orders_count
      dtype: int64
    - name: customer_prior_disputes_count
      dtype: int64
  splits:
  - name: real_standard
    num_bytes: 54539
    num_examples: 100
  - name: real_ambiguous
    num_bytes: 59171
    num_examples: 100
  - name: adversarial_injection
    num_bytes: 63805
    num_examples: 100
  download_size: 127485
  dataset_size: 177515
configs:
- config_name: default
  data_files:
  - split: real_standard
    path: data/real_standard-*
  - split: real_ambiguous
    path: data/real_ambiguous-*
  - split: adversarial_injection
    path: data/adversarial_injection-*
---

# Brazilian E-Commerce Dispute & Adversarial Evaluation Benchmark (PT-BR)

Benchmark de avaliacao reproduzivel para sistemas hibridos de resolucao de disputas e estornos financeiros (**Tier 1 Calibrated LightGBM + Tier 2 GenAI RAG Agent**).

Construido sobre o conjunto **Out-of-Time Holdout** (30% finais cronologicos) do *Brazilian E-Commerce Public Dataset by Olist*, garantindo **Zero Data Leakage** em relacao aos 70% iniciais utilizados no treinamento do modelo preditivo tabular.

## 1. Estrutura dos Splits (300 Casos Balanceados)

| Split | Volume | Cenario Operacional e Financeiro | Camada Alvo Esperada |
| :--- | :--- | :--- | :--- |
| `real_standard` | 100 | Disputas logisticas claras dentro do teto de alcada financeira. | `TIER_1_DETERMINISTIC_ML` |
| `real_ambiguous` | 100 | Casos da Zona Cinzenta: item faltante/avariado ou alto valor (> R$ 120). | `TIER_2_LLM_AGENT` |
| `adversarial_injection` | 100 | Reclamacoes reais concatenadas com vetores de Prompt Injection (POL-SEC-01). | `BLOCKED_SECURITY_GUARDRAIL` |

## 2. Como Utilizar via Hugging Face Datasets

```python
from datasets import load_dataset

dataset = load_dataset("LucasFig0/brazilian-ecommerce-dispute-evals")
print(dataset)
```
