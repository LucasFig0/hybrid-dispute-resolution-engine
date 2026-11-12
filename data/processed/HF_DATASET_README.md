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
- agent-evaluation
---

# Brazilian E-Commerce Dispute & Adversarial Evaluation Benchmark (PT-BR)

Benchmark de avaliacao para sistemas hibridos de resolucao de disputas e reembolsos (Tier 1 LightGBM + Tier 2 LLM Agent).
Construido sobre o conjunto Out-of-Time (Holdout pos-treino) do Brazilian E-Commerce Public Dataset by Olist.

## Estrutura dos Splits (300 Casos)
- `real_standard` (100 casos): Disputas logisticas deterministas para avaliacao de LLM Offload Rate no Tier 1.
- `real_ambiguous` (100 casos): Disputas complexas envolvendo avaria fisica, item faltante em entrega no prazo ou alto valor financeiro (Tier 2).
- `adversarial_injection` (100 casos): Ataques de Prompt Injection e engenharia social em PT-BR/EN acoplados a pedidos reais.
