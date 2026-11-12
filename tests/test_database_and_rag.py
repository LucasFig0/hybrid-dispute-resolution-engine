"""
Suite de testes automatizados para o Repositorio DuckDB e o Motor RAG de Politicas (Tier 2).
"""

import duckdb
from pathlib import Path
import pytest
from src.database.duckdb_client import DuckDBDisputeRepository
from src.database.policy_retriever import RefundPolicyRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "dispute_warehouse.duckdb"


def test_duckdb_repository_fetches_valid_gold_order() -> None:
    """
    Verifica se o DuckDBDisputeRepository recupera corretamente um pedido real da camada Gold
    e popula o contrato OrderGoldContext sem erros de tipagem.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        sample_order_id = con.execute(
            "SELECT order_id FROM gold.order_dispute_context LIMIT 1"
        ).fetchone()[0]
    finally:
        con.close()

    repo = DuckDBDisputeRepository()
    context = repo.get_order_context(sample_order_id)

    assert context is not None
    assert context.order_id == sample_order_id
    assert context.payment_value > 0.0
    assert 0.0 <= context.seller_historical_defect_rate <= 1.0


def test_duckdb_repository_returns_none_for_unknown_order() -> None:
    """
    Garante que a consulta por um order_id inexistente retorna None sem levantar excecao SQL.
    """
    repo = DuckDBDisputeRepository()
    context = repo.get_order_context("non_existent_order_999999")
    assert context is None


def test_policy_retriever_parses_all_seven_clauses() -> None:
    """
    Verifica se o parser do Markdown extrai exatamente as 7 clausulas corporativas oficiais.
    """
    retriever = RefundPolicyRetriever()
    extracted_ids = [chunk.clause_id for chunk in retriever.chunks]

    expected_ids = [
        "POL-LOG-01",
        "POL-LOG-02",
        "POL-LOG-03",
        "POL-ITEM-01",
        "POL-ITEM-02",
        "POL-FRAUD-01",
        "POL-SEC-01",
    ]
    assert extracted_ids == expected_ids


def test_policy_retriever_semantic_ranking_accuracy() -> None:
    """
    Valida se a busca vetorial TF-IDF recupera a clausula normativa esperada no Top-1
    para diferentes cenarios de disputa do Tier 2.
    """
    retriever = RefundPolicyRetriever()

    # Caso 1: Reclama de item faltante / incompleto em entrega no prazo
    res_missing = retriever.retrieve_relevant_clauses(
        "Entrega no prazo, porem veio faltando itens na sacola e pedido incompleto",
        top_k=2,
    )
    assert res_missing[0].clause_id == "POL-ITEM-01"
    assert res_missing[0].similarity_score > 0.10

    # Caso 2: Reclama de produto quebrado, violado ou avariado
    res_damaged = retriever.retrieve_relevant_clauses(
        "O produto chegou quebrado, com embalagem violada e avaria fisica grave",
        top_k=2,
    )
    assert res_damaged[0].clause_id == "POL-ITEM-02"

    # Caso 3: Atraso severo em pedido de alto valor financeiro
    res_high_val_delay = retriever.retrieve_relevant_clauses(
        "Atraso severo superior a 24 horas em pedido de alto valor financeiro acima de 60 reais",
        top_k=2,
    )
    assert res_high_val_delay[0].clause_id == "POL-LOG-02"
