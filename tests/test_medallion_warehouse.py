"""
Suite de testes automatizados de qualidade e integridade da Arquitetura Medallion (DuckDB).

Objetivo:
Validar os contratos de dados das camadas Bronze, Silver e Gold, garantindo ausencia de
duplicatas, cumprimento dos Quality Gates financeiros/temporais e prova matematica de
Zero Data Leakage nas Window Functions de vendedor e cliente.
"""

from pathlib import Path
import duckdb
import pytest

# Caminho absoluto para o banco de dados gerado pelo pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "dispute_warehouse.duckdb"


@pytest.fixture(scope="module")
def db_con() -> duckdb.DuckDBPyConnection:
    """
    Fixture que abre uma conexao somente leitura (read_only=True) com o banco DuckDB
    para execucao segura dos testes sem bloquear o arquivo no sistema operacional.
    """
    if not DB_PATH.exists():
        pytest.fail(f"Banco de dados nao encontrado em {DB_PATH}. Execute build_warehouse.py antes.")

    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()


def test_bronze_layer_lineage_and_volume(db_con: duckdb.DuckDBPyConnection) -> None:
    """
    Verifica se a camada Bronze preservou a contagem original de linhas do CSV da Olist
    e se os metadados obrigatorios de linhagem (_ingested_at e _source_file) nao possuem nulos.
    """
    row_count, null_metadata_count = db_con.execute("""
        SELECT 
            COUNT(*),
            SUM(CASE WHEN _ingested_at IS NULL OR _source_file IS NULL THEN 1 ELSE 0 END)
        FROM bronze.raw_orders;
    """).fetchone()

    assert row_count == 99441, f"Esperado 99441 linhas na bronze.raw_orders, encontrado {row_count}"
    assert null_metadata_count == 0, "Colunas de auditoria da camada Bronze possuem valores nulos"


def test_silver_layer_primary_key_uniqueness(db_con: duckdb.DuckDBPyConnection) -> None:
    """
    Garante que todas as tabelas da camada Silver possuem granularidade estrita de 1 linha por order_id,
    impedindo explosao cartesiana (fan-out) durante os joins.
    """
    silver_tables = [
        "silver.order_items_clean",
        "silver.order_payments_clean",
        "silver.order_reviews_clean",
        "silver.orders_validated"
    ]

    for table in silver_tables:
        total_rows, distinct_orders = db_con.execute(f"""
            SELECT COUNT(*), COUNT(DISTINCT order_id)
            FROM {table};
        """).fetchone()

        assert total_rows > 0, f"A tabela {table} esta vazia"
        assert total_rows == distinct_orders, (
            f"Violacao de chave primaria em {table}: {total_rows} linhas para {distinct_orders} order_ids"
        )


def test_silver_layer_quality_gates(db_con: duckdb.DuckDBPyConnection) -> None:
    """
    Valida se o Data Quality Gate da camada Silver expurgou 100% das anomalias cronologicas
    (entrega anterior a compra) e anomalias financeiras (valores zerados ou negativos).
    """
    anomalies = db_con.execute("""
        SELECT 
            SUM(CASE WHEN delivered_ts < purchase_ts THEN 1 ELSE 0 END) AS negative_delivery_time,
            SUM(CASE WHEN payment_value <= 0 OR total_items_price <= 0 THEN 1 ELSE 0 END) AS invalid_financials,
            SUM(CASE WHEN review_score NOT BETWEEN 1 AND 5 THEN 1 ELSE 0 END) AS invalid_scores
        FROM silver.orders_validated;
    """).fetchone()

    assert anomalies[0] == 0, f"Encontrados {anomalies[0]} pedidos com data de entrega anterior a compra"
    assert anomalies[1] == 0, f"Encontrados {anomalies[1]} pedidos com valores financeiros invalidos"
    assert anomalies[2] == 0, f"Encontrados {anomalies[2]} pedidos com review_score fora do intervalo [1, 5]"


def test_silver_reviews_text_sanitization(db_con: duckdb.DuckDBPyConnection) -> None:
    """
    Garante que o tratamento via expressao regular na camada Silver removeu todas as quebras
    de linha (\r, \n), tabulacoes (\t) e espacos duplos dos textos de avaliacao.
    """
    dirty_text_count = db_con.execute(r"""
        SELECT COUNT(*)
        FROM silver.order_reviews_clean
        WHERE REGEXP_MATCHES(clean_review_text, '[\r\n\t]')
           OR clean_review_text LIKE '%  %';
    """).fetchone()[0]

    assert dirty_text_count == 0, f"Encontrados {dirty_text_count} textos ainda com quebras de linha ou espacos duplos"


def test_gold_layer_completeness_and_cold_start_prior(db_con: duckdb.DuckDBPyConnection) -> None:
    """
    Verifica se a camada Gold preservou todos os registros validados da Silver e se a suavizacao
    Bayesiana aplica exatamente o prior 0.1000 para todo vendedor em seu primeiro pedido (cold start).
    """
    silver_count = db_con.execute("SELECT COUNT(*) FROM silver.orders_validated").fetchone()[0]
    gold_count = db_con.execute("SELECT COUNT(*) FROM gold.order_dispute_context").fetchone()[0]

    assert silver_count == gold_count == 95824, "Divergencia de contagem entre Silver e Gold"

    # Todo pedido onde seller_prior_orders_count == 0 deve ter taxa historica igual a (0 + 1) / (0 + 10) = 0.1
    invalid_cold_start = db_con.execute("""
        SELECT COUNT(*)
        FROM gold.order_dispute_context
        WHERE seller_prior_orders_count = 0
          AND seller_historical_defect_rate != 0.1000;
    """).fetchone()[0]

    assert invalid_cold_start == 0, "Falha no calculo do prior Bayesiano para vendedores sem historico previo"


def test_gold_seller_window_no_data_leakage(db_con: duckdb.DuckDBPyConnection) -> None:
    """
    Prova matematica de Zero Data Leakage na janela temporal do vendedor:
    Seleciona o vendedor com maior volume de pedidos e recalcula linha a linha em Python
    garantindo que o resultado da linha atual (is_severe_operational_failure) NUNCA entra
    no calculo de seller_historical_defect_rate da propria linha.
    """
    top_seller_id = db_con.execute("""
        SELECT seller_id
        FROM gold.order_dispute_context
        GROUP BY seller_id
        ORDER BY COUNT(*) DESC
        LIMIT 1;
    """).fetchone()[0]

    seller_orders = db_con.execute("""
        SELECT 
            order_id,
            is_severe_operational_failure,
            seller_prior_orders_count,
            seller_historical_defect_rate
        FROM gold.order_dispute_context
        WHERE seller_id = ?
        ORDER BY purchase_ts ASC, order_id ASC;
    """, [top_seller_id]).fetchall()

    running_orders = 0
    running_failures = 0

    for order_id, current_failure, db_prior_orders, db_defect_rate in seller_orders:
        expected_rate = round((running_failures + 1.0) / (running_orders + 10.0), 4)

        assert db_prior_orders == running_orders, (
            f"Leakage de contagem no pedido {order_id}: esperado {running_orders}, obtido {db_prior_orders}"
        )
        assert abs(db_defect_rate - expected_rate) < 1e-6, (
            f"Leakage de taxa no pedido {order_id}: esperado {expected_rate}, obtido {db_defect_rate}"
        )

        # Somente APOS validar a linha atual, acumulamos o resultado para os pedidos seguintes
        running_orders += 1
        running_failures += int(current_failure)


def test_gold_customer_window_no_data_leakage(db_con: duckdb.DuckDBPyConnection) -> None:
    """
    Prova matematica de Zero Data Leakage na janela temporal do cliente:
    Verifica se para o cliente mais recorrente da base, a contagem de pedidos e disputas
    anteriores reflete estritamente os eventos passados.
    """
    top_customer_id = db_con.execute("""
        SELECT customer_unique_id
        FROM gold.order_dispute_context
        GROUP BY customer_unique_id
        ORDER BY COUNT(*) DESC
        LIMIT 1;
    """).fetchone()[0]

    customer_orders = db_con.execute("""
        SELECT 
            order_id,
            is_severe_operational_failure,
            customer_prior_orders_count,
            customer_prior_disputes_count
        FROM gold.order_dispute_context
        WHERE customer_unique_id = ?
        ORDER BY purchase_ts ASC, order_id ASC;
    """, [top_customer_id]).fetchall()

    assert len(customer_orders) > 1, "O cliente selecionado deveria ter multiplos pedidos"

    running_orders = 0
    running_disputes = 0

    for order_id, current_failure, db_prior_orders, db_prior_disputes in customer_orders:
        assert db_prior_orders == running_orders, (
            f"Erro na contagem de pedidos anteriores do cliente no pedido {order_id}"
        )
        assert db_prior_disputes == running_disputes, (
            f"Leakage na contagem de disputas do cliente no pedido {order_id}"
        )

        running_orders += 1
        running_disputes += int(current_failure)
