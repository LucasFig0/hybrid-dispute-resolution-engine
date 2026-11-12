"""
Modulo de construcao do Data Warehouse local em DuckDB seguindo a Arquitetura Medallion.

Responsabilidades:
1. Baixar os arquivos CSV brutos do dataset publico da Olist via kagglehub.
2. Camada Bronze: Ingerir os dados brutos preservando o formato original com colunas de auditoria.
3. Camada Silver: Deduplicar registros, higienizar textos, tratar pedidos multi-seller e aplicar Quality Gates.
4. Camada Gold: Construir features temporais de vendedor e cliente livres de Data Leakage.
"""

import logging
import shutil
from pathlib import Path
import duckdb
import kagglehub

# Configuracao de logging estruturado para rastreabilidade de execucao
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Definicao de caminhos absolutos seguros para Windows (suporta espacos e acentos no caminho)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "dispute_warehouse.duckdb"


def download_olist_raw_data() -> None:
    """
    Verifica a existencia dos CSVs brutos da Olist em data/raw/.
    Caso nao existam, realiza o download automatizado via kagglehub.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    expected_file = RAW_DIR / "olist_orders_dataset.csv"

    if expected_file.exists():
        logger.info("Arquivos CSV brutos encontrados em: %s", RAW_DIR)
        return

    logger.info("Iniciando download do Brazilian E-Commerce Dataset (Olist) via kagglehub...")
    dataset_path = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))

    # Copia apenas os arquivos CSV para o diretorio data/raw do projeto
    for csv_file in dataset_path.glob("*.csv"):
        shutil.copy(csv_file, RAW_DIR / csv_file.name)

    logger.info("Download concluido. Arquivos disponiveis em: %s", RAW_DIR)


def ingest_bronze_layer(con: duckdb.DuckDBPyConnection) -> int:
    """
    Camada Bronze: Ingestao 1:1 dos arquivos CSV brutos.
    Adiciona metadados tecnicos (_ingested_at e _source_file) para rastreabilidade de linhagem.
    Retorna o total de pedidos brutos ingeridos.
    """
    logger.info("Executando Camada Bronze: ingestao bruta com metadados de linhagem...")
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")

    bronze_tables = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "customers": "olist_customers_dataset.csv"
    }

    for table_alias, csv_filename in bronze_tables.items():
        # Converte o caminho para formato POSIX exigido pelo motor SQL do DuckDB
        csv_posix_path = (RAW_DIR / csv_filename).as_posix()
        con.execute(f"""
            CREATE OR REPLACE TABLE bronze.raw_{table_alias} AS
            SELECT 
                *,
                CURRENT_TIMESTAMP AS _ingested_at,
                '{csv_filename}' AS _source_file
            FROM read_csv_auto('{csv_posix_path}');
        """)

    raw_orders_count = con.execute("SELECT COUNT(*) FROM bronze.raw_orders").fetchone()[0]
    return int(raw_orders_count)


def transform_silver_layer(con: duckdb.DuckDBPyConnection) -> int:
    """
    Camada Silver: Limpeza, normalizacao, padronizacao de tipos e Data Quality Gate.
    Retorna o total de pedidos validados apos o filtro de qualidade.
    """
    logger.info("Executando Camada Silver: limpeza, deduplicacao e regras de qualidade...")
    con.execute("CREATE SCHEMA IF NOT EXISTS silver;")

    # 1. Agregacao de itens por pedido:
    # Em pedidos multi-seller, utiliza ARG_MAX(seller_id, price) para atribuir a responsabilidade
    # principal ao vendedor do item de maior valor financeiro.
    con.execute("""
        CREATE OR REPLACE TABLE silver.order_items_clean AS
        SELECT 
            order_id,
            ARG_MAX(seller_id, price) AS primary_seller_id,
            COUNT(DISTINCT seller_id) AS distinct_sellers_count,
            COUNT(order_item_id) AS item_count,
            COUNT(DISTINCT product_id) AS distinct_products_count,
            ROUND(SUM(price), 2) AS total_items_price,
            ROUND(SUM(freight_value), 2) AS total_freight_value
        FROM bronze.raw_items
        WHERE price > 0 AND freight_value >= 0
        GROUP BY order_id;
    """)

    # 2. Agregacao de pagamentos por pedido:
    # Consolida pedidos pagos com multiplos meios somando o valor total e extraindo o meio principal.
    con.execute("""
        CREATE OR REPLACE TABLE silver.order_payments_clean AS
        SELECT 
            order_id,
            ROUND(SUM(payment_value), 2) AS payment_value,
            MAX(COALESCE(payment_installments, 1)) AS max_installments,
            ARG_MAX(payment_type, payment_value) AS primary_payment_type
        FROM bronze.raw_payments
        WHERE payment_value > 0
        GROUP BY order_id;
    """)

    # 3. Deduplicacao e sanitizacao de avaliacoes (reviews):
    # Remove quebras de linha (\r, \n), tabulacoes (\t) e espacos multiplos dos relatos em portugues.
    con.execute(r"""
        CREATE OR REPLACE TABLE silver.order_reviews_clean AS
        WITH deduplicated_reviews AS (
            SELECT 
                order_id,
                CAST(review_score AS INTEGER) AS review_score,
                COALESCE(review_comment_title, '') AS raw_title,
                COALESCE(review_comment_message, '') AS raw_message,
                ROW_NUMBER() OVER (
                    PARTITION BY order_id 
                    ORDER BY CAST(review_answer_timestamp AS TIMESTAMP) DESC,
                             CAST(review_creation_date AS TIMESTAMP) DESC
                ) AS row_num
            FROM bronze.raw_reviews
            WHERE review_score BETWEEN 1 AND 5
        )
        SELECT 
            order_id,
            review_score,
            REGEXP_REPLACE(
                TRIM(raw_title || ' ' || raw_message),
                '[\r\n\t\s]+',
                ' ',
                'g'
            ) AS clean_review_text
        FROM deduplicated_reviews
        WHERE row_num = 1;
    """)

    # 4. Consolidacao de pedidos validados (Quality Gate):
    # Expurga pedidos nao entregues, datas nulas e anomalias onde a entrega antecede a compra.
    con.execute("""
        CREATE OR REPLACE TABLE silver.orders_validated AS
        SELECT 
            o.order_id,
            c.customer_unique_id,
            c.customer_state,
            i.primary_seller_id AS seller_id,
            i.distinct_sellers_count,
            CASE WHEN i.distinct_sellers_count > 1 THEN 1 ELSE 0 END AS is_multi_seller_order,
            CAST(o.order_purchase_timestamp AS TIMESTAMP) AS purchase_ts,
            CAST(o.order_delivered_customer_date AS TIMESTAMP) AS delivered_ts,
            CAST(o.order_estimated_delivery_date AS TIMESTAMP) AS estimated_ts,
            i.item_count,
            i.distinct_products_count,
            i.total_items_price,
            i.total_freight_value,
            COALESCE(p.payment_value, ROUND(i.total_items_price + i.total_freight_value, 2)) AS payment_value,
            COALESCE(p.max_installments, 1) AS max_installments,
            COALESCE(p.primary_payment_type, 'unknown') AS primary_payment_type,
            r.review_score,
            r.clean_review_text AS review_comment_message
        FROM bronze.raw_orders o
        INNER JOIN bronze.raw_customers c ON o.customer_id = c.customer_id
        INNER JOIN silver.order_items_clean i ON o.order_id = i.order_id
        LEFT JOIN silver.order_payments_clean p ON o.order_id = p.order_id
        INNER JOIN silver.order_reviews_clean r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_purchase_timestamp IS NOT NULL
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
          AND CAST(o.order_delivered_customer_date AS TIMESTAMP) >= CAST(o.order_purchase_timestamp AS TIMESTAMP)
          AND (i.total_items_price + i.total_freight_value) > 0;
    """)

    silver_orders_count = con.execute("SELECT COUNT(*) FROM silver.orders_validated").fetchone()[0]
    return int(silver_orders_count)


def build_gold_layer(con: duckdb.DuckDBPyConnection) -> tuple:
    """
    Camada Gold: Feature Store unificada para o Roteador Tier 1 e o Agente Tier 2.
    Aplica Window Functions estritas (UNBOUNDED PRECEDING AND 1 PRECEDING) tanto por vendedor
    quanto por cliente, garantindo zero Data Leakage temporal e eliminando Training-Serving Skew.
    """
    logger.info("Executando Camada Gold: engenharia de features de vendedor e cliente sem data leakage...")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")

    con.execute("""
        CREATE OR REPLACE TABLE gold.order_dispute_context AS
        WITH operational_metrics AS (
            SELECT 
                *,
                -- Atraso em horas em relacao a estimativa informada ao cliente (positivo = atraso)
                ROUND(
                    date_diff('minute', estimated_ts, delivered_ts) / 60.0, 2
                ) AS delivery_delay_hours,
                -- Duracao total da entrega em horas desde o momento da compra
                ROUND(
                    date_diff('minute', purchase_ts, delivered_ts) / 60.0, 2
                ) AS total_delivery_duration_hours,
                -- Peso relativo do frete sobre o valor pago pelo cliente
                ROUND(
                    total_freight_value / NULLIF(payment_value, 0), 4
                ) AS freight_to_payment_ratio,
                -- Variavel alvo (Target): 1 indica quebra severa de experiencia/disputa (nota 1)
                CASE WHEN review_score = 1 THEN 1 ELSE 0 END AS is_severe_operational_failure
            FROM silver.orders_validated
        )
        SELECT 
            *,
            -- 1. Historico Temporal do Vendedor (Zero Data Leakage)
            COUNT(*) OVER (
                PARTITION BY seller_id 
                ORDER BY purchase_ts ASC, order_id ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS seller_prior_orders_count,
            ROUND(
                (COALESCE(SUM(is_severe_operational_failure) OVER (
                    PARTITION BY seller_id 
                    ORDER BY purchase_ts ASC, order_id ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) + 1.0) / 
                (COUNT(*) OVER (
                    PARTITION BY seller_id 
                    ORDER BY purchase_ts ASC, order_id ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) + 10.0), 
            4) AS seller_historical_defect_rate,

            -- 2. Historico Temporal do Cliente (Zero Data Leakage)
            -- Contabiliza quantos pedidos anteriores o mesmo cliente ja realizou antes do pedido atual
            COUNT(*) OVER (
                PARTITION BY customer_unique_id 
                ORDER BY purchase_ts ASC, order_id ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS customer_prior_orders_count,
            -- Contabiliza quantas disputas severas (nota 1) o cliente abriu em pedidos anteriores
            COALESCE(SUM(is_severe_operational_failure) OVER (
                PARTITION BY customer_unique_id 
                ORDER BY purchase_ts ASC, order_id ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS customer_prior_disputes_count
        FROM operational_metrics
        ORDER BY purchase_ts ASC, order_id ASC;
    """)

    # Cria View no schema padrao apontando para a tabela Gold
    con.execute("""
        CREATE OR REPLACE VIEW gold_order_dispute_context AS 
        SELECT * FROM gold.order_dispute_context;
    """)

    gold_stats = con.execute("""
        SELECT 
            COUNT(*) AS total_gold,
            SUM(CASE WHEN LENGTH(review_comment_message) >= 5 THEN 1 ELSE 0 END) AS with_text,
            SUM(is_multi_seller_order) AS multi_seller_count,
            SUM(CASE WHEN customer_prior_orders_count > 0 THEN 1 ELSE 0 END) AS repeat_customer_orders,
            ROUND(AVG(is_severe_operational_failure), 4) AS failure_rate
        FROM gold.order_dispute_context
    """).fetchone()

    return gold_stats


def build_medallion_warehouse() -> None:
    """
    Orquestra a construcao completa das camadas Bronze, Silver e Gold e exibe o relatorio de auditoria.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    try:
        bronze_count = ingest_bronze_layer(con)
        silver_count = transform_silver_layer(con)
        gold_stats = build_gold_layer(con)
    finally:
        con.close()

    dropped_records = bronze_count - silver_count
    print("\n=== RELATORIO DE AUDITORIA MEDALLION (DUCKDB) ===")
    print(f"1. Camada Bronze (bronze.raw_orders)        : {bronze_count} registros")
    print(f"2. Expurgados no Quality Gate (Silver)      : {dropped_records} registros")
    print(f"3. Camada Silver (silver.orders_validated)  : {silver_count} registros validos")
    print(f"4. Camada Gold (gold.order_dispute_context) : {gold_stats[0]} registros prontos para ML")
    print(f"   -> Pedidos multi-seller identificados    : {gold_stats[2]}")
    print(f"   -> Pedidos de clientes recorrentes (>0)  : {gold_stats[3]}")
    print(f"   -> Pedidos com texto higienizado (>=5c)  : {gold_stats[1]}")
    print(f"   -> Taxa de falha operacional (Target=1)  : {gold_stats[4] * 100:.2f}%")


if __name__ == "__main__":
    download_olist_raw_data()
    build_medallion_warehouse()
