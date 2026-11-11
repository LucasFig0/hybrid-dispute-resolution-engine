import os
import shutil
import duckdb
import kagglehub

RAW_DIR = os.path.join("data", "raw")
DB_PATH = os.path.join("data", "processed", "dispute_warehouse.duckdb")

def download_olist_raw_data() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    expected_file = os.path.join(RAW_DIR, "olist_orders_dataset.csv")

    if os.path.exists(expected_file):
        print("[INFO] Arquivos CSV brutos encontrados em data/raw/.")
        return

    print("[INFO] Baixando Brazilian E-Commerce Dataset (Olist) via kagglehub...")
    dataset_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".csv"):
            shutil.copy(
                os.path.join(dataset_path, file_name),
                os.path.join(RAW_DIR, file_name)
            )
    print(f"[INFO] Download concluido em {RAW_DIR}")

def build_medallion_warehouse() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)

    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
    con.execute("CREATE SCHEMA IF NOT EXISTS silver;")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")

    # =========================================================================
    # 1. CAMADA BRONZE: Ingestao bruta fiel ao CSV com metadados de linhagem
    # =========================================================================
    print("[BRONZE] Ingerindo tabelas brutas com metadados de auditoria...")
    bronze_tables = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "customers": "olist_customers_dataset.csv"
    }

    for table_name, csv_file in bronze_tables.items():
        csv_path = os.path.join(RAW_DIR, csv_file).replace("\\", "/")
        con.execute(f"""
            CREATE OR REPLACE TABLE bronze.raw_{table_name} AS
            SELECT 
                *,
                CURRENT_TIMESTAMP AS _ingested_at,
                '{csv_file}' AS _source_file
            FROM read_csv_auto('{csv_path}');
        """)

    bronze_orders_count = con.execute("SELECT COUNT(*) FROM bronze.raw_orders").fetchone()[0]

    # =========================================================================
    # 2. CAMADA SILVER: Limpeza, deduplicacao, normalizacao textual e QA Gate
    # =========================================================================
    print("[SILVER] Aplicando deduplicacao, sanitizacao de texto e regras de qualidade...")

    # 2.1 Silver Items: Resolve pedidos multi-seller elegendo o seller de maior valor (ARG_MAX)
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

    # 2.2 Silver Payments: Consolida multiplas formas de pagamento por pedido
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

    # 2.3 Silver Reviews: Deduplicacao por timestamp e limpeza de quebras de linha (\r\n)
    con.execute("""
        CREATE OR REPLACE TABLE silver.order_reviews_clean AS
        WITH dedup_reviews AS (
            SELECT 
                order_id,
                CAST(review_score AS INTEGER) AS review_score,
                COALESCE(review_comment_title, '') AS raw_title,
                COALESCE(review_comment_message, '') AS raw_message,
                ROW_NUMBER() OVER (
                    PARTITION BY order_id 
                    ORDER BY CAST(review_answer_timestamp AS TIMESTAMP) DESC,
                             CAST(review_creation_date AS TIMESTAMP) DESC
                ) AS rn
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
        FROM dedup_reviews
        WHERE rn = 1;
    """)

    # 2.4 Silver Orders: Filtra anomalias temporais (entregas antes da compra) e une entidades
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
          -- Data Quality Gate: garante que a entrega nao ocorreu antes da compra
          AND CAST(o.order_delivered_customer_date AS TIMESTAMP) >= CAST(o.order_purchase_timestamp AS TIMESTAMP)
          -- Data Quality Gate: garante valor financeiro valido
          AND (i.total_items_price + i.total_freight_value) > 0;
    """)

    silver_orders_count = con.execute("SELECT COUNT(*) FROM silver.orders_validated").fetchone()[0]

    # =========================================================================
    # 3. CAMADA GOLD: Feature Engineering Temporal Livre de Data Leakage
    # =========================================================================
    print("[GOLD] Calculando features de SLA e janelas moveis sem data leakage...")

    con.execute("""
        CREATE OR REPLACE TABLE gold.order_dispute_context AS
        WITH base_features AS (
            SELECT 
                *,
                ROUND(
                    date_diff('minute', estimated_ts, delivered_ts) / 60.0, 2
                ) AS delivery_delay_hours,
                ROUND(
                    date_diff('minute', purchase_ts, delivered_ts) / 60.0, 2
                ) AS total_delivery_duration_hours,
                ROUND(
                    total_freight_value / NULLIF(payment_value, 0), 4
                ) AS freight_to_payment_ratio,
                CASE WHEN review_score = 1 THEN 1 ELSE 0 END AS is_severe_operational_failure
            FROM silver.orders_validated
        )
        SELECT 
            *,
            COUNT(*) OVER (
                PARTITION BY seller_id 
                ORDER BY purchase_ts 
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS seller_prior_orders_count,
            ROUND(
                (COALESCE(SUM(is_severe_operational_failure) OVER (
                    PARTITION BY seller_id 
                    ORDER BY purchase_ts 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) + 1.0) / 
                (COUNT(*) OVER (
                    PARTITION BY seller_id 
                    ORDER BY purchase_ts 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) + 10.0), 
            4) AS seller_historical_defect_rate
        FROM base_features
        ORDER BY purchase_ts ASC;
    """)

    # Cria alias na raiz para manter compatibilidade direta com queries simples
    con.execute("""
        CREATE OR REPLACE VIEW gold_order_dispute_context AS 
        SELECT * FROM gold.order_dispute_context;
    """)

    gold_stats = con.execute("""
        SELECT 
            COUNT(*) AS total_gold,
            SUM(CASE WHEN LENGTH(review_comment_message) >= 5 THEN 1 ELSE 0 END) AS with_text,
            SUM(is_multi_seller_order) AS multi_seller_count,
            ROUND(AVG(is_severe_operational_failure), 4) AS failure_rate
        FROM gold.order_dispute_context
    """).fetchone()

    con.close()

    dropped_records = bronze_orders_count - silver_orders_count
    print("\n=== RELATORIO DE AUDITORIA MEDALLION (DUCKDB) ===")
    print(f"1. Camada Bronze (raw_orders)             : {bronze_orders_count} registros")
    print(f"2. Expurgados no Quality Gate (Silver)    : {dropped_records} registros (nao entregues, incompletos ou anomalias temporais)")
    print(f"3. Camada Silver (orders_validated)       : {silver_orders_count} registros limpos")
    print(f"4. Camada Gold (order_dispute_context)    : {gold_stats[0]} registros prontos para ML")
    print(f"   -> Pedidos multi-seller identificados  : {gold_stats[2]}")
    print(f"   -> Pedidos com texto higienizado (>=5c): {gold_stats[1]}")
    print(f"   -> Taxa de falha operacional (Target=1): {gold_stats[3] * 100:.2f}%")

if __name__ == "__main__":
    download_olist_raw_data()
    build_medallion_warehouse()
