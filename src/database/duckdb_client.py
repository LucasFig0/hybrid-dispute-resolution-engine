"""
Modulo de acesso a dados (Repository Pattern) para consulta de baixa latencia no DuckDB.

Responsabilidades:
1. Consultar o contexto enriquecido de um pedido na camada Gold (gold.order_dispute_context).
2. Retornar objetos estritamente validados pelo contrato Pydantic OrderGoldContext.
3. Fornecer amostras reais de pedidos por perfil de risco para testes e simulacoes da API.
"""

from pathlib import Path
from typing import Optional
import duckdb
from src.schemas.dispute import OrderGoldContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "dispute_warehouse.duckdb"


class DuckDBDisputeRepository:
    """
    Repositorio somente leitura para consulta de contexto transacional e historico na camada Gold.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Banco de dados DuckDB nao encontrado em: {self.db_path}. "
                "Execute build_warehouse.py antes de iniciar o repositorio."
            )

    def get_order_context(self, order_id: str) -> Optional[OrderGoldContext]:
        """
        Busca os dados consolidados de um pedido pelo seu order_id na tabela Gold.
        Retorna None caso o pedido nao exista na base validada.
        """
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            cursor = con.execute(
                """
                SELECT 
                    order_id,
                    customer_unique_id,
                    seller_id,
                    item_count,
                    distinct_products_count,
                    distinct_sellers_count,
                    is_multi_seller_order,
                    total_items_price,
                    total_freight_value,
                    payment_value,
                    max_installments,
                    delivery_delay_hours,
                    total_delivery_duration_hours,
                    freight_to_payment_ratio,
                    seller_prior_orders_count,
                    seller_historical_defect_rate,
                    customer_prior_orders_count,
                    customer_prior_disputes_count
                FROM gold.order_dispute_context
                WHERE order_id = ?
                LIMIT 1;
                """,
                [order_id],
            )
            row = cursor.fetchone()
            if row is None:
                return None

            col_names = [desc[0] for desc in cursor.description]
            row_dict = dict(zip(col_names, row))
            return OrderGoldContext(**row_dict)
        finally:
            con.close()
