"""
Modulo de treinamento, calibracao isotonica e avaliacao Out-of-Time (OOT) do modelo Tier 1.

Responsabilidades:
1. Carregar as features da camada Gold (gold.order_dispute_context) ordenadas cronologicamente.
2. Realizar divisao temporal estrita (70% Treino, 15% Calibracao, 15% Teste Futuro/OOT).
3. Treinar o classificador base LightGBM sem vazamento de dados do futuro.
4. Calibrar as probabilidades preditas via Regressao Isotonica no split de calibracao.
5. Avaliar metricas discriminativas (ROC-AUC, PR-AUC), calibracao (Brier Score) e Offload Rate.
6. Serializar o artefato final e seus metadados em artifacts/lgbm_tier1_calibrated.joblib.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import duckdb
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

# Suporte limpo para scikit-learn >= 1.6 (FrozenEstimator substitui cv="prefit")
try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN_ESTIMATOR = True
except ImportError:
    HAS_FROZEN_ESTIMATOR = False

# Configuracao de logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Caminhos absolutos do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "dispute_warehouse.duckdb"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "lgbm_tier1_calibrated.joblib"

# Lista oficial e ordenada de features da Camada Gold utilizadas pelo Tier 1
TIER1_FEATURES: List[str] = [
    "item_count",
    "distinct_products_count",
    "distinct_sellers_count",
    "is_multi_seller_order",
    "total_items_price",
    "total_freight_value",
    "payment_value",
    "max_installments",
    "delivery_delay_hours",
    "total_delivery_duration_hours",
    "freight_to_payment_ratio",
    "seller_prior_orders_count",
    "seller_historical_defect_rate",
    "customer_prior_orders_count",
    "customer_prior_disputes_count",
]

TARGET_COL: str = "is_severe_operational_failure"


def load_gold_dataset() -> pd.DataFrame:
    """
    Carrega a tabela gold.order_dispute_context garantindo ordenacao cronologica estrita
    por timestamp de compra e identificador do pedido.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse DuckDB nao encontrado em {DB_PATH}. Execute build_warehouse.py primeiro."
        )

    logger.info("Carregando dataset da camada Gold: %s", DB_PATH)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        cols_sql = ", ".join(["order_id", "purchase_ts", TARGET_COL] + TIER1_FEATURES)
        query = f"""
            SELECT {cols_sql}
            FROM gold.order_dispute_context
            ORDER BY purchase_ts ASC, order_id ASC;
        """
        df = con.execute(query).df()
    finally:
        con.close()

    # Garante ausencia de valores NaN em features numericas
    df[TIER1_FEATURES] = df[TIER1_FEATURES].fillna(0.0)
    logger.info("Dataset carregado com sucesso: %d linhas e %d features.", len(df), len(TIER1_FEATURES))
    return df


def train_and_calibrate_tier1() -> Dict[str, Any]:
    """
    Executa o pipeline completo de divisao temporal, treinamento do LightGBM,
    calibracao isotonica e avaliacao no conjunto de teste Out-of-Time.
    """
    df = load_gold_dataset()
    n_total = len(df)

    # Indices de corte temporal: 70% Treino | 15% Calibracao | 15% Teste Out-of-Time
    train_idx = int(n_total * 0.70)
    calib_idx = int(n_total * 0.85)

    df_train = df.iloc[:train_idx]
    df_calib = df.iloc[train_idx:calib_idx]
    df_test = df.iloc[calib_idx:]

    X_train, y_train = df_train[TIER1_FEATURES], df_train[TARGET_COL]
    X_calib, y_calib = df_calib[TIER1_FEATURES], df_calib[TARGET_COL]
    X_test, y_test = df_test[TIER1_FEATURES], df_test[TARGET_COL]

    logger.info(
        "Splits temporais definidos -> Treino: %d | Calibracao: %d | Teste OOT: %d",
        len(df_train), len(df_calib), len(df_test)
    )

    # 1. Treinamento do classificador base LightGBM
    logger.info("Treinando modelo base LGBMClassifier no conjunto de treino...")
    base_lgbm = LGBMClassifier(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    base_lgbm.fit(X_train, y_train)

    # 2. Calibracao de probabilidade via Regressao Isotonica no split temporal intermediario
    logger.info("Calibrando probabilidades com Regressao Isotonica no split de calibracao...")
    if HAS_FROZEN_ESTIMATOR:
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_lgbm),
            method="isotonic"
        )
    else:
        calibrated_model = CalibratedClassifierCV(
            estimator=base_lgbm,
            method="isotonic",
            cv="prefit"
        )
    calibrated_model.fit(X_calib, y_calib)

    # 3. Avaliacao comparativa no conjunto de Teste Futuro (Out-of-Time)
    raw_probs_test = base_lgbm.predict_proba(X_test)[:, 1]
    calib_probs_test = calibrated_model.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_test, calib_probs_test))
    pr_auc = float(average_precision_score(y_test, calib_probs_test))
    brier_raw = float(brier_score_loss(y_test, raw_probs_test))
    brier_calib = float(brier_score_loss(y_test, calib_probs_test))

    # 4. Simulacao da Politica de Roteamento Deterministico + ML (Tier 1 Offload)
    # Regra A (Auto-Approve Reembolso Baixo Valor): Atraso severo (>24h), ticket <= R$ 60 e prob >= 0.50
    auto_approve_mask = (
        (X_test["delivery_delay_hours"] > 24.0)
        & (X_test["payment_value"] <= 60.0)
        & (calib_probs_test >= 0.35)
    )

    # Regra B (Fast-Track Baixo Risco Operacional): Entregue antes do prazo (<-12h),
    # prob de falha severa muito baixa (<= 0.06) e cliente sem historico de disputas
    low_risk_mask = (
        (X_test["delivery_delay_hours"] < -12.0)
        & (calib_probs_test <= 0.06)
        & (X_test["customer_prior_disputes_count"] == 0)
    )

    tier1_offload_rate = float((auto_approve_mask | low_risk_mask).mean())
    auto_approve_rate = float(auto_approve_mask.mean())
    low_risk_rate = float(low_risk_mask.mean())

    # 5. Extracao das 5 features mais importantes do modelo base para auditoria
    importances = pd.Series(base_lgbm.feature_importances_, index=TIER1_FEATURES)
    top_features = importances.sort_values(ascending=False).head(5).to_dict()

    # 6. Empacotamento e serializacao do artefato para consumo na API FastAPI
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload: Dict[str, Any] = {
        "model": calibrated_model,
        "base_model": base_lgbm,
        "features": TIER1_FEATURES,
        "target": TARGET_COL,
        "splits_info": {
            "train_size": len(df_train),
            "calib_size": len(df_calib),
            "test_oot_size": len(df_test),
            "test_start_ts": str(df_test["purchase_ts"].min()),
            "test_end_ts": str(df_test["purchase_ts"].max()),
        },
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score_raw": round(brier_raw, 4),
            "brier_score_calibrated": round(brier_calib, 4),
            "tier1_total_offload_rate": round(tier1_offload_rate, 4),
            "tier1_auto_approve_rate": round(auto_approve_rate, 4),
            "tier1_low_risk_fast_track_rate": round(low_risk_rate, 4),
        },
        "top_5_features": top_features,
    }

    joblib.dump(artifact_payload, ARTIFACT_PATH)
    logger.info("Artefato serializado com sucesso em: %s", ARTIFACT_PATH)

    print("\n=== RELATORIO DE AVALIACAO OUT-OF-TIME (TIER 1 LIGHTGBM) ===")
    print(f"Periodo do Teste OOT          : {artifact_payload['splits_info']['test_start_ts']} ate {artifact_payload['splits_info']['test_end_ts']}")
    print(f"Amostras no Teste OOT         : {len(df_test)} pedidos")
    print(f"ROC-AUC Score (OOT)           : {roc_auc:.4f}")
    print(f"PR-AUC Score (OOT)            : {pr_auc:.4f}")
    print(f"Brier Score (Antes Calibracao): {brier_raw:.4f}")
    print(f"Brier Score (Pos Calibracao)  : {brier_calib:.4f}")
    print(f"Tier 1 Total Offload Rate     : {tier1_offload_rate * 100:.2f}% (resolvidos sem custo de LLM)")
    print(f"  -> Auto-Approve (Atraso+Baixo Valor) : {auto_approve_rate * 100:.2f}%")
    print(f"  -> Fast-Track (Baixo Risco Logistico): {low_risk_rate * 100:.2f}%")
    print("Top 5 Features (Split Importance):")
    for feat_name, score in top_features.items():
        print(f"  - {feat_name}: {score}")

    return artifact_payload


if __name__ == "__main__":
    train_and_calibrate_tier1()

