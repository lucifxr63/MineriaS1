"""Script de análisis de congestión en Santiago.

Genera preprocesamiento, EDA, PCA, modelos lineales y un reporte PDF final.
"""

import argparse
import logging
import os
import re
import textwrap
import time
import traceback
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from joblib import dump
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# Configuración global y constantes
# -----------------------------------------------------------------------------

INPUT_CSV = "./congestion_Santiago_14-03-2025 (1) (1).csv"
TARGET = None
TEST_SIZE = 0.2
DEFAULT_TASK = "auto"
RESULTS_DIR = "./resultados"
MODELS_DIR = "./modelos"
PLOT_DPI = 300
RANDOM_STATE = 42

NUMERIC_COLUMNS_ORDER = [
    "X",
    "Y",
    "Z",
    "Ranking Regional",
    "Largo km",
    "Velocidad km/h",
    "duracion_horas",
    "duracion_min",
    "hora_inicio_num",
    "hora_fin_num",
]

CATEGORICAL_COLUMNS = [
    "Calle",
    "Comuna",
    "ID",
    "peak",
]

EXCLUDE_FEATURES = [
    "Fecha",
    "Hora Inicio",
    "Hora Fin",
    "n",
    "dt_inicio",
    "dt_fin",
    "TARGET_BIN",
]

# -----------------------------------------------------------------------------
# Utilidades generales
# -----------------------------------------------------------------------------


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("pipeline_congestion")


@dataclass
class PipelineConfig:
    """Configuración general del pipeline."""

    input_csv: str = INPUT_CSV
    target: Optional[str] = TARGET
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    task: str = DEFAULT_TASK
    results_dir: str = RESULTS_DIR
    models_dir: str = MODELS_DIR
    rare_threshold: float = 0.01
    min_rare_count: int = 10
    top_categories: int = 20


@dataclass
class StageResult:
    """Almacena duración y salida de cada etapa."""

    duration: float
    result: object = None


def configure_warnings() -> None:
    """Configura el filtrado de warnings conocidos y seguros."""

    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories",
        category=UserWarning,
        module="sklearn",
    )


def parse_arguments() -> PipelineConfig:
    """Parsea argumentos de línea de comandos y construye la configuración."""

    parser = argparse.ArgumentParser(description="Pipeline E2E de congestión vehicular")
    parser.add_argument("--input", default=INPUT_CSV, help="Ruta al CSV de entrada")
    parser.add_argument("--target", default=TARGET, help="Columna objetivo a utilizar")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Proporción de datos para prueba")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Semilla aleatoria")
    parser.add_argument("--task", choices=["auto", "reg", "clf"], default=DEFAULT_TASK, help="Tipo de tarea a forzar")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directorio de resultados")
    parser.add_argument("--models-dir", default=MODELS_DIR, help="Directorio para modelos")
    parser.add_argument(
        "--rare-threshold",
        type=float,
        default=0.01,
        help="Frecuencia mínima para conservar categorías antes del one-hot",
    )
    parser.add_argument(
        "--min-rare-count",
        type=int,
        default=10,
        help="Conteo mínimo absoluto para conservar categorías",
    )
    parser.add_argument(
        "--top-categories",
        type=int,
        default=20,
        help="Número máximo de categorías mostradas en gráficos",
    )
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size debe estar entre 0 y 1.")

    return PipelineConfig(
        input_csv=args.input,
        target=args.target if args.target else None,
        test_size=args.test_size,
        random_state=args.seed,
        task=args.task,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        rare_threshold=args.rare_threshold,
        min_rare_count=args.min_rare_count,
        top_categories=args.top_categories,
    )


def run_stage(
    name: str,
    func,
    stage_registry: Dict[str, StageResult],
    *args,
    **kwargs,
):
    """Ejecuta una etapa con logging y medición de tiempo."""

    LOGGER.info("[STEP] %s", name)
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        LOGGER.info("[DONE] %s en %.2fs", name, duration)
        stage_registry[name] = StageResult(duration=duration, result=result)
        return result
    except Exception:
        duration = time.perf_counter() - start
        stage_registry[name] = StageResult(duration=duration, result=None)
        LOGGER.exception("[FAIL] %s", name)
        raise


def ensure_directories(directories: Optional[List[str]] = None) -> None:
    """Garantiza la existencia de los directorios de salida."""

    dirs = directories or [RESULTS_DIR, MODELS_DIR]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    LOGGER.info("Directorios garantizados: %s", ", ".join(dirs))


def safe_strip(series: pd.Series) -> pd.Series:
    """Aplica strip a series de texto manejando valores nulos."""
    return series.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})


def sanitize_filename(name: str) -> str:
    """Convierte cualquier texto en un nombre de archivo seguro.

    - Reemplaza espacios por guiones bajos.
    - Elimina caracteres no permitidos en nombres de archivo de Windows (barra invertida, barra, dos puntos, asterisco,
      signo de interrogación, comillas dobles, menor que, mayor que y barra vertical)
    """
    if name is None:
        return ""
    s = str(name)
    s = s.replace(" ", "_")
    # Quitar caracteres inválidos para nombres de archivos en Windows
    s = re.sub(r"[\\/:*?\"<>|]", "", s)
    return s


def parse_duration_string(value: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Convierte expresiones de duración a horas y minutos."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None

    text = str(value).lower().strip()
    if not text or text in {"nan", ""}:
        return None, None

    hours = None
    minutes = None

    hour_pattern = re.search(r"(\d+(?:[\.,]\d+)?)\s*(?:h|hr|hrs|hora|horas)", text)
    minute_pattern = re.search(r"(\d+(?:[\.,]\d+)?)\s*(?:m|min|mins|minutos)", text)

    if hour_pattern:
        hours = float(hour_pattern.group(1).replace(",", "."))
    if minute_pattern:
        minutes = float(minute_pattern.group(1).replace(",", "."))

    if hours is None and minutes is None:
        numeric_pattern = re.search(r"\d+(?:[\.,]\d+)?", text)
        if numeric_pattern:
            value_float = float(numeric_pattern.group(0).replace(",", "."))
            if "min" in text:
                minutes = value_float
            else:
                hours = value_float

    if hours is None and minutes is not None:
        hours = minutes / 60.0
    if minutes is None and hours is not None:
        minutes = hours * 60.0

    return hours, minutes


class ArrayColumnSelector(BaseEstimator, TransformerMixin):
    """Transformador que selecciona columnas mediante una máscara booleana."""

    def __init__(self, support_mask: np.ndarray):
        self.support_mask = support_mask

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.support_mask is None:
            return X
        return X[:, self.support_mask]


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Agrupa categorías con baja frecuencia en una etiqueta común."""

    def __init__(
        self,
        min_frequency: float = 0.01,
        min_count: int = 10,
        rare_label: str = "__RARE__",
    ):
        self.min_frequency = min_frequency
        self.min_count = min_count
        self.rare_label = rare_label
        self.category_maps_: Dict[str, set] = {}
        self.feature_names_in_: List[str] = []

    def fit(self, X, y=None):
        X_df = self._ensure_dataframe(X)
        n_samples = len(X_df)
        self.feature_names_in_ = X_df.columns.tolist()
        self.category_maps_ = {}
        threshold = max(int(np.ceil(self.min_frequency * n_samples)), self.min_count)
        for col in self.feature_names_in_:
            series = X_df[col].astype(object)
            counts = series.value_counts(dropna=True)
            keep_categories = counts[counts >= threshold].index.tolist()
            self.category_maps_[col] = set(keep_categories)
        return self

    def transform(self, X):
        X_df = self._ensure_dataframe(X)
        for col in self.feature_names_in_:
            if col not in X_df.columns:
                continue
            allowed = self.category_maps_.get(col, set())
            X_df.loc[:, col] = X_df[col].astype(object)
            mask = ~X_df[col].isin(allowed) & X_df[col].notna()
            if mask.any():
                X_df.loc[mask, col] = self.rare_label
        return X_df

    def _ensure_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if hasattr(X, "toarray"):
            X = X.toarray()
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))
        return pd.DataFrame(X)


# -----------------------------------------------------------------------------
# 1. Carga de datos
# -----------------------------------------------------------------------------


def cargar_datos(ruta: str = INPUT_CSV) -> pd.DataFrame:
    """Lee el archivo CSV de entrada y devuelve un DataFrame."""

    LOGGER.info("Cargando datos desde %s ...", ruta)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"El archivo {ruta} no existe.")

    try:
        df = pd.read_csv(
            ruta,
            encoding="utf-8",
            sep=",",
            decimal=".",
            skipinitialspace=True,
        )
        LOGGER.info("Datos cargados con forma %s", df.shape)
        return df
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Error al cargar el archivo: {exc}") from exc


# -----------------------------------------------------------------------------
# 2. Limpieza y enriquecimiento
# -----------------------------------------------------------------------------


def limpiar_y_enriquecer(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y genera variables derivadas requeridas."""
    df = df.copy()
    LOGGER.info("Iniciando limpieza y enriquecimiento de datos ...")

    text_columns = [
        "Calle",
        "Comuna",
        "Fecha",
        "Hora Fin",
        "Hora Inicio",
        "ID",
        "n",
    ]
    for col in text_columns:
        if col in df.columns:
            df.loc[:, col] = safe_strip(df[col])

    numeric_conversion = [
        "X",
        "Y",
        "Z",
        "Ranking Regional",
        "Largo km",
        "Velocidad km/h",
    ]
    for col in numeric_conversion:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    if {"Fecha", "Hora Inicio", "Hora Fin"}.issubset(df.columns):
        df.loc[:, "dt_inicio"] = pd.to_datetime(
            df["Fecha"].astype(str).str.strip() + " " + df["Hora Inicio"].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )
        df.loc[:, "dt_fin"] = pd.to_datetime(
            df["Fecha"].astype(str).str.strip() + " " + df["Hora Fin"].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )
        mask = df["dt_fin"] < df["dt_inicio"]
        df.loc[mask, "dt_fin"] = df.loc[mask, "dt_fin"] + pd.Timedelta(days=1)
    else:
        df.loc[:, "dt_inicio"] = pd.NaT
        df.loc[:, "dt_fin"] = pd.NaT
        LOGGER.warning("No se pudieron crear dt_inicio/dt_fin por falta de columnas.")

    duracion_horas = []
    duracion_min = []
    for _, value in df.get("n", pd.Series([None] * len(df))).items():
        horas, minutos = parse_duration_string(value)
        duracion_horas.append(horas)
        duracion_min.append(minutos)
    df.loc[:, "duracion_horas"] = duracion_horas
    df.loc[:, "duracion_min"] = duracion_min

    if "dt_inicio" in df.columns and "dt_fin" in df.columns:
        delta = df["dt_fin"] - df["dt_inicio"]
        delta_hours = delta.dt.total_seconds() / 3600.0
        need_hours = df["duracion_horas"].isna()
        df.loc[need_hours, "duracion_horas"] = delta_hours.loc[need_hours]
        df.loc[:, "duracion_min"] = df["duracion_min"].fillna(df["duracion_horas"] * 60.0)

    df.loc[:, "hora_inicio_num"] = df["dt_inicio"].dt.hour + df["dt_inicio"].dt.minute / 60.0
    df.loc[:, "hora_fin_num"] = df["dt_fin"].dt.hour + df["dt_fin"].dt.minute / 60.0

    df.loc[:, "peak"] = np.where(
        df["hora_inicio_num"].between(7, 10, inclusive="both")
        | df["hora_inicio_num"].between(17, 20, inclusive="both"),
        "Punta",
        "Valle",
    )

    if "Velocidad km/h" in df.columns:
        df.loc[:, "TARGET_BIN"] = np.where(df["Velocidad km/h"] < 20, "Alta", "Normal")

    LOGGER.info("Limpieza y enriquecimiento finalizados.")
    return df

# -----------------------------------------------------------------------------
# 3. Análisis exploratorio de datos
# -----------------------------------------------------------------------------


def eda(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, List[str]]:
    """Genera tablas y gráficos exploratorios."""

    LOGGER.info("Iniciando EDA ...")
    ensure_directories([config.results_dir])
    sns.set_theme(style="whitegrid")

    numeric_cols = [col for col in NUMERIC_COLUMNS_ORDER if col in df.columns]
    categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in df.columns]

    outputs: Dict[str, List[str]] = {"figures": [], "tables": []}

    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"}, inplace=True)
        desc = desc[["count", "mean", "std", "min", "q25", "median", "q75", "max"]]
        desc["skew"] = df[numeric_cols].skew()
        desc["kurtosis"] = df[numeric_cols].kurtosis()
        desc.reset_index(inplace=True)
        desc.rename(columns={"index": "variable"}, inplace=True)
        desc = desc.round(2)
        summary_path = os.path.join(config.results_dir, "tabla_resumen_numericas.csv")
        desc.to_csv(summary_path, index=False)
        outputs["tables"].append(summary_path)

    cat_records = []
    for col in categorical_cols:
        series = df[col].dropna()
        n_distinct = series.nunique()
        value_counts = series.value_counts()
        if not value_counts.empty:
            top = value_counts.index[0]
            freq_top = value_counts.iloc[0]
        else:
            top = np.nan
            freq_top = np.nan
        cat_records.append({"variable": col, "n_distinct": n_distinct, "top": top, "freq_top": freq_top})
    if cat_records:
        cat_df = pd.DataFrame(cat_records)
        cat_path = os.path.join(config.results_dir, "tabla_categoricas.csv")
        cat_df.to_csv(cat_path, index=False)
        outputs["tables"].append(cat_path)

    missing = df.isna().sum()
    missing_df = pd.DataFrame(
        {
            "variable": missing.index,
            "n_missing": missing.values,
            "pct_missing": (missing.values / len(df) * 100).round(2),
        }
    )
    missing_path = os.path.join(config.results_dir, "tabla_faltantes.csv")
    missing_df.to_csv(missing_path, index=False)
    outputs["tables"].append(missing_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    missing_plot = missing_df.sort_values("pct_missing", ascending=False)
    missing_color = sns.color_palette("Blues_r", 6)[2]
    sns.barplot(x="pct_missing", y="variable", data=missing_plot, ax=ax, color=missing_color)
    ax.set_title("Porcentaje de faltantes por variable")
    ax.set_xlabel("% faltante")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%")
    faltantes_path = os.path.join(config.results_dir, "faltantes_bar.png")
    fig.tight_layout()
    fig.savefig(faltantes_path, dpi=PLOT_DPI)
    outputs["figures"].append(faltantes_path)
    plt.close(fig)

    for col in numeric_cols:
        data = df[col].dropna()
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data, kde=True, ax=ax, color="#1f77b4")
        ax.set_title(f"Histograma de {col}")
        ax.set_xlabel(col)
        hist_path = os.path.join(config.results_dir, f"hist_{sanitize_filename(col)}.png")
        fig.tight_layout()
        fig.savefig(hist_path, dpi=PLOT_DPI)
        outputs["figures"].append(hist_path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(x=data, ax=ax, color="#ff7f0e")
        ax.set_title(f"Boxplot de {col}")
        ax.set_xlabel(col)
        box_path = os.path.join(config.results_dir, f"boxplot_{sanitize_filename(col)}.png")
        fig.tight_layout()
        fig.savefig(box_path, dpi=PLOT_DPI)
        outputs["figures"].append(box_path)
        plt.close(fig)

    for col in categorical_cols:
        counts = df[col].fillna("Desconocido").value_counts().head(config.top_categories)
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        cat_color = sns.color_palette("viridis", 6)[2]
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color=cat_color)
        ax.set_title(f"Frecuencias de {col}")
        ax.set_xlabel("Conteo")
        ax.set_ylabel(col)
        for container in ax.containers:
            ax.bar_label(container, fmt="%d")
        cat_path = os.path.join(config.results_dir, f"categoricas_barras_{sanitize_filename(col)}.png")
        fig.tight_layout()
        fig.savefig(cat_path, dpi=PLOT_DPI)
        outputs["figures"].append(cat_path)
        plt.close(fig)

    if numeric_cols:
        corr_matrix = df[numeric_cols].corr()
        corr_path = os.path.join(config.results_dir, "correlacion_matriz.csv")
        corr_matrix.to_csv(corr_path)
        outputs["tables"].append(corr_path)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Matriz de correlación")
        heatmap_path = os.path.join(config.results_dir, "correlacion_heatmap.png")
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=PLOT_DPI)
        outputs["figures"].append(heatmap_path)
        plt.close(fig)

        records = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_value = corr_matrix.iloc[i, j]
                if pd.notna(corr_value) and abs(corr_value) >= 0.5:
                    records.append({"var1": cols[i], "var2": cols[j], "corr": corr_value, "abs_corr": abs(corr_value)})
        if records:
            top_pairs = pd.DataFrame(records).sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])
        else:
            top_pairs = pd.DataFrame(columns=["var1", "var2", "corr"])
        top_pairs_path = os.path.join(config.results_dir, "correlacion_top_pairs.csv")
        top_pairs.to_csv(top_pairs_path, index=False)
        outputs["tables"].append(top_pairs_path)
    else:
        top_pairs = pd.DataFrame()

    if "Comuna" in df.columns:
        group_df = (
            df.groupby("Comuna")
            .agg(
                n_tramos=("ID", "count"),
                vel_prom=("Velocidad km/h", "mean"),
                vel_std=("Velocidad km/h", "std"),
                largo_prom=("Largo km", "mean"),
                duracion_min_prom=("duracion_min", "mean"),
            )
            .reset_index()
        )
        round_cols = ["vel_prom", "vel_std", "largo_prom", "duracion_min_prom"]
        group_df.loc[:, round_cols] = group_df[round_cols].round(2)
        group_path = os.path.join(config.results_dir, "tabla_grupo_comuna.csv")
        group_df.to_csv(group_path, index=False)
        outputs["tables"].append(group_path)
    else:
        group_df = pd.DataFrame()

    outputs["numeric_cols"] = numeric_cols
    outputs["categorical_cols"] = categorical_cols
    outputs["correlation_pairs"] = top_pairs
    outputs["group_df"] = group_df
    LOGGER.info("EDA finalizada.")
    return outputs

# -----------------------------------------------------------------------------
# 4. Normalidad
# -----------------------------------------------------------------------------


def normalidad(df: pd.DataFrame, numeric_cols: List[str], config: PipelineConfig) -> Dict[str, List[str]]:
    """Evalúa normalidad para la variable continua principal."""

    LOGGER.info("Evaluando normalidad ...")
    outputs: Dict[str, List[str]] = {"figures": []}
    if not numeric_cols:
        LOGGER.warning("No hay columnas numéricas para evaluar normalidad.")
        return outputs

    target_var = "Velocidad km/h" if "Velocidad km/h" in numeric_cols else numeric_cols[0]
    data = df[target_var].dropna()
    if data.empty:
        LOGGER.warning("Datos insuficientes para evaluar normalidad en %s.", target_var)
        return outputs

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot para {target_var}")
    qq_path = os.path.join(config.results_dir, f"qqplot_{sanitize_filename(target_var)}.png")
    fig.tight_layout()
    fig.savefig(qq_path, dpi=PLOT_DPI)
    outputs["figures"].append(qq_path)
    plt.close(fig)

    if len(data) <= 5000:
        test_name = "Shapiro-Wilk"
        stat_value, p_value = stats.shapiro(data)
    else:
        test_name = "Kolmogorov-Smirnov"
        standardized = (data - data.mean()) / data.std(ddof=0)
        stat_value, p_value = stats.kstest(standardized, "norm")

    summary_text = (
        f"Prueba de normalidad: {test_name}\n"
        f"Variable: {target_var}\n"
        f"Estadístico: {stat_value:.4f}\n"
        f"p-valor: {p_value:.4f}\n"
        f"n: {len(data)}\n"
        f"Conclusión: {'No se rechaza' if p_value > 0.05 else 'Se rechaza'} la normalidad al 5%."
    )
    txt_path = os.path.join(config.results_dir, f"normalidad_{sanitize_filename(target_var)}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    outputs["normalidad_text"] = txt_path
    outputs["test_name"] = test_name
    outputs["p_value"] = p_value
    outputs["target_var"] = target_var
    LOGGER.info("Prueba de normalidad guardada en %s", txt_path)
    return outputs

# -----------------------------------------------------------------------------
# 5. PCA
# -----------------------------------------------------------------------------


def pca_analysis(df: pd.DataFrame, numeric_cols: List[str], config: PipelineConfig) -> Dict[str, List[str]]:
    """Realiza PCA sobre las columnas numéricas."""

    LOGGER.info("Ejecutando PCA ...")
    outputs: Dict[str, List[str]] = {"figures": [], "tables": []}
    if not numeric_cols:
        LOGGER.warning("No hay columnas numéricas para PCA.")
        return outputs

    numeric_df = df[numeric_cols].dropna()
    if numeric_df.empty:
        LOGGER.warning("Datos insuficientes para PCA tras eliminar nulos.")
        return outputs

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    pca_full = PCA()
    pca_full.fit(X_scaled)
    var_exp = pca_full.explained_variance_ratio_
    var_exp_acum = var_exp.cumsum()
    n_components = int(np.argmax(var_exp_acum >= 0.95) + 1) if (var_exp_acum >= 0.95).any() else len(var_exp)

    var_df = pd.DataFrame(
        {
            "componente": [f"PC{i+1}" for i in range(len(var_exp))],
            "var_exp": (var_exp * 100).round(2),
            "var_exp_acum": (var_exp_acum * 100).round(2),
        }
    )
    var_path = os.path.join(config.results_dir, "tabla_pca_varianza.csv")
    var_df.to_csv(var_path, index=False)
    outputs["tables"].append(var_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(var_exp) + 1), var_exp * 100, marker="o")
    ax.set_xlabel("Componente")
    ax.set_ylabel("Varianza explicada (%)")
    ax.set_title("Scree plot")
    scree_path = os.path.join(config.results_dir, "pca_varianza.png")
    fig.tight_layout()
    fig.savefig(scree_path, dpi=PLOT_DPI)
    outputs["figures"].append(scree_path)
    plt.close(fig)

    pca = PCA(n_components=n_components)
    componentes = pca.fit_transform(X_scaled)
    loadings = pca.components_
    feature_names = numeric_cols

    if loadings.shape[0] >= 1:
        pc1 = pd.DataFrame({"variable": feature_names, "loading": loadings[0]})
        pc1["abs_loading"] = pc1["loading"].abs()
        pc1.sort_values("abs_loading", ascending=False, inplace=True)
        pc1["abs_loading_rank"] = range(1, len(pc1) + 1)
        pc1_path = os.path.join(config.results_dir, "tabla_pca_cargas_pc1.csv")
        pc1.head(10)[["variable", "loading", "abs_loading_rank"]].to_csv(pc1_path, index=False)
        outputs["tables"].append(pc1_path)
    if loadings.shape[0] >= 2:
        pc2 = pd.DataFrame({"variable": feature_names, "loading": loadings[1]})
        pc2["abs_loading"] = pc2["loading"].abs()
        pc2.sort_values("abs_loading", ascending=False, inplace=True)
        pc2["abs_loading_rank"] = range(1, len(pc2) + 1)
        pc2_path = os.path.join(config.results_dir, "tabla_pca_cargas_pc2.csv")
        pc2.head(10)[["variable", "loading", "abs_loading_rank"]].to_csv(pc2_path, index=False)
        outputs["tables"].append(pc2_path)

    if loadings.shape[0] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.3)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Biplot PCA")
        scale_x = componentes[:, 0].std()
        scale_y = componentes[:, 1].std()
        for i, feature in enumerate(feature_names):
            ax.arrow(0, 0, loadings[0, i] * scale_x, loadings[1, i] * scale_y, color="r", alpha=0.5)
        abs_loadings = np.abs(loadings[0]) + np.abs(loadings[1])
        top_idx = np.argsort(abs_loadings)[-10:]
        for i in top_idx:
            ax.text(
                loadings[0, i] * scale_x * 1.1,
                loadings[1, i] * scale_y * 1.1,
                feature_names[i],
                color="darkred",
                ha="center",
                va="center",
            )
        biplot_path = os.path.join(config.results_dir, "pca_biplot.png")
        fig.tight_layout()
        fig.savefig(biplot_path, dpi=PLOT_DPI)
        outputs["figures"].append(biplot_path)
        plt.close(fig)

    outputs["loadings"] = loadings
    outputs["feature_names"] = feature_names
    outputs["componentes"] = componentes
    outputs["n_components"] = n_components
    outputs["varianza_acumulada_95"] = var_exp_acum[min(n_components - 1, len(var_exp_acum) - 1)] * 100
    LOGGER.info("PCA finalizado.")
    return outputs

# -----------------------------------------------------------------------------
# 6. Preparación de datos para modelado
# -----------------------------------------------------------------------------


def preparar_datos_modelado(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], str, str]:
    """Prepara conjuntos de datos y determina el tipo de problema."""

    if config.target and config.target in df.columns:
        target_col = config.target
    elif "Velocidad km/h" in df.columns:
        target_col = "Velocidad km/h"
    elif "TARGET_BIN" in df.columns:
        target_col = "TARGET_BIN"
    else:
        raise ValueError("No se encontró una variable objetivo para el modelado.")

    if config.task == "reg":
        problem_type = "regression"
    elif config.task == "clf":
        problem_type = "classification"
    else:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            problem_type = "regression"
        else:
            problem_type = "classification"

    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo {target_col} no existe en los datos.")

    if problem_type == "regression" and not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"La variable objetivo {target_col} no es numérica para regresión.")
    if problem_type == "classification" and df[target_col].nunique() < 2:
        raise ValueError(f"La variable objetivo {target_col} no tiene al menos dos clases.")

    numeric_features = [col for col in NUMERIC_COLUMNS_ORDER if col in df.columns and col != target_col]
    categorical_features = [col for col in CATEGORICAL_COLUMNS if col in df.columns]

    feature_cols = [col for col in numeric_features + categorical_features if col not in EXCLUDE_FEATURES]

    model_df = df.dropna(subset=[target_col]).copy()
    X = model_df[feature_cols]
    y = model_df[target_col]

    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[feature_cols]
    y = combined[target_col]

    if X.empty:
        raise ValueError("Los datos para modelar están vacíos tras limpiar nulos.")

    LOGGER.info(
        "Preparados datos de modelado con %d registros y %d variables (%s)",
        len(X),
        X.shape[1],
        problem_type,
    )

    return X, y, numeric_features, categorical_features, target_col, problem_type


# -----------------------------------------------------------------------------
# Utilidades para modelado
# -----------------------------------------------------------------------------


def create_one_hot_encoder() -> OneHotEncoder:
    """Crea un OneHotEncoder compatible con distintas versiones de sklearn."""

    params = {"handle_unknown": "ignore", "drop": "first"}
    try:
        return OneHotEncoder(sparse_output=False, **params)
    except TypeError:
        return OneHotEncoder(sparse=False, **params)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    config: PipelineConfig,
) -> ColumnTransformer:
    """Construye el preprocesador estándar para el pipeline."""

    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([("scaler", StandardScaler())]), numeric_features))
    if categorical_features:
        cat_pipeline = Pipeline(
            [
                (
                    "rare",
                    RareCategoryGrouper(
                        min_frequency=config.rare_threshold,
                        min_count=config.min_rare_count,
                    ),
                ),
                ("ohe", create_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))
    return ColumnTransformer(transformers, remainder="drop")


def to_dense(matrix) -> np.ndarray:
    """Convierte matrices dispersas en densas para statsmodels."""

    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def filter_constant_columns(
    X: np.ndarray, feature_names: np.ndarray
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Elimina columnas de varianza casi nula y devuelve nombres eliminados."""

    variances = np.var(X, axis=0)
    mask = variances > 1e-8
    filtered = X[:, mask]
    kept_features = feature_names[mask].tolist()
    dropped_features = feature_names[~mask].tolist()
    return filtered, kept_features, dropped_features


def build_ridge_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dropped: List[str],
) -> pd.DataFrame:
    """Obtiene coeficientes desde Ridge en ausencia de OLS válido."""

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    coef = np.concatenate(([model.intercept_], model.coef_))
    coef_df = pd.DataFrame(
        {
            "variable": ["const"] + feature_names,
            "coef": coef,
            "std_err": [np.nan] * (len(feature_names) + 1),
            "p_value": [np.nan] * (len(feature_names) + 1),
            "note": ["ridge_fallback"] * (len(feature_names) + 1),
        }
    )
    if dropped:
        dropped_df = pd.DataFrame(
            {
                "variable": dropped,
                "coef": np.nan,
                "std_err": np.nan,
                "p_value": np.nan,
                "note": "columna_eliminada",
            }
        )
        coef_df = pd.concat([coef_df, dropped_df], ignore_index=True)
    return coef_df


def run_ols_with_checks(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    fallback_label: str,
) -> Tuple[pd.DataFrame, str]:
    """Ajusta OLS con chequeos de rango y aplica Ridge como fallback."""

    X_dense = to_dense(X)
    feature_array = np.asarray(feature_names)
    X_filtered, kept_features, dropped = filter_constant_columns(X_dense, feature_array)

    X_sm = sm.add_constant(X_filtered, has_constant="add")
    n, p = X_sm.shape
    rank = np.linalg.matrix_rank(X_sm)
    if n <= p or rank < p:
        LOGGER.warning(
            "OLS inválido (%s): n=%d, p=%d, rank=%d. Se utiliza Ridge como fallback.",
            fallback_label,
            n,
            p,
            rank,
        )
        return build_ridge_coefficients(X_dense, y, feature_names, dropped), "ridge"

    try:
        model = sm.OLS(y, X_sm).fit()
        coef_df = pd.DataFrame(
            {
                "variable": ["const"] + kept_features,
                "coef": model.params,
                "std_err": model.bse,
                "p_value": model.pvalues,
            }
        )
        if dropped:
            dropped_df = pd.DataFrame(
                {
                    "variable": dropped,
                    "coef": np.nan,
                    "std_err": np.nan,
                    "p_value": np.nan,
                }
            )
            coef_df = pd.concat([coef_df, dropped_df], ignore_index=True)
        return coef_df, "ols"
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Fallo al ajustar OLS (%s): %s. Se utiliza Ridge como fallback.", fallback_label, exc)
        return build_ridge_coefficients(X_dense, y, feature_names, dropped), "ridge"


# -----------------------------------------------------------------------------
# 7. Modelado - Modelo 1
# -----------------------------------------------------------------------------


def modelar_modelo1(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    problem_type: str,
    config: PipelineConfig,
) -> Dict:
    """Entrena el modelo base con todas las variables disponibles."""

    LOGGER.info("Entrenando Modelo 1 ...")
    results: Dict = {}

    preprocessor = build_preprocessor(numeric_features, categorical_features, config)

    if problem_type == "regression":
        estimator = LinearRegression()
    else:
        estimator = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=config.random_state)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)

    X_train_processed = pipeline.named_steps["preprocessor"].transform(X_train)
    X_test_processed = pipeline.named_steps["preprocessor"].transform(X_test)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    if problem_type == "regression":
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        bias = float(np.mean(y_pred - y_test))
        metrics_model = {"RMSE": rmse, "R2": r2, "MAE": mae, "Bias": bias}

        baseline_pred = np.full(len(y_test), fill_value=y_train.mean(), dtype=float)
        mse_base = mean_squared_error(y_test, baseline_pred)
        metrics_baseline = {
            "RMSE": float(np.sqrt(mse_base)),
            "R2": r2_score(y_test, baseline_pred),
            "MAE": mean_absolute_error(y_test, baseline_pred),
            "Bias": float(np.mean(baseline_pred - y_test)),
        }

        coef_df, coef_source = run_ols_with_checks(
            X_train_processed,
            y_train.values,
            list(feature_names),
            "modelo1_regresion",
        )
        y_proba = None
    else:
        estimator_model = pipeline.named_steps["model"]
        classes = estimator_model.classes_
        positive_class = "Alta" if "Alta" in classes else classes[-1]

        if hasattr(estimator_model, "predict_proba"):
            proba = estimator_model.predict_proba(X_test_processed)
            pos_index = list(classes).index(positive_class)
            y_proba = proba[:, pos_index]
        else:
            y_proba = np.zeros(len(y_test))

        y_pred = pipeline.predict(X_test)
        metrics_model = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "F1": f1_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "ROC_AUC": roc_auc_score((y_test == positive_class).astype(int), y_proba)
            if len(classes) == 2 and y_proba is not None
            else np.nan,
        }
        baseline_label = y_train.value_counts().idxmax()
        baseline_pred = np.full(len(y_test), fill_value=baseline_label)
        metrics_baseline = {
            "Accuracy": accuracy_score(y_test, baseline_pred),
            "Precision": precision_score(y_test, baseline_pred, pos_label=positive_class, zero_division=0),
            "Recall": recall_score(y_test, baseline_pred, pos_label=positive_class, zero_division=0),
            "F1": f1_score(y_test, baseline_pred, pos_label=positive_class, zero_division=0),
            "ROC_AUC": np.nan,
        }

        X_dense = to_dense(X_train_processed)
        X_filtered, kept_features, dropped = filter_constant_columns(X_dense, np.asarray(feature_names))
        X_sm = sm.add_constant(X_filtered, has_constant="add")
        y_binary = (y_train == positive_class).astype(int)
        try:
            sm_model = sm.Logit(y_binary, X_sm).fit(disp=False)
            coef_df = pd.DataFrame(
                {
                    "variable": ["const"] + kept_features,
                    "coef": sm_model.params,
                    "std_err": sm_model.bse,
                    "p_value": sm_model.pvalues,
                    "odds_ratio": np.exp(sm_model.params),
                }
            )
            if dropped:
                dropped_df = pd.DataFrame(
                    {
                        "variable": dropped,
                        "coef": np.nan,
                        "std_err": np.nan,
                        "p_value": np.nan,
                        "odds_ratio": np.nan,
                    }
                )
                coef_df = pd.concat([coef_df, dropped_df], ignore_index=True)
            coef_source = "logit"
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Fallo al ajustar modelo Logit (modelo1): %s. Se guardan coeficientes del modelo sklearn.",
                exc,
            )
            if hasattr(estimator_model, "coef_"):
                coef = estimator_model.coef_.reshape(-1)
                intercept = float(estimator_model.intercept_.reshape(-1)[0])
            else:
                coef = np.zeros(len(feature_names))
                intercept = 0.0
            coef_df = pd.DataFrame(
                {
                    "variable": ["const"] + list(feature_names),
                    "coef": [intercept] + coef.tolist(),
                    "std_err": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": [np.nan] * (len(feature_names) + 1),
                    "note": "sklearn_fallback",
                }
            )
            coef_source = "sklearn"

    coef_path = os.path.join(config.results_dir, "modelo1_coeficientes.csv")
    coef_df.to_csv(coef_path, index=False)
    dump(pipeline, os.path.join(config.models_dir, "modelo1.pkl"))

    results.update(
        {
            "pipeline": pipeline,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": metrics_model,
            "baseline_metrics": metrics_baseline,
            "feature_names": feature_names,
            "coef_df": coef_df,
            "coef_source": coef_source,
            "X_test_processed": to_dense(X_test_processed),
            "X_train_processed": to_dense(X_train_processed),
        }
    )
    LOGGER.info("Modelo 1 entrenado correctamente.")
    return results

# -----------------------------------------------------------------------------
# 8. Modelado - Modelo 2 (selección secuencial)
# -----------------------------------------------------------------------------


def modelar_modelo2(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    problem_type: str,
    config: PipelineConfig,
) -> Dict:
    """Entrena un modelo con selección secuencial de variables."""
    LOGGER.info("Entrenando Modelo 2 ...")
    results: Dict = {}

    preprocessor = build_preprocessor(numeric_features, categorical_features, config)
    preprocessor.fit(X_train, y_train)

    feature_names = preprocessor.get_feature_names_out()
    X_train_processed = to_dense(preprocessor.transform(X_train))
    X_test_processed = to_dense(preprocessor.transform(X_test))

    if problem_type == "regression":
        estimator = LinearRegression()
        scoring = "neg_root_mean_squared_error"
    else:
        estimator = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=config.random_state)
        scoring = "roc_auc" if pd.concat([y_train, y_test]).nunique() == 2 else "f1_weighted"

    n_features_select = min(10, X_train_processed.shape[1])
    if n_features_select == 0:
        raise ValueError("No hay características disponibles tras el preprocesamiento.")

    sfs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_select,
        direction="forward",
        scoring=scoring,
        cv=5,
        n_jobs=-1,
    )
    sfs.fit(X_train_processed, y_train)
    support_mask = sfs.get_support()
    selected_features = feature_names[support_mask]

    selector = ArrayColumnSelector(support_mask)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)

    X_train_selected = X_train_processed[:, support_mask]
    X_test_selected = X_test_processed[:, support_mask]

    if problem_type == "regression":
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        bias = float(np.mean(y_pred - y_test))
        metrics_model = {"RMSE": rmse, "R2": r2, "MAE": mae, "Bias": bias}
        coef_df, coef_source = run_ols_with_checks(
            X_train_selected,
            y_train.values,
            list(selected_features),
            "modelo2_regresion",
        )
        y_proba = None
    else:
        estimator_model = pipeline.named_steps["model"]
        classes = estimator_model.classes_
        positive_class = "Alta" if "Alta" in classes else classes[-1]
        proba = estimator_model.predict_proba(X_test_selected)
        pos_index = list(classes).index(positive_class)
        y_proba = proba[:, pos_index]
        y_pred = pipeline.predict(X_test)
        metrics_model = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "F1": f1_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "ROC_AUC": roc_auc_score((y_test == positive_class).astype(int), y_proba)
            if y_test.nunique() == 2
            else np.nan,
        }
        X_dense = X_train_selected
        X_filtered, kept_features, dropped = filter_constant_columns(X_dense, np.asarray(selected_features))
        X_sm = sm.add_constant(X_filtered, has_constant="add")
        y_binary = (y_train == positive_class).astype(int)
        try:
            sm_model = sm.Logit(y_binary, X_sm).fit(disp=False)
            coef_df = pd.DataFrame(
                {
                    "variable": ["const"] + kept_features,
                    "coef": sm_model.params,
                    "std_err": sm_model.bse,
                    "p_value": sm_model.pvalues,
                    "odds_ratio": np.exp(sm_model.params),
                }
            )
            if dropped:
                dropped_df = pd.DataFrame(
                    {
                        "variable": dropped,
                        "coef": np.nan,
                        "std_err": np.nan,
                        "p_value": np.nan,
                        "odds_ratio": np.nan,
                    }
                )
                coef_df = pd.concat([coef_df, dropped_df], ignore_index=True)
            coef_source = "logit"
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Fallo al ajustar modelo Logit (modelo2): %s. Se guardan coeficientes del modelo sklearn.",
                exc,
            )
            coef = estimator_model.coef_.reshape(-1)
            intercept = float(estimator_model.intercept_.reshape(-1)[0]) if hasattr(estimator_model, "intercept_") else 0.0
            coef_df = pd.DataFrame(
                {
                    "variable": ["const"] + list(selected_features),
                    "coef": [intercept] + coef.tolist(),
                    "std_err": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": [np.nan] * (len(selected_features) + 1),
                    "note": "sklearn_fallback",
                }
            )
            coef_source = "sklearn"

    coef_path = os.path.join(config.results_dir, "modelo2_coeficientes.csv")
    coef_df.to_csv(coef_path, index=False)
    dump(pipeline, os.path.join(config.models_dir, "modelo2.pkl"))

    results.update(
        {
            "pipeline": pipeline,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": metrics_model,
            "feature_names": selected_features,
            "coef_df": coef_df,
            "coef_source": coef_source,
            "X_test_processed": X_test_selected,
            "X_train_processed": X_train_selected,
            "support_mask": support_mask,
        }
    )
    LOGGER.info("Modelo 2 entrenado correctamente.")
    return results

# -----------------------------------------------------------------------------
# 7b. División de datos
# -----------------------------------------------------------------------------


def dividir_datos(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos en entrenamiento y prueba con semilla fija."""

    stratify = y if (problem_type == "classification" and y.nunique() > 1) else None
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

# -----------------------------------------------------------------------------
# 9. Comparación de modelos y gráficos diagnósticos
# -----------------------------------------------------------------------------


def comparar_modelos(
    problem_type: str,
    baseline_metrics: Dict[str, float],
    modelo1: Dict,
    modelo2: Dict,
    config: PipelineConfig,
) -> Dict[str, List[str]]:
    """Genera métricas comparativas y gráficos específicos."""
    LOGGER.info("Comparando modelos ...")
    outputs = {"figures": []}

    metrics_df = pd.DataFrame(
        [
            {"modelo": "baseline", **baseline_metrics},
            {"modelo": "modelo1", **modelo1["metrics"]},
            {"modelo": "modelo2", **modelo2["metrics"]},
        ]
    )
    metrics_path = os.path.join(config.results_dir, "metricas_modelos.csv")
    metrics_df.to_csv(metrics_path, index=False)

    if problem_type == "regression":
        y_test = modelo1["y_test"]
        for idx, mod in enumerate([modelo1, modelo2], start=1):
            y_pred = mod["y_pred"]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_test, y_pred, alpha=0.7)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
            ax.set_xlabel("y real")
            ax.set_ylabel("y predicho")
            ax.set_title(f"Modelo {idx}: y vs y_hat")
            scatter_path = os.path.join(config.results_dir, f"modelo_scatter_y_vs_yhat_m{idx}.png")
            fig.tight_layout()
            fig.savefig(scatter_path, dpi=PLOT_DPI)
            outputs["figures"].append(scatter_path)
            plt.close(fig)

            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=y_pred, y=residuals, ax=ax)
            ax.axhline(0, color="red", linestyle="--")
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Residuo")
            ax.set_title(f"Modelo {idx}: residuos")
            resid_path = os.path.join(config.results_dir, f"residuos_m{idx}.png")
            fig.tight_layout()
            fig.savefig(resid_path, dpi=PLOT_DPI)
            outputs["figures"].append(resid_path)
            plt.close(fig)
    else:
        y_test = modelo1["y_test"]
        classes = sorted(pd.concat([modelo1["y_train"], y_test]).unique())
        positive_class = "Alta" if "Alta" in classes else classes[-1]
        for idx, mod in enumerate([modelo1, modelo2], start=1):
            y_pred = mod["y_pred"]
            y_proba = mod.get("y_proba")
            if y_proba is not None and len(classes) == 2:
                y_true_binary = (y_test == positive_class).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, label=f"Modelo {idx}")
                ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_title(f"ROC Modelo {idx}")
                ax.legend()
                roc_path = os.path.join(config.results_dir, f"roc_m{idx}.png")
                fig.tight_layout()
                fig.savefig(roc_path, dpi=PLOT_DPI)
                outputs["figures"].append(roc_path)
                plt.close(fig)

                precision, recall, _ = precision_recall_curve(y_true_binary, y_proba)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(recall, precision, label=f"Modelo {idx}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(f"Precision-Recall Modelo {idx}")
                ax.legend()
                pr_path = os.path.join(config.results_dir, f"pr_m{idx}.png")
                fig.tight_layout()
                fig.savefig(pr_path, dpi=PLOT_DPI)
                outputs["figures"].append(pr_path)
                plt.close(fig)

            cm = confusion_matrix(y_test, y_pred, labels=classes)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")
            ax.set_title(f"Matriz de confusión Modelo {idx}")
            cm_path = os.path.join(config.results_dir, f"cm_m{idx}.png")
            fig.tight_layout()
            fig.savefig(cm_path, dpi=PLOT_DPI)
            outputs["figures"].append(cm_path)
            plt.close(fig)

    LOGGER.info("Comparación finalizada.")
    return outputs

# -----------------------------------------------------------------------------
# 10. Generación de conclusiones automáticas
# -----------------------------------------------------------------------------


def generar_conclusiones(
    df: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    group_df: pd.DataFrame,
    modelo1: Dict,
    modelo2: Dict,
    config: PipelineConfig,
) -> str:
    """Crea un archivo de conclusiones basadas en los resultados."""
    LOGGER.info("Generando conclusiones ...")
    lines: List[str] = []

    # Hallazgos principales
    lines.append("Hallazgos:")
    coef1 = modelo1["coef_df"].copy()
    if "abs_coef" not in coef1.columns:
        coef1["abs_coef"] = coef1["coef"].abs()
    top_coef1 = coef1[coef1["variable"] != "const"].sort_values("abs_coef", ascending=False).head(5)
    for _, row in top_coef1.iterrows():
        lines.append(
            f"- Modelo 1 resalta a {row['variable']} con coeficiente {row['coef']:.3f} (p={row['p_value']:.3f})."
        )

    coef2 = modelo2["coef_df"].copy()
    if "abs_coef" not in coef2.columns and "coef" in coef2.columns:
        coef2["abs_coef"] = coef2["coef"].abs()
    top_coef2 = coef2[coef2["variable"] != "const"].sort_values("abs_coef", ascending=False).head(5)
    for _, row in top_coef2.iterrows():
        lines.append(
            f"- Modelo 2 identifica a {row['variable']} con coeficiente {row['coef']:.3f} (p={row['p_value']:.3f})."
        )

    if not correlation_pairs.empty:
        top_corrs = correlation_pairs.sort_values("corr", key=lambda x: x.abs(), ascending=False).head(5)
        for _, row in top_corrs.iterrows():
            lines.append(
                f"- Correlación destacada entre {row['var1']} y {row['var2']}: {row['corr']:.2f}."
            )

    if "peak" in df.columns and "Velocidad km/h" in df.columns:
        peak_stats = df.groupby("peak")["Velocidad km/h"].mean().round(2)
        for peak_label, value in peak_stats.items():
            lines.append(f"- Velocidad promedio en periodo {peak_label}: {value} km/h.")

    if not group_df.empty:
        top_comuna = group_df.sort_values("vel_prom", ascending=True).head(1)
        for _, row in top_comuna.iterrows():
            lines.append(
                f"- La comuna con menor velocidad promedio es {row['Comuna']} ({row['vel_prom']} km/h)."
            )

    # Limitaciones
    lines.append("\nLimitaciones:")
    missing_counts = df.isna().sum()
    cols_missing = missing_counts[missing_counts > 0]
    if not cols_missing.empty:
        lines.append(
            "- Existen valores faltantes en "
            + ", ".join(f"{col} ({int(val)})" for col, val in cols_missing.items())
            + "."
        )
    lines.append("- Los modelos lineales pueden no capturar relaciones no lineales o interacciones complejas.")
    lines.append("- No se incorporaron variables externas (clima, eventos), lo que puede limitar la generalización.")

    # Recomendaciones
    lines.append("\nRecomendaciones:")
    lines.append("- Incorporar nuevas variables temporales (día de la semana, festivos) y condiciones climáticas.")
    lines.append("- Explorar modelos no lineales (árboles, gradient boosting) y validación temporal/espacial.")
    lines.append("- Realizar seguimiento continuo y recalibración con datos más recientes para capturar cambios de movilidad.")

    conclusions_path = os.path.join(config.results_dir, "conclusiones.txt")
    with open(conclusions_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    LOGGER.info("Conclusiones guardadas en %s", conclusions_path)
    return conclusions_path

# -----------------------------------------------------------------------------
# 11. Generación de reporte en PDF
# -----------------------------------------------------------------------------


def generar_reporte_pdf(
    df: pd.DataFrame,
    eda_outputs: Dict,
    normalidad_outputs: Dict,
    pca_outputs: Dict,
    modelo1: Dict,
    modelo2: Dict,
    comparacion_outputs: Dict,
    conclusiones_path: str,
    config: PipelineConfig,
) -> str:
    """Construye un reporte PDF con todos los resultados."""
    LOGGER.info("Generando reporte PDF ...")
    report_path = os.path.join(config.results_dir, "reporte.pdf")

    figure_paths: List[str] = []
    for output in (eda_outputs, normalidad_outputs, pca_outputs, comparacion_outputs):
        figure_paths.extend(output.get("figures", []))

    table_files = [
        "tabla_resumen_numericas.csv",
        "tabla_categoricas.csv",
        "tabla_faltantes.csv",
        "tabla_grupo_comuna.csv",
        "correlacion_top_pairs.csv",
        "tabla_pca_varianza.csv",
        "tabla_pca_cargas_pc1.csv",
        "tabla_pca_cargas_pc2.csv",
        "metricas_modelos.csv",
    ]

    def add_text_page(pdf, title: str, paragraphs: List[str]) -> None:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=16, fontweight="bold")
        y = 0.9
        for paragraph in paragraphs:
            if not paragraph:
                continue
            wrapped = textwrap.fill(paragraph, width=90)
            ax.text(0.05, y, wrapped, ha="left", va="top", fontsize=11)
            y -= 0.07 * (wrapped.count("\n") + 1)
        pdf.savefig(fig)
        plt.close(fig)

    def _format_metrics(metrics: Dict[str, object]) -> str:
        parts: List[str] = []
        for k, v in metrics.items():
            try:
                if v is None:
                    continue
                # Intentar convertir a float; descartar arreglos/listas/Series
                if isinstance(v, (list, np.ndarray, pd.Series)):
                    continue
                vf = float(v)
                if np.isnan(vf):
                    continue
                parts.append(f"{k}: {vf:.3f}")
            except Exception:
                continue
        return ", ".join(parts)

    with PdfPages(report_path) as pdf:
        # Portada
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.5, 0.8, "Reporte de Congestión Vial", ha="center", fontsize=24, fontweight="bold")
        ax.text(0.5, 0.7, "Autores: Equipo de Analítica", ha="center", fontsize=14)
        ax.text(0.5, 0.65, "Curso: Minería de Datos", ha="center", fontsize=14)
        ax.text(0.5, 0.6, f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", ha="center", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        # Introducción y Metodología
        intro_text = [
            "Este reporte resume el flujo de análisis aplicado sobre el dataset de congestión vehicular de Santiago del 14/03/2025.",
            "Se desarrollaron etapas de limpieza, exploración, reducción de dimensionalidad y modelado predictivo con enfoque reproducible en Python.",
        ]
        metodo_text = [
            "La metodología incluyó depuración de datos temporales, creación de variables de duración y períodos peak, evaluación de normalidad, PCA y modelado lineal." ,
            f"El conjunto analizado contiene {len(df)} registros y {df.shape[1]} variables tras el enriquecimiento.",
        ]
        add_text_page(pdf, "Introducción", intro_text)
        add_text_page(pdf, "Metodología", metodo_text)

        # EDA
        eda_text = [
            "Se generaron estadísticas descriptivas para variables numéricas y categóricas, gráficos de distribución, correlaciones y análisis por comuna.",
            "Los resultados completos se presentan en las tablas y figuras asociadas a esta sección.",
        ]
        add_text_page(pdf, "Análisis Exploratorio (EDA)", eda_text)

        # Normalidad
        normal_text = []
        norm_file = normalidad_outputs.get("normalidad_text")
        if norm_file and os.path.exists(norm_file):
            with open(norm_file, "r", encoding="utf-8") as f:
                normal_text.append(f.read())
        add_text_page(pdf, "Evaluación de Normalidad", normal_text or ["No se pudo evaluar la normalidad por falta de datos."])

        # PCA
        pca_summary = []
        var_path = os.path.join(config.results_dir, "tabla_pca_varianza.csv")
        if os.path.exists(var_path):
            var_df = pd.read_csv(var_path)
            if not var_df.empty:
                componentes_95 = var_df[var_df["var_exp_acum"] >= 95]
                if not componentes_95.empty:
                    comp_num = componentes_95.iloc[0]["componente"]
                    pca_summary.append(
                        f"Se requieren {componentes_95.index[0] + 1} componentes (hasta {comp_num}) para explicar al menos el 95% de la varianza."
                    )
        pca_summary.append("Se evaluaron las cargas de las dos primeras componentes para interpretar patrones espaciales y temporales.")
        add_text_page(pdf, "Análisis PCA", pca_summary)

        # Modelos
        model1_metrics = _format_metrics(modelo1["metrics"]) if "metrics" in modelo1 else ""
        model2_metrics = _format_metrics(modelo2["metrics"]) if "metrics" in modelo2 else ""
        add_text_page(
            pdf,
            "Modelo 1",
            [
                "Modelo lineal con todas las variables disponibles (preprocesamiento estándar y codificación one-hot).",
                f"Desempeño: {model1_metrics}.",
            ],
        )
        add_text_page(
            pdf,
            "Modelo 2",
            [
                "Modelo lineal con selección secuencial de variables para mejorar interpretabilidad.",
                f"Desempeño: {model2_metrics}.",
            ],
        )

        # Comparación
        comp_text = [
            "Se compararon métricas clave frente a un baseline simple y se generaron gráficos diagnósticos para ambos modelos.",
        ]
        add_text_page(pdf, "Comparación de Modelos", comp_text)

        # Tablas
        for table_name in table_files:
            path = os.path.join(config.results_dir, table_name)
            if not os.path.exists(path):
                continue
            table_df = pd.read_csv(path)
            fig, ax = plt.subplots(figsize=(11, max(4, 1 + 0.4 * len(table_df))))
            ax.axis("off")
            ax.set_title(table_name.replace("_", " ").replace(".csv", ""), fontsize=14, fontweight="bold")
            table = ax.table(
                cellText=table_df.values,
                colLabels=table_df.columns,
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            pdf.savefig(fig)
            plt.close(fig)

        # Figuras
        for fig_path in figure_paths:
            if not os.path.exists(fig_path):
                continue
            image = plt.imread(fig_path)
            fig, ax = plt.subplots(figsize=(8.27, 6))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(os.path.basename(fig_path))
            pdf.savefig(fig)
            plt.close(fig)

        # Conclusiones y Referencias
        conclusion_text = []
        if conclusiones_path and os.path.exists(conclusiones_path):
            with open(conclusiones_path, "r", encoding="utf-8") as f:
                conclusion_text = f.read().splitlines()
        add_text_page(pdf, "Conclusiones", conclusion_text or ["No se generaron conclusiones automáticas."])

        referencias = [
            "Dataset: Observatorio de Transporte, congestión en Santiago (14/03/2025).",
            "Herramientas: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels.",
        ]
        add_text_page(pdf, "Referencias", referencias)

    LOGGER.info("Reporte PDF guardado en %s", report_path)
    return report_path

# -----------------------------------------------------------------------------
# 12. Flujo principal
# -----------------------------------------------------------------------------


def main() -> None:
    """Punto de entrada principal del script."""
    configure_warnings()
    try:
        config = parse_arguments()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error al parsear argumentos: %s", exc)
        raise

    stage_results: Dict[str, StageResult] = {}

    try:
        ensure_directories([config.results_dir, config.models_dir])
        np.random.seed(config.random_state)

        df_raw = run_stage("cargar_datos", cargar_datos, stage_results, config.input_csv)
        df_clean = run_stage("limpiar_y_enriquecer", limpiar_y_enriquecer, stage_results, df_raw)

        eda_outputs = run_stage("eda", eda, stage_results, df_clean, config)
        normal_outputs = run_stage(
            "normalidad",
            normalidad,
            stage_results,
            df_clean,
            eda_outputs.get("numeric_cols", []),
            config,
        )
        pca_outputs = run_stage(
            "pca_analysis",
            pca_analysis,
            stage_results,
            df_clean,
            eda_outputs.get("numeric_cols", []),
            config,
        )

        X, y, numeric_features, categorical_features, target_col, problem_type = run_stage(
            "preparar_datos_modelado",
            preparar_datos_modelado,
            stage_results,
            df_clean,
            config,
        )
        X_train, X_test, y_train, y_test = run_stage(
            "dividir_datos",
            dividir_datos,
            stage_results,
            X,
            y,
            problem_type,
            config,
        )

        modelo1 = run_stage(
            "modelar_modelo1",
            modelar_modelo1,
            stage_results,
            X_train,
            X_test,
            y_train,
            y_test,
            numeric_features,
            categorical_features,
            problem_type,
            config,
        )
        modelo2 = run_stage(
            "modelar_modelo2",
            modelar_modelo2,
            stage_results,
            X_train,
            X_test,
            y_train,
            y_test,
            numeric_features,
            categorical_features,
            problem_type,
            config,
        )
        comparacion = run_stage(
            "comparar_modelos",
            comparar_modelos,
            stage_results,
            problem_type,
            modelo1["baseline_metrics"],
            modelo1,
            modelo2,
            config,
        )
        conclusiones_path = run_stage(
            "generar_conclusiones",
            generar_conclusiones,
            stage_results,
            df_clean,
            eda_outputs.get("correlation_pairs", pd.DataFrame()),
            eda_outputs.get("group_df", pd.DataFrame()),
            modelo1,
            modelo2,
            config,
        )
        report_path = run_stage(
            "generar_reporte_pdf",
            generar_reporte_pdf,
            stage_results,
            df_clean,
            eda_outputs,
            normal_outputs,
            pca_outputs,
            modelo1,
            modelo2,
            comparacion,
            conclusiones_path,
            config,
        )

        LOGGER.info("Flujo completo finalizado con éxito.")

        eda_result = stage_results.get("eda")
        if eda_result and eda_result.result:
            figs = len(eda_result.result.get("figures", []))
            tables = len(eda_result.result.get("tables", []))
            LOGGER.info("[OK] EDA (%d figuras, %d tablas) en %.2fs", figs, tables, eda_result.duration)

        normal_result = stage_results.get("normalidad")
        if normal_result and normal_result.result:
            target = normal_result.result.get("target_var", "")
            test_name = normal_result.result.get("test_name", "")
            p_value = normal_result.result.get("p_value")
            if p_value is not None:
                LOGGER.info("[OK] Normalidad: %s (%s) p=%.3f", target, test_name, p_value)

        pca_result = stage_results.get("pca_analysis")
        if pca_result and pca_result.result:
            comps = pca_result.result.get("n_components")
            var95 = pca_result.result.get("varianza_acumulada_95")
            if comps is not None and var95 is not None:
                LOGGER.info("[OK] PCA: %d comps → %.1f%% var. explicada", comps, var95)

        if problem_type == "regression":
            m1 = modelo1["metrics"]
            m2 = modelo2["metrics"]
            LOGGER.info("[OK] Modelo1: RMSE=%.3f, R2=%.3f", m1.get("RMSE", float("nan")), m1.get("R2", float("nan")))
            LOGGER.info("[OK] Modelo2 (SFS): RMSE=%.3f, R2=%.3f", m2.get("RMSE", float("nan")), m2.get("R2", float("nan")))
        else:
            m1 = modelo1["metrics"]
            m2 = modelo2["metrics"]
            LOGGER.info(
                "[OK] Modelo1: Accuracy=%.3f, F1=%.3f",
                m1.get("Accuracy", float("nan")),
                m1.get("F1", float("nan")),
            )
            LOGGER.info(
                "[OK] Modelo2 (SFS): Accuracy=%.3f, F1=%.3f",
                m2.get("Accuracy", float("nan")),
                m2.get("F1", float("nan")),
            )

        comp_stage = stage_results.get("comparar_modelos")
        if comp_stage and comp_stage.result:
            LOGGER.info(
                "[OK] Comparación: %d figuras en %.2fs",
                len(comp_stage.result.get("figures", [])),
                comp_stage.duration,
            )

        LOGGER.info("[OK] Reporte PDF: %s", report_path)

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error en la ejecución: %s", exc)
        try:
            ensure_directories([config.results_dir])
            error_path = os.path.join(config.results_dir, "error_trace.txt")
            with open(error_path, "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
            LOGGER.error("Traceback completo en: %s", error_path)
        except Exception:  # noqa: BLE001
            LOGGER.exception("No se pudo escribir el archivo de errores.")
        raise


if __name__ == "__main__":
    main()
