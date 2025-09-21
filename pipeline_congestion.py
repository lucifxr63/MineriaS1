"""Script de análisis de congestión en Santiago.

Genera preprocesamiento, EDA, PCA, modelos lineales y un reporte PDF final.
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression
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
import statsmodels.api as sm
from joblib import dump

# -----------------------------------------------------------------------------
# Configuración global y constantes
# -----------------------------------------------------------------------------

INPUT_CSV = "./congestion_Santiago_14-03-2025 (1) (1).csv"
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


def ensure_directories() -> None:
    """Garantiza la existencia de los directorios de salida."""
    for directory in (RESULTS_DIR, MODELS_DIR):
        os.makedirs(directory, exist_ok=True)
    print(f"Directorios garantizados: {RESULTS_DIR}, {MODELS_DIR}")


def safe_strip(series: pd.Series) -> pd.Series:
    """Aplica strip a series de texto manejando valores nulos."""
    return series.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})


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


# -----------------------------------------------------------------------------
# 1. Carga de datos
# -----------------------------------------------------------------------------


def cargar_datos(ruta: str = INPUT_CSV) -> pd.DataFrame:
    """Lee el archivo CSV de entrada y devuelve un DataFrame."""
    print(f"Cargando datos desde {ruta} ...")
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
        print(f"Datos cargados con forma {df.shape}")
        return df
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Error al cargar el archivo: {exc}") from exc


# -----------------------------------------------------------------------------
# 2. Limpieza y enriquecimiento
# -----------------------------------------------------------------------------


def limpiar_y_enriquecer(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y genera variables derivadas requeridas."""
    df = df.copy()
    print("Iniciando limpieza y enriquecimiento de datos ...")

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
            df[col] = safe_strip(df[col])

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
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"Fecha", "Hora Inicio", "Hora Fin"}.issubset(df.columns):
        df["dt_inicio"] = pd.to_datetime(
            df["Fecha"].astype(str).str.strip() + " " + df["Hora Inicio"].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )
        df["dt_fin"] = pd.to_datetime(
            df["Fecha"].astype(str).str.strip() + " " + df["Hora Fin"].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )
        mask = df["dt_fin"] < df["dt_inicio"]
        df.loc[mask, "dt_fin"] = df.loc[mask, "dt_fin"] + pd.Timedelta(days=1)
    else:
        df["dt_inicio"] = pd.NaT
        df["dt_fin"] = pd.NaT
        print("Advertencia: no se pudieron crear dt_inicio/dt_fin por falta de columnas.")

    duracion_horas = []
    duracion_min = []
    for _, value in df.get("n", pd.Series([None] * len(df))).items():
        horas, minutos = parse_duration_string(value)
        duracion_horas.append(horas)
        duracion_min.append(minutos)
    df["duracion_horas"] = duracion_horas
    df["duracion_min"] = duracion_min

    if "dt_inicio" in df.columns and "dt_fin" in df.columns:
        delta = df["dt_fin"] - df["dt_inicio"]
        delta_hours = delta.dt.total_seconds() / 3600.0
        need_hours = df["duracion_horas"].isna()
        df.loc[need_hours, "duracion_horas"] = delta_hours.loc[need_hours]
        df["duracion_min"].fillna(df["duracion_horas"] * 60.0, inplace=True)

    df["hora_inicio_num"] = df["dt_inicio"].dt.hour + df["dt_inicio"].dt.minute / 60.0
    df["hora_fin_num"] = df["dt_fin"].dt.hour + df["dt_fin"].dt.minute / 60.0

    df["peak"] = np.where(
        df["hora_inicio_num"].between(7, 10, inclusive="both")
        | df["hora_inicio_num"].between(17, 20, inclusive="both"),
        "Punta",
        "Valle",
    )

    if "Velocidad km/h" in df.columns:
        df["TARGET_BIN"] = np.where(df["Velocidad km/h"] < 20, "Alta", "Normal")

    print("Limpieza y enriquecimiento finalizados.")
    return df

# -----------------------------------------------------------------------------
# 3. Análisis exploratorio de datos
# -----------------------------------------------------------------------------


def eda(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Genera tablas y gráficos exploratorios."""
    print("Iniciando EDA ...")
    ensure_directories()
    sns.set(style="whitegrid")

    numeric_cols = [col for col in NUMERIC_COLUMNS_ORDER if col in df.columns]
    categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in df.columns]

    outputs: Dict[str, List[str]] = {"figures": []}

    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"}, inplace=True)
        desc = desc[["count", "mean", "std", "min", "q25", "median", "q75", "max"]]
        desc["skew"] = df[numeric_cols].skew()
        desc["kurtosis"] = df[numeric_cols].kurtosis()
        desc.reset_index(inplace=True)
        desc.rename(columns={"index": "variable"}, inplace=True)
        desc = desc.round(2)
        summary_path = os.path.join(RESULTS_DIR, "tabla_resumen_numericas.csv")
        desc.to_csv(summary_path, index=False)
        print(f"Resumen numérico guardado en {summary_path}")

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
        cat_df.to_csv(os.path.join(RESULTS_DIR, "tabla_categoricas.csv"), index=False)

    missing = df.isna().sum()
    missing_df = pd.DataFrame(
        {
            "variable": missing.index,
            "n_missing": missing.values,
            "pct_missing": (missing.values / len(df) * 100).round(2),
        }
    )
    missing_df.to_csv(os.path.join(RESULTS_DIR, "tabla_faltantes.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    missing_plot = missing_df.sort_values("pct_missing", ascending=False)
    sns.barplot(x="pct_missing", y="variable", data=missing_plot, ax=ax, palette="Blues_r")
    ax.set_title("Porcentaje de faltantes por variable")
    ax.set_xlabel("% faltante")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%")
    faltantes_path = os.path.join(RESULTS_DIR, "faltantes_bar.png")
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
        hist_path = os.path.join(RESULTS_DIR, f"hist_{col.replace(' ', '_')}.png")
        fig.tight_layout()
        fig.savefig(hist_path, dpi=PLOT_DPI)
        outputs["figures"].append(hist_path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(x=data, ax=ax, color="#ff7f0e")
        ax.set_title(f"Boxplot de {col}")
        ax.set_xlabel(col)
        box_path = os.path.join(RESULTS_DIR, f"boxplot_{col.replace(' ', '_')}.png")
        fig.tight_layout()
        fig.savefig(box_path, dpi=PLOT_DPI)
        outputs["figures"].append(box_path)
        plt.close(fig)

    for col in categorical_cols:
        counts = df[col].fillna("Desconocido").value_counts().head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
        ax.set_title(f"Frecuencias de {col}")
        ax.set_xlabel("Conteo")
        ax.set_ylabel(col)
        for container in ax.containers:
            ax.bar_label(container, fmt="%d")
        cat_path = os.path.join(RESULTS_DIR, f"categoricas_barras_{col.replace(' ', '_')}.png")
        fig.tight_layout()
        fig.savefig(cat_path, dpi=PLOT_DPI)
        outputs["figures"].append(cat_path)
        plt.close(fig)

    if numeric_cols:
        corr_matrix = df[numeric_cols].corr()
        corr_matrix.to_csv(os.path.join(RESULTS_DIR, "correlacion_matriz.csv"))
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Matriz de correlación")
        heatmap_path = os.path.join(RESULTS_DIR, "correlacion_heatmap.png")
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
        top_pairs.to_csv(os.path.join(RESULTS_DIR, "correlacion_top_pairs.csv"), index=False)
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
        group_df[["vel_prom", "vel_std", "largo_prom", "duracion_min_prom"]] = group_df[
            ["vel_prom", "vel_std", "largo_prom", "duracion_min_prom"]
        ].round(2)
        group_df.to_csv(os.path.join(RESULTS_DIR, "tabla_grupo_comuna.csv"), index=False)
    else:
        group_df = pd.DataFrame()

    outputs["numeric_cols"] = numeric_cols
    outputs["categorical_cols"] = categorical_cols
    outputs["correlation_pairs"] = top_pairs
    outputs["group_df"] = group_df
    print("EDA finalizada.")
    return outputs

# -----------------------------------------------------------------------------
# 4. Normalidad
# -----------------------------------------------------------------------------


def normalidad(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, List[str]]:
    """Evalúa normalidad para la variable continua principal."""
    print("Evaluando normalidad ...")
    outputs = {"figures": []}
    if not numeric_cols:
        print("No hay columnas numéricas para evaluar normalidad.")
        return outputs

    target_var = "Velocidad km/h" if "Velocidad km/h" in numeric_cols else numeric_cols[0]
    data = df[target_var].dropna()
    if data.empty:
        print("Datos insuficientes para evaluar normalidad.")
        return outputs

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot para {target_var}")
    qq_path = os.path.join(RESULTS_DIR, f"qqplot_{target_var.replace(' ', '_')}.png")
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
    txt_path = os.path.join(RESULTS_DIR, f"normalidad_{target_var.replace(' ', '_')}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    outputs["normalidad_text"] = txt_path
    print(f"Prueba de normalidad guardada en {txt_path}")
    return outputs

# -----------------------------------------------------------------------------
# 5. PCA
# -----------------------------------------------------------------------------


def pca_analysis(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, List[str]]:
    """Realiza PCA sobre las columnas numéricas."""
    print("Ejecutando PCA ...")
    outputs = {"figures": []}
    if not numeric_cols:
        print("No hay columnas numéricas para PCA.")
        return outputs

    numeric_df = df[numeric_cols].dropna()
    if numeric_df.empty:
        print("Datos insuficientes para PCA tras eliminar nulos.")
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
    var_df.to_csv(os.path.join(RESULTS_DIR, "tabla_pca_varianza.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(var_exp) + 1), var_exp * 100, marker="o")
    ax.set_xlabel("Componente")
    ax.set_ylabel("Varianza explicada (%)")
    ax.set_title("Scree plot")
    scree_path = os.path.join(RESULTS_DIR, "pca_varianza.png")
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
        pc1.head(10)[["variable", "loading", "abs_loading_rank"]].to_csv(
            os.path.join(RESULTS_DIR, "tabla_pca_cargas_pc1.csv"), index=False
        )
    if loadings.shape[0] >= 2:
        pc2 = pd.DataFrame({"variable": feature_names, "loading": loadings[1]})
        pc2["abs_loading"] = pc2["loading"].abs()
        pc2.sort_values("abs_loading", ascending=False, inplace=True)
        pc2["abs_loading_rank"] = range(1, len(pc2) + 1)
        pc2.head(10)[["variable", "loading", "abs_loading_rank"]].to_csv(
            os.path.join(RESULTS_DIR, "tabla_pca_cargas_pc2.csv"), index=False
        )

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
        biplot_path = os.path.join(RESULTS_DIR, "pca_biplot.png")
        fig.tight_layout()
        fig.savefig(biplot_path, dpi=PLOT_DPI)
        outputs["figures"].append(biplot_path)
        plt.close(fig)

    outputs["loadings"] = loadings
    outputs["feature_names"] = feature_names
    outputs["componentes"] = componentes
    print("PCA finalizado.")
    return outputs

# -----------------------------------------------------------------------------
# 6. Preparación de datos para modelado
# -----------------------------------------------------------------------------


def preparar_datos_modelado(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], str, str]:
    """Prepara conjuntos de datos y determina el tipo de problema."""
    if "Velocidad km/h" in df.columns:
        target_col = "Velocidad km/h"
        problem_type = "regression"
    else:
        target_col = "TARGET_BIN"
        problem_type = "classification"

    if target_col not in df.columns:
        raise ValueError("No se encontró una variable objetivo para el modelado.")

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

    return X, y, numeric_features, categorical_features, target_col, problem_type

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
) -> Dict:
    """Entrena el modelo base con todas las variables disponibles."""
    print("Entrenando Modelo 1 ...")
    results: Dict = {}

    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), [col for col in numeric_features if col in X_train.columns]))
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                [col for col in categorical_features if col in X_train.columns],
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    if problem_type == "regression":
        estimator = LinearRegression()
    else:
        estimator = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)

    X_train_processed = pipeline.named_steps["preprocessor"].transform(X_train)
    X_test_processed = pipeline.named_steps["preprocessor"].transform(X_test)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    if problem_type == "regression":
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        bias = float(np.mean(y_pred - y_test))
        metrics_model = {"RMSE": rmse, "R2": r2, "MAE": mae, "Bias": bias}

        baseline_pred = np.full(len(y_test), fill_value=y_train.mean(), dtype=float)
        metrics_baseline = {
            "RMSE": mean_squared_error(y_test, baseline_pred, squared=False),
            "R2": r2_score(y_test, baseline_pred),
            "MAE": mean_absolute_error(y_test, baseline_pred),
            "Bias": float(np.mean(baseline_pred - y_test)),
        }

        sm_model = sm.OLS(y_train, sm.add_constant(X_train_processed, has_constant="add")).fit()
        coef_df = pd.DataFrame(
            {
                "variable": ["const"] + list(feature_names),
                "coef": sm_model.params,
                "std_err": sm_model.bse,
                "p_value": sm_model.pvalues,
            }
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
        sm_model = sm.Logit(
            (y_train == positive_class).astype(int),
            sm.add_constant(X_train_processed, has_constant="add"),
        ).fit(disp=False)
        coef_df = pd.DataFrame(
            {
                "variable": ["const"] + list(feature_names),
                "coef": sm_model.params,
                "std_err": sm_model.bse,
                "p_value": sm_model.pvalues,
                "odds_ratio": np.exp(sm_model.params),
            }
        )

    coef_df.to_csv(os.path.join(RESULTS_DIR, "modelo1_coeficientes.csv"), index=False)
    dump(pipeline, os.path.join(MODELS_DIR, "modelo1.pkl"))

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
            "X_test_processed": X_test_processed,
            "X_train_processed": X_train_processed,
        }
    )
    print("Modelo 1 entrenado correctamente.")
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
) -> Dict:
    """Entrena un modelo con selección secuencial de variables."""
    print("Entrenando Modelo 2 ...")
    results: Dict = {}

    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), [col for col in numeric_features if col in X_train.columns]))
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                [col for col in categorical_features if col in X_train.columns],
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    preprocessor.fit(X_train, y_train)

    feature_names = preprocessor.get_feature_names_out()
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if problem_type == "regression":
        estimator = LinearRegression()
        scoring = "neg_root_mean_squared_error"
    else:
        estimator = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
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
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        bias = float(np.mean(y_pred - y_test))
        metrics_model = {"RMSE": rmse, "R2": r2, "MAE": mae, "Bias": bias}
        sm_model = sm.OLS(y_train, sm.add_constant(X_train_selected, has_constant="add")).fit()
        coef_df = pd.DataFrame(
            {
                "variable": ["const"] + list(selected_features),
                "coef": sm_model.params,
                "std_err": sm_model.bse,
                "p_value": sm_model.pvalues,
            }
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
            if y.nunique() == 2
            else np.nan,
        }
        sm_model = sm.Logit(
            (y_train == positive_class).astype(int),
            sm.add_constant(X_train_selected, has_constant="add"),
        ).fit(disp=False)
        coef_df = pd.DataFrame(
            {
                "variable": ["const"] + list(selected_features),
                "coef": sm_model.params,
                "std_err": sm_model.bse,
                "p_value": sm_model.pvalues,
                "odds_ratio": np.exp(sm_model.params),
            }
        )

    coef_df.to_csv(os.path.join(RESULTS_DIR, "modelo2_coeficientes.csv"), index=False)
    dump(pipeline, os.path.join(MODELS_DIR, "modelo2.pkl"))

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
            "X_test_processed": X_test_selected,
            "X_train_processed": X_train_selected,
            "support_mask": support_mask,
        }
    )
    print("Modelo 2 entrenado correctamente.")
    return results

# -----------------------------------------------------------------------------
# 7b. División de datos
# -----------------------------------------------------------------------------


def dividir_datos(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos en entrenamiento y prueba con semilla fija."""
    stratify = y if (problem_type == "classification" and y.nunique() > 1) else None
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
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
) -> Dict[str, List[str]]:
    """Genera métricas comparativas y gráficos específicos."""
    print("Comparando modelos ...")
    outputs = {"figures": []}

    metrics_df = pd.DataFrame(
        [
            {"modelo": "baseline", **baseline_metrics},
            {"modelo": "modelo1", **modelo1["metrics"]},
            {"modelo": "modelo2", **modelo2["metrics"]},
        ]
    )
    metrics_path = os.path.join(RESULTS_DIR, "metricas_modelos.csv")
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
            scatter_path = os.path.join(RESULTS_DIR, f"modelo_scatter_y_vs_yhat_m{idx}.png")
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
            resid_path = os.path.join(RESULTS_DIR, f"residuos_m{idx}.png")
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
                roc_path = os.path.join(RESULTS_DIR, f"roc_m{idx}.png")
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
                pr_path = os.path.join(RESULTS_DIR, f"pr_m{idx}.png")
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
            cm_path = os.path.join(RESULTS_DIR, f"cm_m{idx}.png")
            fig.tight_layout()
            fig.savefig(cm_path, dpi=PLOT_DPI)
            outputs["figures"].append(cm_path)
            plt.close(fig)

    print("Comparación finalizada.")
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
) -> str:
    """Crea un archivo de conclusiones basadas en los resultados."""
    print("Generando conclusiones ...")
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

    conclusions_path = os.path.join(RESULTS_DIR, "conclusiones.txt")
    with open(conclusions_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Conclusiones guardadas en {conclusions_path}")
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
) -> str:
    """Construye un reporte PDF con todos los resultados."""
    print("Generando reporte PDF ...")
    report_path = os.path.join(RESULTS_DIR, "reporte.pdf")

    figure_paths: List[str] = []
    for output in (eda_outputs, normalidad_outputs, pca_outputs, comparacion_outputs):
        figure_paths.extend(output.get("figures", []))

    table_files = [
        "tabla_resumen_numericas.csv",
        "tabla_categoricas.csv",
        "tabla_faltantes.csv",
        "correlacion_top_pairs.csv",
        "tabla_pca_varianza.csv",
        "tabla_pca_cargas_pc1.csv",
        "tabla_pca_cargas_pc2.csv",
        "tabla_grupo_comuna.csv",
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
        var_path = os.path.join(RESULTS_DIR, "tabla_pca_varianza.csv")
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
        model1_metrics = ", ".join(f"{k}: {v:.3f}" for k, v in modelo1["metrics"].items() if pd.notna(v))
        model2_metrics = ", ".join(f"{k}: {v:.3f}" for k, v in modelo2["metrics"].items() if pd.notna(v))
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
            path = os.path.join(RESULTS_DIR, table_name)
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

    print(f"Reporte PDF guardado en {report_path}")
    return report_path

# -----------------------------------------------------------------------------
# 12. Flujo principal
# -----------------------------------------------------------------------------


def main() -> None:
    """Punto de entrada principal del script."""
    try:
        ensure_directories()
        df_raw = cargar_datos(INPUT_CSV)
        df_clean = limpiar_y_enriquecer(df_raw)

        eda_outputs = eda(df_clean)
        normal_outputs = normalidad(df_clean, eda_outputs.get("numeric_cols", []))
        pca_outputs = pca_analysis(df_clean, eda_outputs.get("numeric_cols", []))

        X, y, numeric_features, categorical_features, target_col, problem_type = preparar_datos_modelado(df_clean)
        X_train, X_test, y_train, y_test = dividir_datos(X, y, problem_type)

        modelo1 = modelar_modelo1(
            X_train,
            X_test,
            y_train,
            y_test,
            numeric_features,
            categorical_features,
            problem_type,
        )
        modelo2 = modelar_modelo2(
            X_train,
            X_test,
            y_train,
            y_test,
            numeric_features,
            categorical_features,
            problem_type,
        )

        comparacion = comparar_modelos(problem_type, modelo1["baseline_metrics"], modelo1, modelo2)
        conclusiones_path = generar_conclusiones(
            df_clean,
            eda_outputs.get("correlation_pairs", pd.DataFrame()),
            eda_outputs.get("group_df", pd.DataFrame()),
            modelo1,
            modelo2,
        )
        generar_reporte_pdf(
            df_clean,
            eda_outputs,
            normal_outputs,
            pca_outputs,
            modelo1,
            modelo2,
            comparacion,
            conclusiones_path,
        )

        print("Flujo completo finalizado con éxito.")
    except Exception as exc:  # noqa: BLE001
        print(f"Error en la ejecución: {exc}")


if __name__ == "__main__":
    main()
