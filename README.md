# MineriaS1 – Pipeline de análisis de congestión en Santiago

Este repositorio contiene un script principal `pipeline_congestion.py` que ejecuta de punta a punta un flujo reproducible de análisis de datos de congestión vial en Santiago a partir del archivo `congestion_Santiago_14-03-2025 (1) (1).csv`.

El pipeline realiza: preprocesamiento y enriquecimiento de datos, análisis exploratorio (EDA), evaluación de normalidad, PCA, entrenamiento de dos modelos lineales (con y sin selección de variables), comparación de resultados y generación de un reporte final en PDF con tablas y gráficos.

## Flujo del programa

El flujo está implementado en la función `main()` de `pipeline_congestion.py` e incluye las siguientes etapas (en orden):

- `cargar_datos()` lee el CSV de entrada.
- `limpiar_y_enriquecer()` normaliza textos, convierte numéricos, construye fechas/horas y variables derivadas como duración y peak.
- `eda()` genera estadísticas, tablas y gráficos (histogramas, boxplots, faltantes, correlaciones por pares, agrupaciones por comuna).
- `normalidad()` produce Q-Q plot y prueba de normalidad (Shapiro o KS) para la variable objetivo o la primera numérica disponible.
- `pca_analysis()` estandariza, calcula PCA, scree plot, cargas de PC1/PC2 y un biplot.
- `preparar_datos_modelado()` define variables predictoras y objetivo (por defecto, `Velocidad km/h` si está disponible).
- `dividir_datos()` separa entrenamiento y prueba (stratify en clasificación).
- `modelar_modelo1()` entrena un pipeline de preprocesamiento + modelo lineal (regresión o clasificación) y calcula métricas. Además, ajusta un modelo `statsmodels` para obtener coeficientes y p-valores.
- `modelar_modelo2()` repite el proceso pero con selección secuencial de variables (SFS) antes del modelo.
- `comparar_modelos()` guarda métricas comparativas y gráficos (dispersión, residuos o ROC/PR y matriz de confusión según el tipo de problema).
- `generar_conclusiones()` compila hallazgos, limitaciones y recomendaciones en un archivo de texto.
- `generar_reporte_pdf()` construye el reporte final PDF incorporando tablas, figuras y conclusiones.

## Requisitos

El proyecto está probado con Python 3.10 dentro de un entorno virtual. Dependencias principales:

- numpy, pandas, matplotlib, seaborn
- scikit-learn, statsmodels, scipy

## Configuración del entorno

Se recomienda usar un entorno virtual local para asegurar compatibilidad:

```powershell
# Crear venv (ya incluido en este repo el uso de .venv)
python -m venv .venv

# Activar (PowerShell)
.\.venv\Scripts\Activate.ps1

# Actualizar herramientas
python -m pip install --upgrade pip setuptools wheel

# Instalar dependencias
pip install numpy scipy scikit-learn pandas matplotlib seaborn statsmodels
```

## Ejecución

Con el entorno activado, ejecutar el script principal:

```powershell
python .\pipeline_congestion.py
```

## Entradas y salidas

- Entrada por defecto: `./congestion_Santiago_14-03-2025 (1) (1).csv` (configurable editando `INPUT_CSV` en `pipeline_congestion.py`).
- Salidas generadas:
  - Carpeta `resultados/`: imágenes (png), tablas (csv), textos (txt) y `reporte.pdf`.
  - Carpeta `modelos/`: artefactos de modelos entrenados (`modelo1.pkl`, `modelo2.pkl`).

## Problemas y avisos conocidos

- **Avisos de seaborn sobre `palette` sin `hue`**: son deprecations que no afectan el resultado actual. Se pueden ajustar las llamadas a `sns.barplot()` para silenciarlos.
- **Aviso de pandas por `fillna(..., inplace=True)` en encadenamiento**: es un `FutureWarning`. El cálculo se realiza, pero se recomienda reemplazar por asignaciones sin `inplace` para futuras versiones.
- **Matrices dispersas en statsmodels**: se convirtió explícitamente a denso antes de llamar a `statsmodels` para evitar el error “The truth value of an array is ambiguous”. Esto ya está corregido en el código.
- **Backend gráfico**: se fuerza `matplotlib.use("Agg")` para evitar dependencias de GUI (Tcl/Tk) en ejecución por consola y prevenir errores al cerrar figuras.
- **Categorías desconocidas en OneHotEncoder**: se notifica con `UserWarning`, pero está configurado `handle_unknown="ignore"` y no interrumpe la ejecución.
- **Versionado de Python**: se usa Python 3.10 en `.venv/` para maximizar la compatibilidad (especialmente con scikit-learn).

Si ocurre alguna excepción, el script intenta escribir el traceback completo en `resultados/error_trace.txt` para facilitar el diagnóstico.

## Estructura del proyecto (resumen)

```
MineriaS1/
├─ pipeline_congestion.py        # Script principal del pipeline
├─ congestion_Santiago_*.csv     # Dataset de entrada (por defecto)
├─ resultados/                   # Salidas: imágenes, tablas, reporte.pdf
├─ modelos/                      # Modelos serializados (.pkl)
├─ .venv/                        # Entorno virtual (ignorarlo en git)
└─ README.md                     # Este archivo
```

## .gitignore

Se ignoran por defecto las carpetas de entorno:

```
.env/
.venv/
```

Puedes añadir también artefactos de ejecución si no quieres versionarlos:

```
resultados/
modelos/
__pycache__/
.pytest_cache/
```

---

Si necesitas adaptar el pipeline a otra fecha/dataset o agregar nuevos modelos, abre un issue o comenta qué cambios quieres y te ayudo a implementarlos.