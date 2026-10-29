"""Miniproyecto 1: la serie, de las líneas base al mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y la tabla en resultados/metricas.csv. Las
partes marcadas con TODO las completo; mientras falten, el guion las
reporta como pendientes y sigue. El informe, docs/informe.md, cita las cifras
y las figuras que produce este guion.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, ingenuo)
import itertools  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

warnings.filterwarnings("ignore")
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

# ============================================================
# 1. La serie
# ============================================================
# Cambie estos cuatro datos por los de la serie y documente la
# fuente en datos/README.md. La serie de ejemplo solo sirve para probar que
# el guion corre.
ARCHIVO = "datos/comercio_minorista_colombia.csv"
FECHA = "observation_date"
VALOR = "COLSLRTTO01IXOBM"
PERIODO = 12

# Lo que se hizo con los huecos y con los datos atípicos, en una línea. Va
# a la columna `decision` de la tabla de métricas.
DECISION = ("Sin huecos (interpolación lineal de respaldo); "
            "atípicos estacionales de diciembre conservados")


def preparar(serie):
    """Aplica la decisión sobre huecos y datos atípicos.

    La fuente trae los 135 meses completos, sin vacíos: la interpolación
    lineal queda solo como respaldo y no toca valores reales. Los máximos de
    diciembre son el pico estacional real del comercio minorista (índice por
    encima de 160 en 2021-2023), no errores, así que se conservan.
    """
    return serie.interpolate(method="linear")


def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.set_title(f"{VALOR}: serie mensual")
    eje.set_ylabel(VALOR)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. Descomposición
# ============================================================

def descomponer(serie):
    """STL robusta con el periodo de la serie; guarda figuras/stl.png."""
    stl = STL(serie, period=PERIODO, robust=True).fit()
    figura, ejes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    ejes[0].plot(serie.index, serie.to_numpy())
    ejes[0].set_title(f"{VALOR}: serie observada")
    ejes[1].plot(stl.trend.index, stl.trend.to_numpy(), color="tab:orange")
    ejes[1].set_title("Tendencia")
    ejes[2].plot(stl.seasonal.index, stl.seasonal.to_numpy(), color="tab:green")
    ejes[2].set_title("Estacional")
    ejes[3].plot(stl.resid.index, stl.resid.to_numpy(), color="tab:red")
    ejes[3].set_title("Residuo")
    figura.tight_layout()
    figura.savefig(FIGURAS / "stl.png", dpi=120)
    plt.close(figura)

    estacional = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    mes_alto = estacional.idxmax()
    mes_bajo = estacional.idxmin()
    residuo_abs = stl.resid.abs()
    print(f"STL robusta (periodo {PERIODO}): tendencia "
          f"{stl.trend.iloc[0]:.1f} -> {stl.trend.iloc[-1]:.1f}")
    print(f"  mes estacional más alto: {mes_alto} (+{estacional.max():.1f}); "
          f"más bajo: {mes_bajo} ({estacional.min():.1f})")
    print(f"  residuo más grande: {residuo_abs.idxmax():%Y-%m} "
          f"({stl.resid.loc[residuo_abs.idxmax()]:.1f})")


# ============================================================
# 3. Estacionariedad e identificación
# ============================================================

def estacionariedad(serie):
    """ADF sobre la serie, su diferencia regular, la estacional y las dos.

    Imprime el valor p de cada versión, elige d y D, y guarda la ACF y la
    PACF de la serie elegida en figuras/acf_pacf.png. Devuelve (d, D).
    """
    def p_adf(x, regresion, nombre):
        p = adfuller(x.dropna(), regression=regresion, autolag="AIC")[1]
        print(f"  ADF {nombre} (regression={regresion}): p = {p:.4f}")
        return p

    print("Prueba ADF (H0: raíz unitaria, la serie no es estacionaria):")
    p_adf(serie, "ct", "serie")
    p_adf(serie.diff(1), "c", "diferencia regular (d=1)")
    p_adf(serie.diff(PERIODO), "ct", "diferencia estacional (D=1)")
    p_adf(serie.diff(1).diff(PERIODO), "c", "ambas diferencias (d=1, D=1)")

    # d=1 quita la tendencia; D=1 quita la raíz unitaria estacional.
    d, D = 1, 1
    estacionaria = serie
    if D == 1:
        estacionaria = estacionaria.diff(PERIODO)
    if d == 1:
        estacionaria = estacionaria.diff(1)
    estacionaria = estacionaria.dropna()
    print(f"  d = {d}, D = {D}: {len(estacionaria)} meses en la serie estacionaria")

    figura, ejes = plt.subplots(1, 2, figsize=(11, 3.6))
    plot_acf(estacionaria, lags=24, ax=ejes[0])
    plot_pacf(estacionaria, lags=24, ax=ejes[1])
    ejes[0].set_title("ACF de la serie estacionaria")
    ejes[1].set_title("PACF de la serie estacionaria")
    figura.tight_layout()
    figura.savefig(FIGURAS / "acf_pacf.png", dpi=120)
    plt.close(figura)
    return d, D


# ============================================================
# 4. Modelos clásicos
# ============================================================
# Cada modelo es una función modelo(entrenamiento, h) que devuelve h
# pronósticos y solo usa `entrenamiento`. Así entra al backtest común.

def elegir_sarima(serie, d, D):
    """Rejilla pequeña de órdenes (p, q, P, Q) comparada por AIC.

    Ajusta con toda la serie preparada, imprime la tabla de AIC, revisa los
    residuos del elegido con Ljung-Box y guarda figuras/residuos.png.
    Devuelve el orden y el orden estacional elegidos.
    """
    filas = []
    for p_, q_, P_, Q_ in itertools.product([0, 1], repeat=4):
        ajuste = ARIMA(serie, order=(p_, d, q_),
                       seasonal_order=(P_, D, Q_, PERIODO)).fit()
        filas.append({"p": p_, "q": q_, "P": P_, "Q": Q_, "AIC": ajuste.aic})
    rejilla = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    print("Rejilla SARIMA (p,d,q)(P,D,Q)12 ordenada por AIC:")
    print(rejilla.round(1).to_string(index=False))

    orden = (int(rejilla.loc[0, "p"]), d, int(rejilla.loc[0, "q"]))
    estacional = (int(rejilla.loc[0, "P"]), D, int(rejilla.loc[0, "Q"]), PERIODO)
    mejor = ARIMA(serie, order=orden, seasonal_order=estacional).fit()
    print(f"Elegido: SARIMA{orden}{estacional}, AIC {mejor.aic:.1f}")

    residuos = mejor.resid.iloc[d + PERIODO * D:]
    lb = acorr_ljungbox(residuos, lags=[12, 24])
    print("Ljung-Box de los residuos (H0: ruido blanco):")
    print(lb.round(3).to_string())

    figura, ejes = plt.subplots(2, 1, figsize=(9, 5))
    ejes[0].plot(mejor.resid.index, mejor.resid.to_numpy())
    ejes[0].set_title("Residuos del SARIMA elegido")
    plot_acf(residuos, lags=24, ax=ejes[1])
    ejes[1].set_title("ACF de los residuos")
    figura.tight_layout()
    figura.savefig(FIGURAS / "residuos.png", dpi=120)
    plt.close(figura)
    return orden, estacional


def modelo_sarima(orden, orden_estacional):
    """Devuelve la función modelo(entrenamiento, h) del SARIMA elegido."""
    def sarima(entrenamiento, h):
        ajuste = ARIMA(entrenamiento, order=orden,
                       seasonal_order=orden_estacional).fit()
        return ajuste.forecast(h).to_numpy()
    return sarima


def exogenas(serie):
    """DataFrame de festivos de Colombia por mes, conocido de antemano.

    Cubre el índice de la serie y los 12 meses que siguen a su final.
    """
    import holidays
    anios = range(serie.index[0].year, serie.index[-1].year + 2)
    indice = pd.date_range(serie.index[0], periods=len(serie) + PERIODO, freq="MS")
    calendario = holidays.country_holidays("CO", years=anios)
    fechas = pd.to_datetime(list(calendario.keys()))
    conteo = pd.Series(1, index=fechas).resample("MS").sum()
    festivos = conteo.reindex(indice, fill_value=0).astype(float).rename("festivos")
    return pd.DataFrame({"festivos": festivos})


def modelo_sarimax(orden, orden_estacional, x):
    """Devuelve modelo(entrenamiento, h) de un SARIMAX con exógenas x.

    Ajusta con x.loc[entrenamiento.index] y pronostica con
    x.loc[fechas_siguientes(entrenamiento, h)], sin fuga: la exógena solo
    usa meses ya conocidos y los 12 futuros se conocen de antemano.
    """
    def sarimax(entrenamiento, h):
        futuro = fechas_siguientes(entrenamiento, h)
        ajuste = ARIMA(entrenamiento, order=orden,
                       seasonal_order=orden_estacional,
                       exog=x.loc[entrenamiento.index]).fit()
        return ajuste.forecast(h, exog=x.loc[futuro]).to_numpy()
    return sarimax


# ============================================================
# 5. La tabla de métricas
# ============================================================

def fila(nombre, serie, modelo):
    _, promedio = backtest(serie, modelo)
    return {"modelo": nombre, "decision": DECISION,
            "RMSE": round(promedio["RMSE"], 2), "MAE": round(promedio["MAE"], 2),
            "MAPE": round(promedio["MAPE"], 2)}


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def main():
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    serie = preparar(cruda)
    graficar_serie(serie)

    filas = [fila("ingenuo", serie, ingenuo),
             fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]

    try:
        descomponer(serie)
    except NotImplementedError as error:
        pendiente("parte 2", error)

    try:
        d, D = estacionariedad(serie)
    except NotImplementedError as error:
        pendiente("parte 3", error)
        d = D = None

    if d is not None:
        try:
            orden, estacional = elegir_sarima(serie, d, D)
            filas.append(fila(f"SARIMA{orden}{estacional}", serie,
                              modelo_sarima(orden, estacional)))
            x = exogenas(serie)
            filas.append(fila(f"SARIMAX{orden}{estacional} + exógenas", serie,
                              modelo_sarimax(orden, estacional, x)))
        except NotImplementedError as error:
            pendiente("parte 4", error)

    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.drop(columns="decision").to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])


if __name__ == "__main__":
    main()
