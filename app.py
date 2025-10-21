import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from io import BytesIO

# ===============================
# 🟦 CONFIGURACIÓN INICIAL
# ===============================
st.set_page_config(page_title="Modelos de Series de Tiempo", layout="centered")
st.title("📈 Predicción automática de Series de Tiempo")
st.write("Sube tu archivo Excel y obtén el mejor modelo para cada producto.")

# ===============================
# 🟩 CARGA DE ARCHIVO
# ===============================
uploaded_file = st.file_uploader("📤 Sube tu archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.write("### Vista previa del archivo:")
    st.dataframe(df.head())

    # ===============================
    # 🧭 CREAR COLUMNA DE PERIODO
    # ===============================
    try:
        df["AÑO"] = df["AÑO"].astype(int)
        df["SEMANA"] = df["SEMANA"].astype(int)
        df["periodo"] = pd.to_datetime(
            df["AÑO"].astype(str) + df["SEMANA"].astype(str) + "1",
            format="%G%V%u"
        )
        st.success("✅ Columna 'periodo' creada exitosamente.")
    except Exception as e:
        st.error(f"Error al crear columna periodo: {e}")

    # ===============================
    # ⚙️ PARÁMETROS DE MODELADO
    # ===============================
    productos = df["PRODUCTO"].unique()
    resultados = []

    st.write("🔄 Ejecutando modelos... esto puede tardar unos segundos.")

    for prod in productos:
        df_prod = df[df["PRODUCTO"] == prod].copy()
        df_prod = df_prod[["periodo", "CANTIDAD"]].sort_values("periodo")

        # Asegurar índice temporal
        df_prod = df_prod.set_index("periodo").asfreq("W-MON")
        df_prod["CANTIDAD"] = df_prod["CANTIDAD"].interpolate()

        # Separar train/test
        train_size = int(len(df_prod) * 0.8)
        train, test = df_prod.iloc[:train_size], df_prod.iloc[train_size:]

        if len(train) < 10:
            continue  # muy pocos datos

        # ===============================
        # MODELOS
        # ===============================

        modelos = {}
        errores = {}

        # Modelo 1: Holt-Winters
        try:
            hw = ExponentialSmoothing(train["CANTIDAD"], trend="add", seasonal=None).fit()
            pred_hw = hw.forecast(len(test))
            mae_hw = mean_absolute_error(test["CANTIDAD"], pred_hw)
            rmse_hw = math.sqrt(mean_squared_error(test["CANTIDAD"], pred_hw))
            modelos["Holt-Winters"] = (hw, mae_hw, rmse_hw)
        except:
            pass

        # Modelo 2: ARIMA
        try:
            arima = ARIMA(train["CANTIDAD"], order=(1, 1, 1)).fit()
            pred_arima = arima.forecast(len(test))
            mae_arima = mean_absolute_error(test["CANTIDAD"], pred_arima)
            rmse_arima = math.sqrt(mean_squared_error(test["CANTIDAD"], pred_arima))
            modelos["ARIMA(1,1,1)"] = (arima, mae_arima, rmse_arima)
        except:
            pass

        # Modelo 3: Prophet
        try:
            df_prophet = train.reset_index().rename(columns={"periodo": "ds", "CANTIDAD": "y"})
            model_prophet = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            model_prophet.fit(df_prophet)
            future = model_prophet.make_future_dataframe(periods=len(test), freq='W-MON')
            forecast = model_prophet.predict(future)
            pred_prophet = forecast.tail(len(test))["yhat"].values
            mae_p = mean_absolute_error(test["CANTIDAD"], pred_prophet)
            rmse_p = math.sqrt(mean_squared_error(test["CANTIDAD"], pred_prophet))
            modelos["Prophet"] = (model_prophet, mae_p, rmse_p)
        except:
            pass

        # ===============================
        # 📊 Seleccionar mejor modelo
        # ===============================
        if not modelos:
            continue

        mejor = min(modelos.items(), key=lambda x: x[1][1])  # menor MAE
        nombre_mejor, (modelo_mejor, mae, rmse) = mejor

        # Pronóstico de próximas 2 semanas
        if nombre_mejor == "Prophet":
            future2 = modelo_mejor.make_future_dataframe(periods=2, freq='W-MON')
            forecast2 = modelo_mejor.predict(future2).tail(2)["yhat"].values
        else:
            forecast2 = modelo_mejor.forecast(2)

        resultados.append({
            "PRODUCTO": prod,
            "MEJOR_MODELO": nombre_mejor,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "PRED_SEM1": round(forecast2[0], 2),
            "PRED_SEM2": round(forecast2[1], 2)
        })

    # ===============================
    # 📦 EXPORTAR RESULTADOS
    # ===============================
    if resultados:
        df_result = pd.DataFrame(resultados)
        st.write("### 📊 Resultados:")
        st.dataframe(df_result)

        # Botón de descarga
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_result.to_excel(writer, index=False, sheet_name="Resultados")
        st.download_button(
            label="📥 Descargar resultados en Excel",
            data=output.getvalue(),
            file_name="resultados_modelos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No se pudo generar ningún modelo válido.")
else:
    st.info("👆 Esperando que subas un archivo Excel.")
