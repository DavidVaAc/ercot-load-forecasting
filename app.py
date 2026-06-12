import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
import time
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar

# =====================================================================
# 1. CONFIGURACIÓN DE LA APP & CREDENCIALES
# =====================================================================
st.set_page_config(page_title="Predicción de Demanda Eléctrica - Texas (ERCOT)", layout="wide")

EIA_API_KEY = st.secrets["EIA_API_KEY"]
VISUAL_CROSSING_KEY = st.secrets["VISUAL_CROSSING_KEY"]

EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Lista exacta de columnas requeridas por el LightGBM (Mismo orden del R&D)
COLUMNS_ORDER = [
    'houston_temp', 'houston_humidity', 'houston_apparent_temp', 'houston_wind_speed',
    'dallas_temp', 'dallas_humidity', 'dallas_apparent_temp', 'dallas_wind_speed',
    'austin_temp', 'austin_humidity', 'austin_apparent_temp', 'austin_wind_speed',
    'texas_avg_temp', 'hour', 'day_of_week', 'month', 'is_weekend',
    'load_lag_24', 'load_lag_48', 'load_lag_168',
    'load_rolling_mean_24h', 'load_rolling_std_24h', 'load_rolling_max_24h',
    'is_holiday', 'temp_delta_24h', 'CDD', 'HDD'
]

@st.cache_resource
def load_saved_model():
    return lgb.Booster(model_file='data/modelo_final_ercot_lgb.json')

# =====================================================================
# 2. FUNCIONES DE EXTRACCIÓN EN VIVO CON ARQUITECTURA EN CASCADA
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_live_data():
    hoy = datetime.utcnow()
    hace_8_dias = hoy - timedelta(days=8)
    
    backup_eia_path = 'data/backup_live_eia.csv'
    backup_clima_path = 'data/backup_live_clima.csv'
    usando_backup = False
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    # --- A. EXTRAER DEMANDA HISTÓRICA RECIENTE (EIA) ---
    params_eia = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ERCO",
        "facets[type][]": "D",
        "start": hace_8_dias.strftime("%Y-%m-%dT%H"),
        "end": hoy.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000
    }
    
    try:
        response_eia = requests.get(EIA_URL, params=params_eia, headers=HEADERS, timeout=(5, 15))
        response_eia.raise_for_status()
        res_eia = response_eia.json()
        
        df_eia = pd.DataFrame(res_eia['response']['data'])
        df_eia['timestamp'] = pd.to_datetime(df_eia['period'])
        df_eia = df_eia.set_index('timestamp').sort_index()
        df_eia['value'] = df_eia['value'].astype(float)
        
        os.makedirs('data', exist_ok=True)
        df_eia.to_csv(backup_eia_path)
    except Exception as e:
        st.sidebar.warning(f"EIA API Offline, cargando respaldo local. Motivo: {e}")
        if os.path.exists(backup_eia_path):
            df_eia = pd.read_csv(backup_eia_path, index_col='timestamp', parse_dates=True)
            usando_backup = True
        else:
            return None, None, False

    # --- B. EXTRAER CLIMA: CASCADA INTELIGENTE ---
    df_clima = None
    api_clima_exito = False
    
    # 🌲 NIVEL 1: Intentar Open-Meteo (Fuente Principal)
    params_om = {
        "latitude": [29.7604, 32.7767, 30.2672],
        "longitude": [-95.3698, -96.7970, -97.7431],
        "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m",
        "past_days": 8,
        "forecast_days": 2,
        "timezone": "UTC"
    }
    try:
        response_om = requests.get(OPEN_METEO_URL, params=params_om, headers=HEADERS, timeout=(5, 15))
        response_om.raise_for_status()
        res_om = response_om.json()
        
        ciudades = ["houston", "dallas", "austin"]
        df_clima = pd.DataFrame({"timestamp": pd.to_datetime(res_om[0]["hourly"]["time"])})
        
        for i, ciudad in enumerate(ciudades):
            h = res_om[i]["hourly"]
            df_clima[f"{ciudad}_temp"] = h["temperature_2m"]
            df_clima[f"{ciudad}_humidity"] = h["relative_humidity_2m"]
            df_clima[f"{ciudad}_apparent_temp"] = h["apparent_temperature"]
            df_clima[f"{ciudad}_wind_speed"] = h["wind_speed_10m"]
            
        df_clima = df_clima.set_index('timestamp').sort_index()
        api_clima_exito = True
        st.sidebar.success("Clima extraído con éxito desde Open-Meteo (Fuente Principal)")
    except Exception as e_om:
        st.sidebar.warning(f"Open-Meteo falló ({e_om}). Saltando a Nivel 2: Visual Crossing...")
        
        # 🌤️ NIVEL 2: Fallback a Visual Crossing (Fuente Secundaria)
        start_str = hace_8_dias.strftime("%Y-%m-%d")
        end_str = (hoy + timedelta(days=2)).strftime("%Y-%m-%d")
        ciudades = ["houston", "dallas", "austin"]
        vc_exito = True
        df_clima_vc = None
        
        for ciudad in ciudades:
            url_vc = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{ciudad}/{start_str}/{end_str}"
            params_vc = {
                "key": VISUAL_CROSSING_KEY,
                "unitGroup": "metric",
                "include": "hours",
                "contentType": "json",
                "timezone": "Z" # Sincronización horaria absoluta en UTC
            }
            try:
                response_vc = requests.get(url_vc, params=params_vc, headers=HEADERS, timeout=(5, 15))
                response_vc.raise_for_status()
                res_vc = response_vc.json()
                
                hours_data = []
                for day in res_vc['days']:
                    for hour in day['hours']:
                        ts_str = f"{day['datetime']} {hour['datetime']}"
                        hours_data.append({
                            "timestamp": pd.to_datetime(ts_str),
                            f"{ciudad}_temp": float(hour['temp']),
                            f"{ciudad}_humidity": float(hour['humidity']),
                            f"{ciudad}_apparent_temp": float(hour['feelslike']),
                            f"{ciudad}_wind_speed": float(hour['windspeed'])
                        })
                
                df_ciudad = pd.DataFrame(hours_data).set_index('timestamp')
                if df_clima_vc is None:
                    df_clima_vc = df_ciudad
                else:
                    df_clima_vc = df_clima_vc.join(df_ciudad, how='outer')
                
                time.sleep(1) # Evitar penalización por ráfaga
            except Exception as e_vc:
                st.sidebar.error(f"Visual Crossing falló en {ciudad}: {e_vc}")
                vc_exito = False
                break
                
        if vc_exito and df_clima_vc is not None:
            df_clima = df_clima_vc
            api_clima_exito = True
            st.sidebar.success("Clima extraído con éxito desde Visual Crossing (Respaldo)")

    # 📦 NIVEL 3: Si ambas APIs caen, usar datos estáticos de Contingencia Local
    if not api_clima_exito:
        if os.path.exists(backup_clima_path):
            df_clima = pd.read_csv(backup_clima_path, index_col='timestamp', parse_dates=True)
            usando_backup = True
            st.sidebar.warning("Usando almacenamiento local de contingencia para Clima.")
        else:
            return None, None, False
    else:
        # Si alguna API respondió con éxito, guardamos una copia fresca para futuras contingencias
        os.makedirs('data', exist_ok=True)
        df_clima.to_csv(backup_clima_path)
        
    # Redondeo absoluto de marcas de tiempo para evitar microsegundos huérfanos
    if df_eia is not None:
        df_eia.index = df_eia.index.floor('h')
    if df_clima is not None:
        df_clima.index = df_clima.index.floor('h')
        
    return df_eia, df_clima, usando_backup

# =====================================================================
# 3. PIPELINE DE INFERENCIA (FEATURE ENGINEERING EN VIVO)
# =====================================================================
def build_live_features(df_eia, df_clima):
    df_live = df_clima.join(df_eia['value'], how='left')
    
    df_live['hour'] = df_live.index.hour
    df_live['day_of_week'] = df_live.index.dayofweek
    df_live['month'] = df_live.index.month
    df_live['is_weekend'] = df_live['day_of_week'].isin([5, 6]).astype(int)
    
    df_live['load_lag_24'] = df_live['value'].shift(24)
    df_live['load_lag_48'] = df_live['value'].shift(48)
    df_live['load_lag_168'] = df_live['value'].shift(168)
    
    df_live['load_rolling_mean_24h'] = df_live['value'].shift(24).rolling(window=24).mean()
    df_live['load_rolling_std_24h'] = df_live['value'].shift(24).rolling(window=24).std()
    df_live['load_rolling_max_24h'] = df_live['value'].shift(24).rolling(window=24).max()
    
    df_live["texas_avg_temp"] = df_live[["houston_temp", "dallas_temp", "austin_temp"]].mean(axis=1)
    df_live['temp_delta_24h'] = df_live['texas_avg_temp'] - df_live['texas_avg_temp'].shift(24)
    df_live['CDD'] = np.maximum(0, df_live['texas_avg_temp'] - 18.3)
    df_live['HDD'] = np.maximum(0, 18.3 - df_live['texas_avg_temp'])
    
    cal = USFederalHolidayCalendar()
    feriados = cal.holidays(start=df_live.index.min(), end=df_live.index.max())
    df_live['is_holiday'] = df_live.index.normalize().isin(feriados).astype(int)
    
    # Relleno de seguridad anti-NaNs para variables temporales complejas
    for col in COLUMNS_ORDER:
        if col in df_live.columns:
            df_live[col] = df_live[col].ffill().bfill()
            
    # 🛠️ FILTRADO DETERMINISTA BASADO EN LA ÚLTIMA FECHA REAL DE LA EIA
    ultimo_ts_real = df_eia.index.max()
    
    # Bloque Futuro: Las 24 horas siguientes al último dato conocido
    df_predict = df_live[df_live.index > ultimo_ts_real].head(24)
    
    # Bloque Pasado: Las últimas 24 horas con datos reales para control de calidad
    df_past = df_live[df_live.index <= ultimo_ts_real].tail(24)
    
    return df_predict[COLUMNS_ORDER], df_past[COLUMNS_ORDER], df_past['value']

# =====================================================================
# 4. INTERFAZ GRÁFICA (STREAMLIT)
# =====================================================================
st.title("⚡ Sistema Predictivo de Carga Eléctrica: ERCOT (Texas)", anchor="panel-principal")
st.markdown("Dashboard de MLOps con arquitectura redundante para el pronóstico de demanda energética de la red eléctrica tejana.")

# 🕒 RELOJ UTC EN VIVO EN UN CONTENEDOR SEGURO
html_reloj = """
<div style="background-color: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #2d2d2d; text-align: center;">
    <p style="margin: 0; color: #777; font-size: 11px; font-weight: bold; letter-spacing: 1.5px; font-family: sans-serif;">HORA OPERATIVA DE LA RED (UTC)</p>
    <p id="utc-live-clock" style="margin: 5px 0 0 0; color: #00ffcc; font-size: 24px; font-weight: bold; font-family: 'Courier New', monospace; text-shadow: 0 0 10px rgba(0,255,204,0.3);">Sincronizando reloj...</p>
</div>

<script>
    function updateUTClock() {
        const now = new Date();
        const year = now.getUTCFullYear();
        const month = String(now.getUTCMonth() + 1).padStart(2, '0');
        const day = String(now.getUTCDate()).padStart(2, '0');
        const hours = String(now.getUTCHours()).padStart(2, '0');
        const minutes = String(now.getUTCMinutes()).padStart(2, '0');
        const seconds = String(now.getUTCSeconds()).padStart(2, '0');
        
        const timeString = `${year}-${month}-${day} | ${hours}:${minutes}:${seconds} UTC`;
        document.getElementById('utc-live-clock').innerText = timeString;
    }
    setInterval(updateUTClock, 1000);
    updateUTClock();
</script>
"""

# Renderizamos el HTML/JS dentro de un componente nativo de Streamlit
components.html(html_reloj, height=100)

try:
    model = load_saved_model()
    df_eia, df_clima, usando_backup = fetch_live_data()
    
    if df_eia is None or df_clima is None:
        st.error("🔌 Error Crítico de Inicialización: No se pudo establecer comunicación con las APIs climáticas ni existen archivos semilla en la carpeta data/.")
        st.stop()
        
    if usando_backup:
        st.warning("⚠️ Modo de Contingencia Activo: Las APIs externas están congestionadas o sin créditos. Operando de forma segura con datos históricos en caché local.")
        
    X_live_future, X_live_past, y_past_real = build_live_features(df_eia, df_clima)
    
    if len(X_live_future) == 0 or len(X_live_past) == 0:
        st.warning("Alineando flujos de datos temporales... Intenta recargar la página en unos segundos.")
    else:
        # Ejecutar inferencias matemáticas
        predictions_future = model.predict(X_live_future)
        df_resultados_futuro = pd.DataFrame(index=X_live_future.index)
        df_resultados_futuro['Demanda_Proyectada_MW'] = predictions_future
        
        predictions_past = model.predict(X_live_past)
        df_resultados_pasado = pd.DataFrame(index=X_live_past.index)
        df_resultados_pasado['Real'] = y_past_real
        df_resultados_pasado['Predicho'] = predictions_past
        
        # --- PANEL DE MÉTRICAS GENERALES (REDISEÑO ESTRATÉGICO) ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 🔮 MÉTRICA 1: Inferencia del Pico Futuro + Delta de Riesgo Proyectado
            pico_max = df_resultados_futuro['Demanda_Proyectada_MW'].max()
            es_pico_critico = pico_max > 65000
            
            delta_pico = "Riesgo de Alta Demanda" if es_pico_critico else "Margen Operativo Seguro"
            color_pico = "red" if es_pico_critico else "green"
            
            st.metric(
                label="Pico Máximo Proyectado (24h)",
                value=f"{pico_max:,.2f} MW",
                delta=delta_pico,
                delta_color=color_pico
            )
            
        with col2:
            # 🌡️ MÉTRICA 2: Termómetro Dinámico Estilizado (Se queda igual de robusto)
            temp_actual = X_live_future['texas_avg_temp'].iloc[0]
            
            if temp_actual < 12.0:
                color_termometro = "blue"
                estado_temp = "Estrés por Frío" if temp_actual < 4.0 else "Inercia Invernal"
            elif temp_actual <= 26.0:
                color_termometro = "green"
                estado_temp = "Zona de Confort"
            elif temp_actual <= 34.0:
                color_termometro = "orange"
                estado_temp = "Carga Térmica Media"
            else:
                color_termometro = "red"
                estado_temp = "Calor Crítico (HVAC Máximo)"
                
            st.metric(
                label="Temperatura Promedio Actual",
                value=f"{temp_actual:.1f} °C",
                delta=f"{estado_temp}",
                delta_color=color_termometro
            )
            
        with col3:
            # 🔌 MÉTRICA 3: Estado Actual del Sistema (Basado en el último dato real de la EIA)
            demanda_actual = y_past_real.iloc[-1]  # Extrae el valor más reciente del pasado conocido
            es_actual_critico = demanda_actual > 65000
            
            estado_actual = "Estrés Operativo Activo" if es_actual_critico else "Operación Estable (Normal)"
            color_actual = "red" if es_actual_critico else "green"
            
            st.metric(
                label="Demanda Actual de la Red",
                value=f"{demanda_actual:,.2f} MW",
                delta=estado_actual,
                delta_color=color_actual
            )
        # --- GRAFICA 1: PRÓXIMAS 24 HORAS (CON EJE SECUNDARIO DE TEMPERATURA) ---
        st.space()
        st.subheader("📈 Curva de Demanda Eléctrica Proyectada (Próximas 24 Horas)", anchor="pronostico-24h")
        
        # Inicializamos el gráfico con doble eje Y
        fig_fut = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Eje Principal (Izquierdo): Demanda Proyectada
        fig_fut.add_trace(
            go.Scatter(x=df_resultados_futuro.index, y=df_resultados_futuro['Demanda_Proyectada_MW'], 
                       mode='lines+markers', name='Pronóstico Carga (MW)', line=dict(color='cyan', width=3)),
            secondary_y=False
        )
        
        # Eje Secundario (Derecho): Temperatura Proyectada
        fig_fut.add_trace(
            go.Scatter(x=X_live_future.index, y=X_live_future['texas_avg_temp'], 
                       mode='lines', name='Pronóstico Temperatura (°C)', line=dict(color='rgba(251, 140, 0, 0.6)', width=2, dash='dot')),
            secondary_y=True
        )
        
        # Estilización del layout doble eje
        fig_fut.update_layout(template="plotly_dark", xaxis_title="Fecha y Hora (UTC)", hovermode="x unified")
        fig_fut.update_yaxes(title_text="Demanda de Energía (MW)", secondary_y=False)
        fig_fut.update_yaxes(title_text="Temperatura Promedio (°C)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_fut, use_container_width=True)
        
        # --- GRAFICA 2: CONTROL DE CALIDAD ---
        st.space()
        st.markdown("---")
        st.subheader("🔄 Control de Calidad: Rendimiento del Modelo en las Últimas 24 Horas", anchor="control-calidad")
        
        errores = df_resultados_pasado['Real'] - df_resultados_pasado['Predicho']
        live_mae = np.mean(np.abs(errores))
        live_rmse = np.sqrt(np.mean(errores ** 2))
        live_mape = np.mean(np.abs(errores / df_resultados_pasado['Real'])) * 100
        
        # 📊 BENCHMARK REAL DE MLOps (Test Ciego 2025)
        BASELINE_MAPE = 3.11
        mape_desviacion = live_mape - BASELINE_MAPE
        
        # Lógica de semáforo de 3 niveles para una transición suave (Green -> Orange -> Red)
        if mape_desviacion <= 0:
            color_mape = "green"
            delta_mape_texto = f"{mape_desviacion:.2f}% (Óptimo vs R&D)"
        elif 0 < mape_desviacion <= 0.50:
            color_mape = "orange"
            delta_mape_texto = f"+{mape_desviacion:.2f}% (Margen de Tolerancia)"
        else:
            color_mape = "red"
            delta_mape_texto = f"+{mape_desviacion:.2f}% (Degradación Crítica)"
        
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric(
                label="📊 MAPE en Vivo (Últimas 24h)", 
                value=f"{live_mape:.2f} %",
                delta=delta_mape_texto,
                delta_color=color_mape
            )
            
        with m_col2: 
            st.metric(label="🎯 MAE en Vivo", value=f"{live_mae:,.1f} MW")
            
        with m_col3: 
            st.metric(label="📉 RMSE en Vivo", value=f"{live_rmse:,.1f} MW")
        
        # Inicializamos el gráfico de control de calidad con doble eje Y
        fig_past = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Eje Principal (Izquierdo): Real vs Predicho
        fig_past.add_trace(
            go.Scatter(x=df_resultados_pasado.index, y=df_resultados_pasado['Real'], 
                       mode='lines+markers', name='Consumo Real EIA (MW)', line=dict(color='#00FF00', width=3)),
            secondary_y=False
        )
        fig_past.add_trace(
            go.Scatter(x=df_resultados_pasado.index, y=df_resultados_pasado['Predicho'], 
                       mode='lines+markers', name='Predicción LightGBM (MW)', line=dict(color='#FFA500', width=2, dash='dash')),
            secondary_y=False
        )
        
        # Eje Secundario (Derecho): Temperatura Real que provocó ese consumo
        fig_past.add_trace(
            go.Scatter(x=X_live_past.index, y=X_live_past['texas_avg_temp'], 
                       mode='lines', name='Temperatura Real (°C)', line=dict(color='rgba(239, 83, 80, 0.5)', width=2, dash='dot')),
            secondary_y=True
        )
        
        fig_past.update_layout(template="plotly_dark", xaxis_title="Fecha y Hora (UTC)", hovermode="x unified")
        fig_past.update_yaxes(title_text="Demanda de Energía (MW)", secondary_y=False)
        fig_past.update_yaxes(title_text="Temperatura Promedio (°C)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_past, use_container_width=True)

        # =====================================================================
        # 📊 SECCIÓN: INTERPRETABILIDAD Y DIAGNÓSTICO EJECUTIVO DEL MODELO
        # =====================================================================
        st.space()
        st.markdown("---")
        st.subheader("📊 Arquitectura y Gobernanza del Modelo LightGBM", anchor="arquitectura-modelo")
        st.markdown("Análisis estratégico y detallado sobre cómo el algoritmo distribuye su atención para predecir la carga de ERCOT.")
        
        # 1. Base de datos detallada (Mapeo R&D)
        importancia_data = {
            'Variable': [
                'temp_delta_24h', 'load_lag_24', 'hour', 'load_rolling_std_24h', 
                'load_rolling_max_24h', 'load_lag_168', 'day_of_week', 'dallas_apparent_temp', 
                'month', 'dallas_temp', 'load_rolling_mean_24h', 'houston_apparent_temp', 
                'austin_apparent_temp', 'load_lag_48', 'dallas_humidity', 'dallas_wind_speed', 
                'houston_wind_speed', 'houston_humidity', 'austin_temp', 'houston_temp', 
                'austin_wind_speed', 'austin_humidity', 'texas_avg_temp', 'HDD', 'is_holiday', 'CDD', 'is_weekend'
            ],
            'Importancia': [
                832, 490, 476, 448, 404, 380, 368, 327, 313, 299, 244, 233, 232, 227, 227, 183, 167, 159, 150, 147, 139, 134, 67, 61, 59, 59, 17
            ]
        }
        df_importance = pd.DataFrame(importancia_data)
        
        # 2. Mapeo a Categorías Macro para la Presentación Ejecutiva
        def categorizar_variable(var):
            if any(x in var for x in ['temp', 'humidity', 'wind', 'CDD', 'HDD', 'delta_24h']):
                return 'Meteorología & Clima (Termodinámica)'
            elif any(x in var for x in ['load_', 'lag_']):
                return 'Inercia de Red & Historial (Autoregresivo)'
            else:
                return 'Calendario & Tiempo (Comportamiento Humano)'
                
        df_importance['Categoría'] = df_importance['Variable'].apply(categorizar_variable)
        
        # Agrupación macro para el gráfico de dona
        df_macro = df_importance.groupby('Categoría')['Importancia'].sum().reset_index()
        
        # 3. Renderizado en dos columnas (Izquierda: Macro, Derecha: Micro)
        col_macro, col_micro = st.columns([2, 2])
        
        with col_macro:
            st.markdown("#### **Resumen Ejecutivo (Pilares)**")
            
            # Paleta de colores ejecutiva con alto contraste oscuro
            colores_macro = ['#aa66ff', '#ffa500', '#00ffcc']
            
            fig_donut = go.Figure()
            fig_donut.add_trace(go.Pie(
                labels=df_macro['Categoría'],
                values=df_macro['Importancia'],
                hole=0.55,
                marker=dict(colors=colores_macro, line=dict(color='#1e1e1e', width=2)),
                hovertemplate="<b>%{label}</b><br>Splits totales: %{value}<br>%{percent}<extra></extra>",
                textinfo='percent',          # 🛠️ Ajuste: Solo muestra el porcentaje adentro
                insidetextfont=dict(size=18), # 🛠️ Ajuste: Texto más grande y legible
                textposition='inside'
            ))
            
            fig_donut.update_layout(
                template="plotly_dark",
                showlegend=True,             
                legend=dict(
                    orientation="h",         # 🌟 Horizontal
                    yanchor="bottom",
                    y=1.1,                  # 🌟 La posiciona justo arriba del gráfico
                    xanchor="center",
                    x=0.5,
                    font=dict(size=16)       # 🌟 Texto más grande y prominente
                ),
                height=500,
                margin=dict(l=20, r=20, t=80, b=10) # 🌟 Más espacio arriba (t) y menos abajo (b)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_micro:
            st.markdown("#### **Frontera de Decisión (Detalle de Variables)**")
            
            df_micro_sorted = df_importance.sort_values(by='Importancia', ascending=True)
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=df_micro_sorted['Variable'],
                x=df_micro_sorted['Importancia'],
                orientation='h',
                marker=dict(
                    color=df_micro_sorted['Importancia'],
                    colorscale='Viridis',
                    showscale=False
                ),
                hovertemplate="<b>Variable:</b> %{y}<br><b>Splits:</b> %{x}<extra></extra>"
            ))
            
            fig_imp.update_layout(
                template="plotly_dark",
                height=500,
                margin=dict(l=150, r=20, t=10, b=40),
                xaxis_title="Número de Divisiones (Splits)",
                yaxis=dict(tickfont=dict(size=10)),
                hovermode="closest"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        # =====================================================================
        # 🛠️ SECCIÓN: EXPORTACIÓN DE DATOS & NAVEGACIÓN
        # =====================================================================
        st.sidebar.markdown("---")
        st.sidebar.subheader("📍 Índice de Navegación")
        st.sidebar.markdown("""
        - ⚡ [Panel de Control Principal](#panel-principal)
        - 📈 [Pronóstico Próximas 24h](#pronostico-24h)
        - 🔄 [Validación Histórica](#control-calidad)
        - 📊 [Arquitectura del Modelo](#arquitectura-modelo)
        """)

        st.sidebar.markdown("---")
        st.sidebar.subheader("💾 Exportar Resultados")

        # Consolidamos las predicciones para la descarga
        df_export = df_resultados_futuro.copy()
        df_export['Unidad'] = 'MW'
        df_export['Estado_Red'] = ["Alerta" if x > 65000 else "Normal" for x in df_export['Demanda_Proyectada_MW']]

        csv = df_export.to_csv().encode('utf-8')

        st.sidebar.download_button(
            label="📥 Descargar Predicciones (CSV)",
            data=csv,
            file_name=f'prediccion_ercot_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
        )

        # =====================================================================
        # 💼 TARJETA DE CONTACTO PROFESIONAL (AL FINAL DEL SIDEBAR)
        # =====================================================================
        st.sidebar.markdown("---")
        st.sidebar.subheader("👨‍💻 Desarrollador del Sistema")

        # Un contenedor estilizado para tu tarjeta de presentación
        st.sidebar.markdown(
            """
<div style="background-color: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #2d2d2d; margin-bottom: 10px;">
<p style="margin: 0; font-weight: bold; color: #fff; font-size: 16px;">David Valle</p>
<p style="margin: 2px 0 12px 0; color: #00ffcc; font-size: 12px; font-weight: 500; letter-spacing: 0.5px;">Physicist & Data Scientist</p>

<div style="display: flex; flex-direction: column; gap: 8px; font-size: 13px;">
<a href="https://davidvaac.github.io/DavidVaAc/" target="_blank" style="color: #cbd5e1; text-decoration: none; display: flex; align-items: center; gap: 8px;">
    💼 <b>Portafolio:</b> /DavidVaAc
</a>
<a href="https://www.linkedin.com/in/david-fernando-valle-acosta" target="_blank" style="color: #cbd5e1; text-decoration: none; display: flex; align-items: center; gap: 8px;">
    🔵 <b>LinkedIn:</b> /in/david-fernando-valle-acosta
</a>
<a href="https://github.com/DavidVaAc" target="_blank" style="color: #cbd5e1; text-decoration: none; display: flex; align-items: center; gap: 8px;">
    🐈 <b>GitHub:</b> /DavidVaAc
</a>
<a href="mailto:davidfervalle@gmail.com" style="color: #cbd5e1; text-decoration: none; display: flex; align-items: center; gap: 8px;">
    ✉️ <b>Email:</b> davidfervalle@gmail.com
</a>
</div>
</div>
            """, 
            unsafe_allow_html=True
        )

        # --- Sección Final de Contacto ---
        st.space()
        st.space()
        st.space()
        st.markdown("---")
        st.markdown(
            """
            <p style='text-align: center;'>
                <a href='https://davidvaac.github.io/DavidVaAc/'>💼 Portafolio</a> &nbsp;&nbsp; | &nbsp;&nbsp;
                <a href='https://www.linkedin.com/in/david-fernando-valle-acosta'>🔗 LinkedIn</a> &nbsp;&nbsp; | &nbsp;&nbsp;
                <a href='https://github.com/DavidVaAc'>📁 GitHub</a> &nbsp;&nbsp; | &nbsp;&nbsp;
                <a href='mailto:davidfervalle@gmail.com'>✉️ Email</a>
            </p>
            """, 
            unsafe_allow_html=True
        )

        # Un pie de página discreto
        st.caption("© 2026 | Desarrollado por David Valle Acosta - Físico, UNAM")

except Exception as e:
    st.error(f"Ocurrió un error crítico en el pipeline de producción: {e}")
