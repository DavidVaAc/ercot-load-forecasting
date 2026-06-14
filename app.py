import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score
import gridstatus
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
    <p style="margin: 0; color: #777; font-size: 11px; font-weight: bold; letter-spacing: 1.5px; font-family: sans-serif;">HORA OPERATIVA DE LA RED (TEXAS CT)</p>
    <p id="texas-live-clock" style="margin: 5px 0 0 0; color: #00ffcc; font-size: 24px; font-weight: bold; font-family: 'Courier New', monospace; text-shadow: 0 0 10px rgba(0,255,204,0.3);">Sincronizando telemetría...</p>
</div>

<script>
    function updateTexasClock() {
        const now = new Date();
        
        // Usamos el locale 'sv-SE' (Suecia) porque da el formato estándar ISO YYYY-MM-DD HH:mm:ss de forma nativa
        const formatter = new Intl.DateTimeFormat('sv-SE', {
            timeZone: 'America/Chicago',
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
        
        // Formateamos y estilizamos para mantener tu diseño limpio
        const timeString = formatter.format(now).replace(" ", " | ") + " CT";
        document.getElementById('texas-live-clock').innerText = timeString;
    }
    setInterval(updateTexasClock, 1000);
    updateTexasClock();
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
        # 1. Ejecutar inferencias matemáticas del modelo predictivo
        predictions_future = model.predict(X_live_future)
        df_resultados_futuro = pd.DataFrame(index=X_live_future.index)
        df_resultados_futuro['Demanda_Proyectada_MW'] = predictions_future
        df_resultados_futuro['texas_avg_temp'] = X_live_future['texas_avg_temp']
        
        predictions_past = model.predict(X_live_past)
        df_resultados_pasado = pd.DataFrame(index=X_live_past.index)
        df_resultados_pasado['Real'] = y_past_real
        df_resultados_pasado['Predicho'] = predictions_past
        df_resultados_pasado['texas_avg_temp'] = X_live_past['texas_avg_temp']

        # =====================================================================
        # 🔌 TELEMETRÍA EN TIEMPO REAL (GRIDSTATUS API) - EXTRACCIÓN TEMPRANA
        # =====================================================================
        gridstatus_exito = False
        try:
            ercot = gridstatus.Ercot()
            rt_conditions = ercot.get_real_time_system_conditions()
            capacity_forecast = ercot.get_capacity_forecast()

            # 🌟 REQUERIMIENTO: Extraemos la Demanda Actual Directamente de Gridstatus
            demanda_actual = float(rt_conditions['Actual System Demand'].iloc[0])
            capacity_live = float(rt_conditions['Total System Capacity excluding Ancillary Services'].iloc[0])
            gridstatus_exito = True
        except Exception as e:
            # Fallback seguro en caso de congestión de la API de ERCOT
            demanda_actual = float(y_past_real.iloc[-1])
            capacity_live = 98000 # Proxy de capacidad disponible promedio de verano
            st.sidebar.warning(f"API de ERCOT en tiempo real no disponible, usando fallback. Motivo: {e}")

        # --- 2. CÁLCULO DE VALORES CORE ---
        # (demanda_actual ya está unificada y validada arriba)
        pico_max = df_resultados_futuro['Demanda_Proyectada_MW'].max()
        pico_min = df_resultados_futuro['Demanda_Proyectada_MW'].min()
        
        temp_actual = X_live_future['texas_avg_temp'].iloc[0]
        temp_max_pred = X_live_future['texas_avg_temp'].max()
        temp_min_pred = X_live_future['texas_avg_temp'].min()

        # =====================================================================
        # 🛡️ CÁLCULO DE MÁRGENES DE RESERVA OPERATIVA
        # =====================================================================
        if gridstatus_exito:
            try:
                # Reserva actual calculada con datos homogéneos de Gridstatus
                reserva_actual_pct = ((capacity_live - demanda_actual) / demanda_actual) * 100
                
                # Sincronización del pronóstico horario
                df_cap_hourly = capacity_forecast.copy()
                df_cap_hourly['Interval Start'] = pd.to_datetime(df_cap_hourly['Interval Start'])
                df_cap_hourly.set_index('Interval Start', inplace=True)
                df_cap_hourly = df_cap_hourly[['Available Capacity']].resample('h').mean()
                df_cap_hourly.index = df_cap_hourly.index.tz_convert('UTC').tz_localize(None)
                
                # Unimos con los resultados futuros de nuestro LightGBM
                df_resultados_futuro = df_resultados_futuro.join(df_cap_hourly, how='left')
                
                # 🌟 REQUERIMIENTO CRÍTICO: Rellenar el futuro lejano para evaluar las 24 horas completas
                # Arrastramos la última capacidad conocida en el día
                df_resultados_futuro['Available Capacity'] = df_resultados_futuro['Available Capacity'].ffill().fillna(capacity_live)
                
                # Cálculos vectoriales de las curvas de reserva
                df_resultados_futuro['Reserva_Proyectada_MW'] = df_resultados_futuro['Available Capacity'] - df_resultados_futuro['Demanda_Proyectada_MW']
                df_resultados_futuro['Reserva_Proyectada_Pct'] = (df_resultados_futuro['Reserva_Proyectada_MW'] / df_resultados_futuro['Demanda_Proyectada_MW']) * 100
                
                # 🌟 REQUERIMIENTO: El mínimo real considerando toda la jornada predictiva
                reserva_min_predicha = df_resultados_futuro['Reserva_Proyectada_Pct'].min()
                reserva_min_pct = min(reserva_actual_pct, reserva_min_predicha)
                
            except Exception as e_proc:
                gridstatus_exito = False
                st.sidebar.error(f"Error en procesamiento de curvas de reserva: {e_proc}")

        if not gridstatus_exito:
            # Fallback robusto en espejo si la API se cae por completo
            capacidad_segura_proxy = 98000
            reserva_actual_pct = ((capacidad_segura_proxy - demanda_actual) / demanda_actual) * 100
            reserva_min_pct = ((capacidad_segura_proxy - pico_max) / pico_max) * 100
            df_resultados_futuro['Reserva_Proyectada_MW'] = capacidad_segura_proxy - df_resultados_futuro['Demanda_Proyectada_MW']
            df_resultados_futuro['Reserva_Proyectada_Pct'] = (df_resultados_futuro['Reserva_Proyectada_MW'] / df_resultados_futuro['Demanda_Proyectada_MW']) * 100

# =====================================================================
        # 🛡️ UNIFICACIÓN DE SEMÁFOROS NATIVOS (100% CONSISTENTES)
        # =====================================================================

        # --- 1. SEMÁFOROS DE TEMPERATURA (SE MANTIENEN POR SU PROPIA NATURALEZA) ---
        # Temperatura Actual
        if temp_actual > 35.0:
            msg_tmp_act, col_tmp_act = "Calor Extremo (HVAC)", "red"
        elif temp_actual < 4.0:
            msg_tmp_act, col_tmp_act = "Estrés por Frío", "purple"
        elif 12.0 <= temp_actual <= 26.0:
            msg_tmp_act, col_tmp_act = "Zona de Confort", "green"
        elif 4.0 <= temp_actual < 12.0:
            msg_tmp_act, col_tmp_act = "Transición Fría", "blue"
        else:
            msg_tmp_act, col_tmp_act = "Transición Térmica", "orange"

        # Máxima Temperatura Proyectada
        if temp_max_pred > 35.0:
            msg_tmp_max, col_tmp_max = "Domo de Calor Proyectado", "red"
        elif temp_max_pred > 28.0:
            msg_tmp_max, col_tmp_max = "Elevación Térmica", "orange"
        else:
            msg_tmp_max, col_tmp_max = "Techo Térmico Seguro", "green"

        # Mínima Temperatura Proyectada
        if temp_min_pred < 4.0:
            msg_tmp_min, col_tmp_min = "Helada / Riesgo Térmico", "purple"
        elif temp_min_pred < 12.0:
            msg_tmp_min, col_tmp_min = "Descenso Moderado", "blue"
        else:
            msg_tmp_min, col_tmp_min = "Suelo Térmico Seguro", "green"


        # --- 2. SEMÁFOROS SINCRONIZADOS: ESTADO ACTUAL (TIEMPO REAL) ---
        # Vinculamos la gravedad de la Demanda Actual directamente al Margen de Reserva Actual
        if reserva_actual_pct < 13.75:
            msg_res_act, col_res_act = "Reserva Crítica (Alerta)", "red"
            msg_dem_act, col_dem_act = "Estrés Crítico de Red", "red"
        elif reserva_actual_pct < 20.0:
            msg_res_act, col_res_act = "Reserva Moderada", "orange"
            msg_dem_act, col_dem_act = "Carga Alta sobre Capacidad", "orange"
        else:
            msg_res_act, col_res_act = "Red Solvente (Segura)", "green"
            msg_dem_act, col_dem_act = "Operación Estable", "green"


        # --- 3. SEMÁFOROS SINCRONIZADOS: PRONÓSTICO 24H (MÁXIMOS Y MÍNIMOS) ---
        # Sincronizamos el Pico Máximo de Demanda con la Mínima Reserva Proyectada del día
        if reserva_min_pct < 13.75:
            msg_res_min, col_res_min = "Riesgo de Apagón", "red"
            msg_dem_max, col_dem_max = "Demanda Supera Umbral", "red"
        elif reserva_min_pct < 20.0:
            msg_res_min, col_res_min = "Compromiso de Margen", "orange"
            msg_dem_max, col_dem_max = "Pico de Carga Alto", "orange"
        else:
            msg_res_min, col_res_min = "Colchón Seguro", "green"
            msg_dem_max, col_dem_max = "Margen Seguro", "green"


        # --- 4. SEMÁFORO DE VALLE DE CARGA (INDEPENDIENTE) ---
        # Se mantiene evaluando MW mínimos ya que el exceso de generación base es un problema puramente de carga
        if pico_min < 32000:
            msg_dem_min, col_dem_min = "Valle Crítico (Exceso Gen)", "red"
        elif pico_min < 38000:
            msg_dem_min, col_dem_min = "Valle Bajo (Ajustar Base)", "orange"
        else:
            msg_dem_min, col_dem_min = "Valle Estable", "green"


        # =====================================================================
        # --- RENDERIZADO EN LA INTERFAZ (DISEÑO GRID INDUSTRIAL 4x2) ---
        # =====================================================================
        st.space()
        st.subheader("🕹️ Control Operativo y Demanda Proyectada (Próximas 24 Horas)", anchor="pronostico-24h")
        st.space()
        
        main_col1, main_col2 = st.columns([1, 2.2])

        # =====================================================================
        # BLOQUE IZQUIERDO: MATRIZ DE KPIs 4x2 (Alineación Geométrica Perfecta)
        # =====================================================================
        with main_col1:
            st.subheader("🎛️ KPIs Operativos")
            st.write("")      

            # --- RENGLÓN 1: ESTADO ACTUAL (TIEMPO REAL) ---
            r1_c1, r1_c2 = st.columns(2)
            with r1_c1:
                st.metric(
                    label="⚡ Demanda Actual",
                    value=f"{demanda_actual:,.0f} MW".replace(",", " "),
                    delta=msg_dem_act,
                    delta_color=col_dem_act,
                    delta_arrow="off"
                )
            with r1_c2:
                st.metric(
                    label="🌡️ Temp Actual", # Se usará un emoji estándar limpio
                    value=f"{temp_actual:.1f} °C",
                    delta=msg_tmp_act,
                    delta_color=col_tmp_act,
                    delta_arrow="off"
                )
            
            # --- RENGLÓN 2: ESCENARIO MÁXIMO (PRÓXIMAS 24H) ---
            r2_c1, r2_c2 = st.columns(2)
            with r2_c1:
                st.metric(
                    label="🔺 Demanda Máx",
                    value=f"{pico_max:,.0f} MW".replace(",", " "),
                    delta=msg_dem_max,
                    delta_color=col_dem_max,
                    delta_arrow="off"
                )
            with r2_c2:
                st.metric(
                    label="🔥 Máx Temp",
                    value=f"{temp_max_pred:.1f} °C",
                    delta=msg_tmp_max,
                    delta_color=col_tmp_max,
                    delta_arrow="off"
                )
                
            # --- RENGLÓN 3: ESCENARIO MÍNIMO (PRÓXIMAS 24H) ---
            r3_c1, r3_c2 = st.columns(2)
            with r3_c1:
                st.metric(
                    label="🔻 Demanda Min",
                    value=f"{pico_min:,.0f} MW".replace(",", " "),
                    delta=msg_dem_min,
                    delta_color=col_dem_min,
                    delta_arrow="off"
                )
            with r3_c2:
                st.metric(
                    label="❄️ Mín Temp",
                    value=f"{temp_min_pred:.1f} °C",
                    delta=msg_tmp_min,
                    delta_color=col_tmp_min,
                    delta_arrow="off"
                )

            # --- RENGLÓN 4: SOLVENCIA Y RESERVAS (MÉTRICA DE RIESGO ESTRATÉGICO) ---
            r4_c1, r4_c2 = st.columns(2)
            with r4_c1:
                st.metric(
                    label="🔋 Reserva Actual",
                    value=f"{reserva_actual_pct:.1f} %",
                    delta=msg_res_act,
                    delta_color=col_res_act,
                    delta_arrow="off"
                )
            with r4_c2:
                st.metric(
                    label="🪫 Mín Reserva (24h)",
                    value=f"{reserva_min_pct:.1f} %",
                    delta=msg_res_min,
                    delta_color=col_res_min,
                    delta_arrow="off"
                )

        # =====================================================================
        # BLOQUE DERECHO: VISUALIZACIÓN TRIPLE TRAZO CON DOBLE EJE Y
        # =====================================================================

        # Ejemplo rápido para tus gráficos antes de Plotly:
        df_plot = df_resultados_futuro.copy()
        # Convertimos el índice de UTC a hora de Texas y lo hacemos limpio (naive)
        df_plot.index = df_plot.index.tz_localize('UTC').tz_convert('US/Central').tz_localize(None)

        with main_col2:
            st.subheader("📈 Demanda Proyectada e Impacto Térmico")

            fig_fut = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 1. Carga Proyectada (Eje Izquierdo - MW)
            fig_fut.add_trace(
                go.Scatter(x=df_plot.index, y=df_plot['Demanda_Proyectada_MW'], 
                           mode='lines+markers', name='Carga (MW)', line=dict(color='cyan', width=3)),
                secondary_y=False
            )
            
            # 2. Margen de Reserva (Eje Izquierdo - MW)
            fig_fut.add_trace(
                go.Scatter(x=df_plot.index, y=df_plot['Reserva_Proyectada_MW'], 
                           mode='lines', name='Reserva Disp. (MW)', line=dict(color='#E040FB', width=2, dash='longdash')),
                secondary_y=False
            )
            
            # 3. 🌟 CORREGIDO: Temperatura Proyectada (Eje Derecho - °C)
            fig_fut.add_trace(
                go.Scatter(x=df_plot.index, y=df_plot['texas_avg_temp'], 
                           mode='lines', name='Temperatura (°C)', line=dict(color='rgba(251, 140, 0, 0.6)', width=2, dash='dot')),
                secondary_y=True
            )
            
            # Ajustes estéticos finales de la leyenda externa a la derecha
            fig_fut.update_layout(
                template="plotly_dark", 
                margin=dict(l=10, r=60, t=25, b=10), 
                # 🌟 ACTUALIZADO: Ahora el eje X explícitamente avisa que es la hora de Texas
                xaxis_title="Fecha y Hora (Texas CT)", 
                hovermode="x unified",
                legend=dict(
                    orientation="v",
                    y=0.7,
                    x=1.05,
                    xanchor="left",
                    yanchor="top"
                ) 
            )
            fig_fut.update_yaxes(title_text="Potencia Eléctrica (MW)", secondary_y=False)
            fig_fut.update_yaxes(title_text="Temperatura (°C)", secondary_y=True, showgrid=False)
            
            st.plotly_chart(fig_fut, use_container_width=True)
        
        # --- GRAFICA 2: CONTROL DE CALIDAD (DISEÑO SCADA SIMÉTRICO) ---
        st.space()
        st.markdown("---")
        st.subheader("🔄 Control de Calidad: Rendimiento del Modelo en las Últimas 24 Horas", anchor="control-calidad")
        st.space()
        
        # --- CÁLCULO DE MÉTRICAS AVANZADAS DE ERROR ---
        errores = df_resultados_pasado['Real'] - df_resultados_pasado['Predicho']
        
        live_mape = np.mean(np.abs(errores / df_resultados_pasado['Real'])) * 100
        live_max_ape = np.max(np.abs(errores / df_resultados_pasado['Real'])) * 100
        
        live_mae = np.mean(np.abs(errores))
        live_max_ae = np.max(np.abs(errores))
        
        live_rmse = np.sqrt(np.mean(errores ** 2))
        live_mbe = np.mean(errores) # Sesgo Medio (Bias)

        # --- CÁLCULOS PARA EL RENGLÓN 4 DE CALIDAD ---
        # 1. Calculamos los residuales históricos (Real - Predicho)
        residuales = df_resultados_pasado['Real'] - df_resultados_pasado['Predicho']

        # 2. Coeficiente de Skewness (Asimetría de errores)
        live_skew = residuales.skew()

        # 3. Coeficiente R² (Varianza explicada)        
        live_r2 = r2_score(df_resultados_pasado['Real'], df_resultados_pasado['Predicho'])
        
        # =====================================================================
        # 📊 CALIBRACIÓN DE SEMÁFOROS Y DELTAS (SIMETRÍA DE FILAS)
        # =====================================================================
        
        # --- RENGLÓN 1: DELTAS PORCENTUALES ---
        # MAPE (vs Benchmark)
        BASELINE_MAPE = 3.11
        mape_desviacion = live_mape - BASELINE_MAPE
        
        if mape_desviacion <= 0:
            color_mape = "green"
            delta_mape_texto = f"{mape_desviacion:.2f}% (Óptimo)"
        elif mape_desviacion <= 0.89: # Hasta 4% MAPE total
            color_mape = "orange"
            delta_mape_texto = f"+{mape_desviacion:.2f}% (Tolerancia)"
        else:
            color_mape = "red"
            delta_mape_texto = f"+{mape_desviacion:.2f}% (Degradación)"
            
        # Max APE (Peor escenario en porcentaje)
        if live_max_ape > 6.0:
            msg_max_ape, col_max_ape = "Desviación Crítica", "red"
        elif live_max_ape > 4.5:
            msg_max_ape, col_max_ape = "Pico de Error Alto", "orange"
        else:
            msg_max_ape, col_max_ape = "Pico Bajo Control", "green"

        # --- RENGLÓN 2: DELTAS DE MAGNITUD (MW) ---
        # MAE Promedio (Basado en tolerancia operativa de volumen)
        if live_mae > 2200:
            msg_mae, col_mae = "Desviación Volumétrica Alta", "red"
        elif live_mae > 1500:
            msg_mae, col_mae = "Margen de Tolerancia", "orange"
        else:
            msg_mae, col_mae = "Precisión Óptima", "green"
            
        # Max AE (El error más grande del día en Megavatios)
        if live_max_ae > 4500:
            msg_max_ae, col_max_ae = "Desajuste Crítico de Carga", "red"
        elif live_max_ae > 3000:
            msg_max_ae, col_max_ae = "Excursión de Error Moderada", "orange"
        else:
            msg_max_ae, col_max_ae = "Pico Absoluto Seguro", "green"

        # --- RENGLÓN 3: DELTAS DE VARIANZA Y SESGO ---
        # RMSE (Sensibilidad a errores grandes)
        # Si el RMSE se aleja mucho del MAE, significa que hubo errores aislados gigantescos
        relacion_rmse_mae = live_rmse / (live_mae if live_mae > 0 else 1)
        if relacion_rmse_mae > 1.5:
            # Caso crítico: El RMSE se disparó por un error puntual masivo
            msg_rmse, col_rmse = "Outliers Críticos (Falla Puntual)", "red"
        elif relacion_rmse_mae > 1.3:
            # Caso moderado: Pérdida de homogeneidad en los errores
            msg_rmse, col_rmse = "Presencia de Outliers", "orange"
        else:
            # Caso óptimo: Errores distribuidos normalmente cerca de 1.25
            msg_rmse, col_rmse = "Errores Homogéneos", "green"
            
        # --- CALIBRACIÓN DE SESGO MEDIO (MBE) TRICOLOR ---
        abs_mbe = abs(live_mbe)
        
        if abs_mbe > 3000:
            # 🚨 CASO CRÍTICO: Desajuste estructural masivo
            color_mbe = "red"
            msg_mbe = "Subestimación Crítica" if live_mbe > 0 else "Sobreestimación Crítica"
            
        elif abs_mbe > 1200:
            # 🔸 CASO MODERADO: Deriva o desfase estacional
            color_mbe = "orange"
            msg_mbe = "Sesgo: Subestimando Demanda" if live_mbe > 0 else "Sesgo: Sobreestimando Demanda"
            
        else:
            # ✅ CASO ÓPTIMO: El modelo está perfectamente balanceado
            color_mbe = "green"
            msg_mbe = "Alineación Óptima (Sesgo Mínimo)"


        # Semáforo Skewness (Buscamos que sea cercano a 0, error normal)
        if abs(live_skew) < 0.5:
            msg_skew, col_skew = "Distribución Simétrica", "green"
        elif live_skew >= 0.5:
            msg_skew, col_skew = "Atípicos por Subestimación", "orange"
        else:
            msg_skew, col_skew = "Atípicos por Sobreestimación", "orange"

        # Semáforo R²
        if live_r2 > 0.90:
            msg_r2, col_r2 = "Ajuste Excelente", "green"
        elif live_r2 > 0.75:
            msg_r2, col_r2 = "Capacidad Aceptable", "orange"
        else:
            msg_r2, col_r2 = "Desviación de Varianza", "red"

        # =====================================================================
        # --- RENDERIZADO DEL LAYOUT EN ESPEJO ---
        # =====================================================================
        past_col1, past_col2 = st.columns([1, 2.2])

        # BLOQUE IZQUIERDO: MATRIZ DE ERRORES 3x2 (Alineación Forzada de Deltas)
        with past_col1:
            st.markdown("#### ⚙️ Métricas de Calidad")
            st.write("")
            
            # --- RENGLÓN 1: ERRORES PORCENTUALES ---
            pr_c1, pr_c2 = st.columns(2)
            with pr_c1:
                st.metric(
                    label="📊 MAPE", 
                    value=f"{live_mape:.2f} %",
                    delta=delta_mape_texto,
                    delta_color=color_mape,
                    delta_arrow="off" # Sin flechas para el valor actual, solo color y texto de estado
                )
            with pr_c2:
                st.metric(
                    label="📈 Max APE", 
                    value=f"{live_max_ape:.2f} %",
                    delta=msg_max_ape,
                    delta_color=col_max_ape,
                    delta_arrow="off" # Sin flechas para el valor actual, solo color y texto de estado
                )
            
            st.write("")
            
            # --- RENGLÓN 2: ERRORES ABSOLUTOS (MW) ---
            r2_pc1, r2_pc2 = st.columns(2)
            with r2_pc1:
                st.metric(
                    label="🎯 MAE", 
                    value=f"{live_mae:,.0f} MW".replace(",", " "),
                    delta=msg_mae,
                    delta_color=col_mae,
                    delta_arrow="off" # Sin flechas para el valor actual, solo color y texto de estado
                )
            with r2_pc2:
                st.metric(
                    label="⚠️ Max AE", 
                    value=f"{live_max_ae:,.0f} MW".replace(",", " "),
                    delta=msg_max_ae,
                    delta_color=col_max_ae,
                    delta_arrow="off" # Sin flechas para el valor actual, solo color y texto de estado
                )
                
            st.write("")
            
            # --- RENGLÓN 3: VARIANZA Y SESGO ---
            r3_pc1, r3_pc2 = st.columns(2)
            with r3_pc1:
                st.metric(
                    label="📉 RMSE", 
                    value=f"{live_rmse:,.0f} MW".replace(",", " "),
                    delta=msg_rmse,
                    delta_color=col_rmse,
                    delta_arrow="off" # Sin flechas para el valor actual, solo color y texto de estado
                )
            with r3_pc2:
                st.metric(
                    label="⚖️ MBE", 
                    value=f"{live_mbe:,.0f} MW".replace(",", " "),
                    delta=msg_mbe,
                    delta_color=color_mbe,
                    delta_arrow="off" # Sin flechas para el valor actual, solo color y texto de estado
                )

                # --- RENGLÓN 4: ANÁLISIS ESTADÍSTICO AVANZADO ---
            r4_pc1, r4_pc2 = st.columns(2)

            with r4_pc1:
                st.metric(
                    label="🔮 R² Score", 
                    value=f"{live_r2:.3f}",
                    delta=msg_r2,
                    delta_color=col_r2,
                    delta_arrow="off"
                )
            with r4_pc2:
                st.metric(
                    label="🔄 Skewness (Residuales)", 
                    value=f"{live_skew:.2f}",
                    delta=msg_skew,
                    delta_color=col_skew,
                    delta_arrow="off"
                )

        # BLOQUE DERECHO: GRÁFICA HISTÓRICA REAL VS PREDICHO

        # Ejemplo rápido para tus gráficos antes de Plotly:
        df_plot2 = df_resultados_pasado.copy()
        # Convertimos el índice de UTC a hora de Texas y lo hacemos limpio (naive)
        df_plot2.index = df_plot2.index.tz_localize('UTC').tz_convert('US/Central').tz_localize(None)

        with past_col2:
            st.markdown("#### 📊 Desempeño Histórico (Últimas 24h)")
            
            fig_past = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_past.add_trace(
                go.Scatter(x=df_plot2.index, y=df_plot2['Real'], 
                           mode='lines+markers', name='Real EIA (MW)', line=dict(color='#00FF00', width=3)),
                secondary_y=False
            )
            fig_past.add_trace(
                go.Scatter(x=df_plot2.index, y=df_plot2['Predicho'], 
                           mode='lines+markers', name='Predicción Lgbm (MW)', line=dict(color='#FFA500', width=2, dash='dash')),
                secondary_y=False
            )
            # 🌟 CORREGIDO: Consumimos la columna correcta de temperatura histórica
            fig_past.add_trace(
                go.Scatter(x=df_plot2.index, y=df_plot2['texas_avg_temp'], 
                           mode='lines', name='Temp Real (°C)', line=dict(color='rgba(239, 83, 80, 0.5)', width=2, dash='dot')),
                secondary_y=True
            )
            
            fig_past.update_layout(
                template="plotly_dark", 
                margin=dict(l=10, r=60, t=25, b=10), 
                # 🌟 ACTUALIZADO: El eje X ahora refleja la hora de Texas de forma consistente
                xaxis_title="Fecha y Hora (Texas CT)", 
                hovermode="x unified",
                legend=dict(
                    orientation="v",
                    y=0.6,
                    x=1.05,
                    xanchor="left",
                    yanchor="top"
                ) 
            )
            fig_past.update_yaxes(title_text="Energía (MW)", secondary_y=False)
            fig_past.update_yaxes(title_text="Temperatura (°C)", secondary_y=True, showgrid=False)
            
            st.plotly_chart(fig_past, use_container_width=True)
        # =====================================================================
        # 📊 SECCIÓN: INTERPRETABILIDAD Y DIAGNÓSTICO EJECUTIVO DEL MODELO
        # =====================================================================
        st.space()
        st.markdown("---")
        st.subheader("🧠 Arquitectura y Gobernanza del Modelo LightGBM", anchor="arquitectura-modelo")
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
