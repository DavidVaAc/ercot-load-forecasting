# ⚡ Sistema Predictivo de Carga Eléctrica: ERCOT (Texas)

> Pipeline de MLOps de alta disponibilidad y modelo autoregresivo-termodinámico para el pronóstico de demanda energética horaria en la red eléctrica de Texas.

[![Streamlit App](https://static.streamlit.io/badge-svg.svg)]([LINK_AL_DASHBOARD])
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-LightGBM-ff69b4)](https://lightgbm.readthedocs.io/)

---

## 📈 Resumen del Proyecto

Este sistema orquesta flujos de datos asíncronos en tiempo real para predecir la demanda de energía de las **próximas 24 horas** en el Operador de Red de Texas (ERCOT). Dado que Texas opera como una "isla energética" aislada de las interconexiones nacionales de Estados Unidos, la precisión en el despacho de carga es un factor crítico de estabilidad de infraestructura y seguridad pública.

El motor de inferencia combina variables meteorológicas granulares de los tres nodos de consumo más importantes (**Houston, Dallas y Austin**) junto con la inercia histórica de la red, procesados mediante un algoritmo de Gradient Boosting optimizado para series temporales.

---

## 🚀 Arquitectura y Resiliencia de Datos (Failover en Cascada)

Para garantizar un Service Level Agreement (**SLA**) continuo en entornos de producción, el backend implementa una arquitectura redundante de extracción de datos climáticos e históricos estructurada en tres niveles jerárquicos:

| Nivel | Componente | Tipo | Función |
| :--- | :--- | :--- | :--- |
| **Nivel 1** | **Open-Meteo API** | Fuente Principal | Ingesta horaria en UTC sin límite de créditos de ráfaga. |
| **Nivel 2** | **Visual Crossing API** | Respaldo Comercial | Activación automática en caso de interrupción en Nivel 1. Sincronización absoluta en tiempo Zulu (`Z`). |
| **Nivel 3** | **Semilla Local (CSV)** | Contingencia Offline | Relleno e inferencia segura basada en snapshots locales si toda la conectividad externa falla. |

El pipeline cuenta además con un mecanismo de **filtrado determinista** basado en la última marca de tiempo real reportada por la EIA (Energy Information Administration), eliminando solapamientos temporales y blindando el modelo contra valores nulos mediante técnicas automáticas de *Forward-fill* y *Backward-fill*.

---

## 🖥️ Interfaz Operativa y Arquitectura del Dashboard

La aplicación de producción está desplegada en la nube y se divide en tres secciones estratégicas de monitoreo:

* 📊 **[Streamlit Dashboard]([LINK_AL_DASHBOARD])**

### 1. Panel de Control Principal e Inferencia de Futuro
Presenta la sincronización maestra del sistema mediante un reloj digital UTC nativo en JavaScript. Despliega tarjetas métricas que evalúan el estado térmico actual del estado mediante un termómetro dinámico multicolor y el pico de carga proyectado.

![Panel Principal e Inferencia a Futuro](images/dashb_panel.png)
*Figura 1: Cabecera operativa, métricas termodinámicas en tiempo real y curva sinusoidal de proyección de carga de las próximas 24 horas.*

### 2. Control de Calidad de Inferencia (MLOps Monitoring)
Evalúa de forma retroactiva las últimas 24 horas de operación, contrastando la curva real contra la predicción del modelo. Implementa un monitor de estabilidad del **MAPE** (Mean Absolute Percentage Error) que activa alertas visuales preventivas en color **naranja** (Margen de Tolerancia) o **rojo** (Degradación Crítica) si el error se desvía de la línea base.

![Control de Calidad del Modelo](images/dashb_quality.png)
*Figura 2: Monitoreo en vivo de métricas de error (MAPE, MAE, RMSE) y gráfica de desfase entre la demanda real de la EIA y la predicción.*

### 3. Gobernanza del Modelo e Interpretabilidad
Desglosa la lógica interna de toma de decisiones del árbol del LightGBM para eliminar el efecto de "caja negra", facilitando auditorías técnicas y presentaciones ejecutivas.

![Gobernanza del Modelo](images/dashb_modelarq.png)
*Figura 3: Gráfico de dona con la contribución relativa por categoría macro y gráfico de barras horizontal detallando el peso por split de las 27 características.*

<div align="center">
  <img src="images/shap.png" width="45%" />
</div>
*Figura 4: Resumen de valores SHAP para las 20 variables más importantes, mostrando su impacto positivo o negativo en la predicción de demanda.*

---

## 🔬 Análisis Estadístico & Ciencia de Datos (R&D)

El proceso de investigación y modelado se encuentra documentado exhaustivamente en el Jupyter Notebook Oficial.El algoritmo fue entrenado con datos históricos completos de los años **2022 a 2024 inclusive**, y validado ante un **año ciego de prueba (2025)**, obteniendo un sobresaliente **MAPE base de 3.11%**.

* 📓 **[Notebook de Investigación y Modelado](<|LINK_AL_NOTEBOOK|>)**

### Hallazgos Clave de Investigación:
* **La Curva Térmica en U:** Al analizar la relación entre la demanda y la temperatura promedio de Texas, se identificó una respuesta parabólica no lineal. El sistema basal de la red se estresa significativamente por debajo de los **12°C** (encendido de calefacción eléctrica residencial) y de forma crítica por encima de los **34°C** (operación continua de compresores HVAC).
* **Derivadas Térmicas:** La ingeniería de características demostró que la variable `temp_delta_24h` (la tasa de cambio de temperatura respecto al día anterior) es el predictor con mayor cantidad de divisiones en los árboles de decisión, capturando la inercia física del choque térmico en las estructuras urbanas.

<div align="center">
  <img src="images/u_graph.png" width="45%" />
</div>
*Figura 5: Curva parabólica en U empírica del consumo eléctrico de Texas contra variables térmicas.*

---

## 🛠️ Tecnologías Utilizadas

* **Modelado:** LightGBM, Scikit-Learn
* **Procesamiento de Datos:** Pandas, NumPy
* **Visualización:** Plotly Graph Objects, Matplotlib, Seaborn
* **Infraestructura de Dashboard:** Streamlit, Streamlit Components (HTML/JS)
* **Ingesta de Datos:** Requests API, REST Gateways (EIA & Open-Meteo)

---

## 🧠 Limitaciones del Modelo & Siguientes Pasos (Roadmap)

Todo sistema de producción real opera bajo restricciones. Para mantener la transparencia y mejora continua del pipeline, se identifican los siguientes puntos:

### Limitaciones Actuales:
1. **Dependencia de Clima Puntual:** El modelo utiliza los centros meteorológicos de tres ciudades principales como *proxy* del estado. Frentes climáticos atípicos en zonas rurales de alta densidad industrial podrían generar ligeros desvíos.
2. **Inferencia Determinista:** El modelo genera una predicción puntual (*point forecast*). En escenarios de crisis energética, la red se beneficia más de predicciones probabilísticas (bandas de confianza).

### Siguientes Pasos (Roadmap Técnico):
* **Inclusión de Generación Renovable:** Integrar las curvas de capacidad de generación eólica y solar en tiempo real de Texas como variables exógenas para predecir la demanda neta.
* **Despliegue de Reentrenamiento Automatizado (CI/CD):** Programar un script en GitHub Actions que ejecute un reentrenamiento mensual automático si el monitor del MAPE en vivo se mantiene en terreno rojo durante más de 72 horas consecutivas.

---

## ⚙️ Instalación y Reproducción Local

Si deseas ejecutar este dashboard de forma local para auditorías o desarrollo, sigue estos pasos en tu terminal:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
   cd TU_REPOSITORIO
    ```
2. **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Configurar credenciales locales:**
Crea una carpeta llamada .streamlit/ y dentro de ella un archivo secrets.toml. Agrega tus API Keys correspondientes:
```toml
EIA_API_KEY = "tu_clave_de_la_eia_aqui"
VISUAL_CROSSING_KEY = "tu_clave_de_visual_crossing_aqui"
```
4. **Ejecutar el dashboard:**
```bash
streamlit run app.py
```
---

## 📁 Estructura del Repositorio

La arquitectura del proyecto está organizada de forma modular para separar la etapa de investigación (*R&D*) del despliegue operativo en producción (*Deployment*):

```text
ercot-load-forecasting/
├── 📁 .streamlit/
│   └── secrets.toml          # Configuración y llaves de API locales (Excluido en Git)
├── 📁 data/
│   ├── modelo_final_ercot_lgb.json  # Binario serializado del modelo LightGBM
│   ├── backup_live_clima.csv      # Snapshot de contingencia para inferencia offline
│   ├── backup_live_eia.csv        # Snapshot de contingencia para inferencia offline
│   ├── ercot_load_2022_2025_static.csv  # Dataset histórico completo para entrenamiento y análisis
│   └── texas_weather_2022_2025_static.csv  # Dataset histórico completo para entrenamiento y análisis
├── 📁 images/
│   ├── dashb_panel.png  # Capturas de pantalla para documentación
│   ├── dashb_modelarq.png
│   ├── dashb_quality.png
│   ├── u_graph.png
│   └── shap.png
├── 📁 notebooks/
│   ├── electricity_demand.ipynb  # Fase de R&D, entrenamiento y análisis SHAP
│   ├── EIA_API.ipynb                # Exploración y validación de la API de la EIA
│   └── OPEN_METEO_API.ipynb          # Exploración y validación de la API de Open-Meteo
├── .gitignore                # Archivos y credenciales ocultas para control de versiones
├── app.py                    # Código fuente de la interfaz operativa en Streamlit
├── README.md                 # Documentación técnica principal del sistema
├── requirements.txt          # Lista de dependencias y librerías para la nube
├── LICENSE                    # Licencia de uso y distribución del proyecto
├── Dockerfile                   # Configuración para contenerización (opcional)
└── seed_backup.py
```

## 👨‍💻 Autor

Desarrollado por David Valle Físico y Científico de Datos (UNAM / TripleTen Data Science)

* 💼 [Portafolio](https://davidvaac.github.io/DavidVaAc/#)
* 🌐 [LinkedIn](https://linkedin.com/in/david-fernando-valle-acosta)
* 📋 [Curriculum](https://drive.google.com/file/d/1epmNOV5wLOiH2na0B_kiDaaevGUPrUdF/view?usp=sharing)
* ✉️ [Email](mailto:davidfervalle@gmail.com)