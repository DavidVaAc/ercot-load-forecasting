# seed_backup.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("🌱 Re-generando archivos semilla con redondeo absoluto de horas...")

hoy = datetime.utcnow()
inicio_historico = hoy - timedelta(days=8)

fin_demanda = hoy
fin_clima = hoy + timedelta(days=2)

# 🛠️ Forzamos el piso de la hora (.floor('h')) para eliminar minutos y microsegundos
rango_eia = pd.date_range(start=inicio_historico, end=fin_demanda, freq='h').floor('h')
rango_clima = pd.date_range(start=inicio_historico, end=fin_clima, freq='h').floor('h')

# 1. Crear respaldo para la Demanda (EIA)
df_eia_seed = pd.DataFrame(index=rango_eia)
df_eia_seed.index.name = 'timestamp'
df_eia_seed['value'] = np.random.uniform(45000, 60000, size=len(rango_eia))

# 2. Crear respaldo para el Clima (Open-Meteo)
df_clima_seed = pd.DataFrame(index=rango_clima)
df_clima_seed.index.name = 'timestamp'

ciudades = ["houston", "dallas", "austin"]
for ciudad in ciudades:
    df_clima_seed[f"{ciudad}_temp"] = np.random.uniform(28, 38, size=len(rango_clima))
    df_clima_seed[f"{ciudad}_humidity"] = np.random.uniform(40, 70, size=len(rango_clima))
    df_clima_seed[f"{ciudad}_apparent_temp"] = df_clima_seed[f"{ciudad}_temp"] + 2
    df_clima_seed[f"{ciudad}_wind_speed"] = np.random.uniform(5, 18, size=len(rango_clima))

# 3. Guardar sobrescribiendo de forma limpia
os.makedirs('data', exist_ok=True)
df_eia_seed.to_csv('data/backup_live_eia.csv')
df_clima_seed.to_csv('data/backup_live_clima.csv')

print("✅ Archivos semilla alineados a la hora en punto.")