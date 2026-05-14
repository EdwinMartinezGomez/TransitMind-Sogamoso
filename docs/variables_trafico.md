# Variables de Tráfico — TransitMind Sogamoso

## Variables Temporales

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| `timestamp` | datetime | — | Marca temporal de la observación |
| `hour` | int | [0, 23] | Hora del día |
| `day_of_week` | int | [0, 6] | Día de la semana (0=Lunes) |
| `is_peak_hour` | bool | — | Hora pico: 6-9am, 12-2pm, 5-8pm |

## Variables Vehiculares

| Variable | Tipo | Unidad | Rango | Descripción |
|----------|------|--------|-------|-------------|
| `vehicle_flow` | float | veh/15min | [0, 300] | Flujo vehicular |
| `heavy_vehicle_ratio` | float | — | [0, 1] | Proporción vehículos pesados |
| `motorcycle_ratio` | float | — | [0, 1] | Proporción de motocicletas |
| `avg_speed_kmh` | float | km/h | [0, 80] | Velocidad promedio |
| `congestion_level` | float | — | [0, 1] | **TARGET**: Nivel de congestión |

## Variables Contextuales

| Variable | Tipo | Categorías | Descripción |
|----------|------|-----------|-------------|
| `weather_code` | int | 0=despejado, 1=nublado, 2=lluvia, 3=niebla | Condición climática |
| `event_impact` | float | [0, 1] | Impacto de eventos locales |
| `is_market_day` | bool | — | Día de mercado campesino |

## Variables Espaciales

| Variable | Categorías |
|----------|-----------|
| `intersection_id` | carrera_11_norte, carrera_11_sur, av_castellana_entrada, av_castellana_salida, calle_14_centro_historico, acceso_morca |
