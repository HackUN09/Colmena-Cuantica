# Configuración de Workflows n8n

## Workflow 01: Data_Pump
- **Trigger**: Cron (cada 5 minutos)
- **Acción 1**: Nodo HTTP Request a Binance API (`GET /api/v3/klines`) para 100 tickers.
- **Acción 2**: Nodo HTTP Request a CryptoPanic API para noticias en tiempo real.
- **Nodo final**: HTTP POST a `http://isaac-sim:8000/process_inference` con el JSON de datos.

## Workflow 02: Inferencia & Decisión
*(Manejado internamente por el orquestador Python disparado por el Workflow 01)*

## Workflow 03: Order_Execution
- **Trigger**: WebHook (recibe el JSON de carteras óptimas del orquestador).
- **Acción**: Nodo 'Execute Command' para llamar al script de ejecución de CCXT con los pesos validados.
- **Notificación**: Nodo Discord/Slack con el reporte del rebalanceo.

> [!NOTE]
> Los archivos JSON de los flujos se deben importar en la UI de n8n desde la carpeta `n8n/workflows/`.
