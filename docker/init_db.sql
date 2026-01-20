-- COLMENA-CUÁNTICA // ESQUEMA DE PERSISTENCIA FINANCIERA
-- Persistencia de balances virtuales y auditoría de cosecha

CREATE TABLE IF NOT EXISTS ledger_agentes (
    agente_id VARCHAR(50) PRIMARY KEY,
    balance_virtual DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    beneficio_acumulado DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    cosecha_total DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    ultima_operacion TIMESTAMP WITH TIME ZONE,
    estado_agente VARCHAR(20) DEFAULT 'ACTIVO' -- ACTIVO, MARGIN_CALL, HIBERNACION
);

CREATE TABLE IF NOT EXISTS historial_operaciones (
    operacion_id SERIAL PRIMARY KEY,
    agente_id VARCHAR(50) REFERENCES ledger_agentes(agente_id),
    asset VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- BUY, SELL
    monto_nominal DECIMAL(18, 8) NOT NULL,
    pnl_realizado DECIMAL(18, 8) NOT NULL,
    monto_cosechado DECIMAL(18, 8) NOT NULL, -- El 'Impuesto' de la Colmena
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tesoreria_maestra (
    id SERIAL PRIMARY KEY,
    fondo_reserva_usdc DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    total_capital_enjambre DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    ultima_actualizacion TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
