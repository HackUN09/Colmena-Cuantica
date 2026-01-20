-- COLMENA-CUANTICA: Inicialización de Tesorería
-- Este script se ejecuta automáticamente al iniciar el contenedor de postgres

-- 1. Tabla de Carteras (Wallets)
-- Almacena el saldo virtual actual de cada agente
CREATE TABLE IF NOT EXISTS wallets (
    agente_id VARCHAR(50) PRIMARY KEY,
    balance DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Tabla de Transacciones (Log de Auditoría)
-- Registro inmutable de cada operación de cosecha
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    agente_id VARCHAR(50) NOT NULL,
    pnl_bruto DECIMAL(20, 8) NOT NULL,
    tax_cosecha DECIMAL(20, 8) NOT NULL,
    saldo_final DECIMAL(20, 8) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_agent FOREIGN KEY(agente_id) REFERENCES wallets(agente_id)
);

-- Índices para búsqueda rápida
CREATE INDEX IF NOT EXISTS idx_transactions_agent ON transactions(agente_id);
CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions(timestamp);
