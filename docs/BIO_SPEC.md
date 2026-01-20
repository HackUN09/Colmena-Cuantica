# PROYECTO COLMENA CUÁNTICA: ESPECIFICACIÓN BIOLÓGICA (V1.0)

> "No es el más fuerte el que sobrevive, sino el que mejor se adapta al cambio de frecuencia."

## 1. El Genoma del Agente ($G$)

Cada agente en la colmena se define por un conjunto de genes inmutables durante su vida (Generación) y un fenotipo mutable (pesos neuronales).

$$
G_i = [T_i, \rho_i, \lambda_i]
$$

Donde:
*   **$T_i$ (Dilatación Temporal)**: Entero $\in [10, 60]$. Define la ventana de "percepción subjetiva" (Meso-Layer). El agente solo despierta y decide cuando $t \mod T_i = 0$.
*   **$\rho_i$ (Aversión al Riesgo)**: Escalar $\in [0, 1]$. Modula la función de recompensa interna (no implementado en V1.0, reservado para V2.0).
*   **$\lambda_i$ (Tasa de Aprendizaje)**: Hiperparámetro del optimizador Adam específico para este agente.

## 2. Dinámica de Población (The Reaper)

La evolución se rige por un algoritmo genético de estado estacionario modificado (Steady-State GA).

### 2.1 Función de Aptitud (Fitness)
La métrica de supervivencia $F(i)$ se basa estrictamente en el PnL Realizado ajustado por la inacción (penalización por "zombie").

$$
F(i) = \text{PnL}_i - \beta \cdot \mathbb{I}(\text{trades}_i = 0)
$$

### 2.2 Selección (Truncation Selection)
Al final de cada Generación (Época):
1.  Se ordena la población $\mathcal{P}$ según $F(i)$.
2.  **Culling (Muerte)**: Se elimina al $20\%$ inferior ($\mathcal{P}_{weak}$).
3.  **Elite (Supervivencia)**: Se preserva al $10\%$ superior ($\mathcal{P}_{elite}$).

### 2.3 Reproducción (Crossover Neuronal)
Los huecos dejados por $\mathcal{P}_{weak}$ son llenados por hijos de $\mathcal{P}_{elite}$.
Sea $\theta_A$ y $\theta_B$ los tensores de pesos de dos padres elite. El hijo $\theta_{child}$ hereda:

$$
\theta_{child} = \alpha \theta_A + (1-\alpha) \theta_B + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

*   **Average Crossover**: $\alpha = 0.5$.
*   **Mutación**: $\epsilon$ es ruido gaussiano para evitar mínimos locales.

## 3. Inteligencia de Enjambre (Swarm Consensus)

Aunque los agentes son egoístas, la "decisión de la colmena" emerge de la agregación de sus vectores de acción.

Sea $a_i \in \mathbb{R}^{11}$ el vector de pesos de portafolio propuesto por el agente $i$. La acción global del sistema no es un promedio simple, sino una suma ponderada por el capital virtual ($v_i$) de cada agente.

$$
\mathbf{A}_{global} = \frac{\sum v_i \cdot a_i}{\sum v_i}
$$

Esto implementa una **Meritocracia Ponderada**: Los agentes ricos (exitosos) tienen más voto en la dirección del mercado que los agentes pobres.

---
**Protocolo**: Bio-Spectral Resonance V1.0
**Clasificación**: Biological Computing Class A
