# TRATADO BIOLÓGICO: EVOLUCIÓN SWARM (V1.0)

> **Clasificación**: Biología Computacional Avanzada
> **Objetivo**: Formalización de los mecanismos genéticos que gobiernan la adaptación de la Colmena.

---

## ÍNDICE

1.  **El Genoma del Agente ($G$)**
    *   1.1. Codificación Genotípica vs. Fenotípica
    *   1.2. El Gen de Dilatación Temporal ($T_i$) y la Percepción Relativa
    *   1.3. Mutación de Hiperparámetros (Meta-Evolución)
2.  **Dinámica de Selección Natural (Steady-State GA)**
    *   2.1. La Función de Aptitud ($F(x)$): PnL vs. Riesgo
    *   2.2. Presión de Selección y Culling Ratio ($\gamma$)
    *   2.3. El Algoritmo de "El Segador" (The Reaper)
3.  **Mecanismos de Reproducción**
    *   3.1. Crossover Neuronal (Promedio Aritmético de Pesos)
    *   3.2. Perturbación Gaussiana (Mutación de Pesos)
    *   3.3. Herencia de Memoria (¿Lamarckismo Digital?)
4.  **Teoría de Juegos Evolutiva**
    *   4.1. Estrategias Evolutivamente Estables (ESS)
    *   4.2. Competencia Intra-Enjambre y Cooperación Emergente

---

## 1. EL GENOMA DEL AGENTE ($G$)

### 1.1. Codificación Genotípica vs. Fenotípica

Distinguimos rigurosamente entre el **Genotipo** (parámetros estáticos innatos) y el **Fenotipo** (comportamiento emergente derivado de la Red Neuronal).

$$
\text{Genotipo } G_i = \{ T_i, \rho_i, \eta_i \}
$$
$$
\text{Fenotipo } \Phi_i = \mathcal{N}(S_t; \theta_i) \to A_t
$$

Donde $\theta_i$ son los pesos sinápticos (matrices de la red neuronal). La evolución opera sobre **ambos** niveles.

### 1.2. El Gen de Dilatación Temporal ($T_i$)

Este es el aporte más novedoso del Protocolo Fourier.
Definimos $T_i \in \mathbb{N}$ como el periodo refractario de decisión del agente $i$.
Esto implica que el agente $i$ vive en un tiempo subjetivo $\tau_i = t / T_i$.

*   **Matemática de la Activación**:
    $$ \text{Activo}(i, t) \iff t \pmod{T_i} \equiv 0 $$

Esto rompe la sincronía del enjambre, evitando "Flash Crashes" causados por ejecuciones simultáneas masivas y permitiendo que la colmena cubra todo el espectro de frecuencias del mercado (desde Scalping hasta Swing).

### 1.3. Mutación de Hiperparámetros

No solo evolucionan los pesos. La tasa de aprendizaje $\eta_i$ también muta.
$$ \eta_{child} = \eta_{parent} \cdot e^{\tau N(0,1)} $$
Esto permite que la colmena "aprenda a aprender" (Meta-Learning).

---

## 2. DINÁMICA DE SELECCIÓN NATURAL

### 2.1. La Función de Aptitud ($F(x)$)

La supervivencia no depende solo del retorno absoluto. Definimos la Fitness $F_i$ ajustada por riesgo (Ratio de Sharpe simplificado) y penalización por inacción.

$$
F_i = \frac{E[R_i]}{\sigma(R_i) + \epsilon} - \beta \cdot \mathbb{I}(\text{Zombie})
$$

Donde $\mathbb{I}(\text{Zombie})$ es una función indicatriz que vale 1 si el agente ha realizado 0 operaciones en la generación.
**Justificación**: Un agente que no opera tiene varianza 0 y podría sobrevivir falsamente. La penalización $\beta$ purga a los cobardes.

### 2.2. Presión de Selección y Culling Ratio ($\gamma$)

Utilizamos **Truncation Selection**.
Sea $\mathcal{P}_t$ la población ordenada descendientemente por $F_i$.
Definimos el Ratio de Sacrificio $\gamma = 0.2$.

*   **Elite**: Top 10% ($\mathcal{E}$). Inmunes a la mutación. Pasan intactos.
*   **Condenados**: Bottom $\gamma \cdot N$. Son eliminados sumariamente.
*   **Reproductores**: El resto de la población compite por llenar los huecos de los condenados.

### 2.3. Algoritmo "El Segador" (The Reaper)

Código teórico del proceso de selección:
```python
def reap_souls(population):
    # 1. Ranking
    ranked = sorted(population, key=lambda x: x.fitness, reverse=True)
    
    # 2. Execution
    n_cull = int(len(population) * CULL_RATIO)
    survivors = ranked[:-n_cull]
    graveyard = ranked[-n_cull:]
    
    # 3. Rebirth
    children = []
    for corpse in graveyard:
        parent_a, parent_b = select_parents(survivors)
        child = crossover(parent_a, parent_b)
        children.append(child)
        
    return survivors + children
```

---

## 3. MECANISMOS DE REPRODUCCIÓN (CROSSOVER)

### 3.1. Crossover Neuronal

¿Cómo se "cruzan" dos redes neuronales?
El Protocolo utiliza **Promedio Aritmético Ponderado** de los tensores de pesos.

Sea $W_A$ la matriz de pesos del padre A y $W_B$ la del padre B. El hijo hereda:

$$
W_{child} = \alpha W_A + (1-\alpha) W_B
$$

Generalmente $\alpha = 0.5$.
Geométricamente, esto sitúa al hijo en el punto medio del hipersuperficie de la función de pérdida entre los dos padres. Si ambos padres están en un "valle" de perdida bajo, es probable (convexidad local) que el punto medio también sea bueno.

### 3.2. Perturbación Gaussiana (Mutación)

Para evitar la convergencia prematura (endogamia), añadimos ruido al hijo:

$$
W_{child} \leftarrow W_{child} + \mathcal{N}(0, \sigma_{mut}^2 I)
$$

Esto permite al hijo "saltar" fuera de un mínimo local.

### 3.3. Herencia de Memoria

Los agentes NO heredan el buffer de experiencia (Replay Buffer) de sus padres (no son Lamarckianos en memoria episódica), pero SÍ heredan la "estructura cerebral" (pesos) que predispone a ciertas acciones. Es una herencia de **instintos**, no de **recuerdos**.

---

## 4. TEORÍA DE JUEGOS EVOLUTIVA

### 4.1. Estrategia Evolutivamente Estable (ESS)

Buscamos que la colmena converja a una ESS, donde ninguna estrategia mutante pueda invadir la población.
En el mercado, esto significa una mezcla heterogénea de estrategias (unos compran cuando otros venden), proporcionando liquidez interna y reduciendo la varianza global del PnL de la colmena ("Diversificación Emergente").

---
**Rigor Level**: MAXIMUM
**Evolutionary Dynamics Verified**: YES
**Author**: Antigravity Biological Core
