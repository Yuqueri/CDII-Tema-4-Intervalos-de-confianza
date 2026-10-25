# Intervalos de confianza con Python (Tema 4)

Resumen de la parte de programación del Tema 4: intervalos de confianza para una media o proporción, para diferencias entre dos poblaciones y mediante bootstrap. Todo el código usa `numpy`, `scipy` y `matplotlib`.

```bash
pip install numpy scipy matplotlib
```

## Contenido

1. [Qué función de SciPy usar](#1-qué-función-de-scipy-usar)
2. [Una muestra: media y proporción](#2-una-muestra-media-y-proporción)
3. [Dos muestras: diferencias y cociente de varianzas](#3-dos-muestras-diferencias-y-cociente-de-varianzas)
4. [Bootstrap](#4-bootstrap)
5. [Números pseudoaleatorios](#5-números-pseudoaleatorios)
6. [Gráficos útiles](#6-gráficos-útiles)

---

## 1. Qué función de SciPy usar

Casi todo el tema se reduce a calcular un **valor crítico** con `.ppf` (la inversa de la función de distribución acumulada) y un **error estándar**. El intervalo es siempre `estimador ± crítico · error estándar`.

| Situación | Distribución | Valor crítico (nivel 95 %, `alfa = 0.05`) |
|---|---|---|
| Media, varianza conocida | Normal | `norm.ppf(1 - alfa/2)` |
| Media, varianza desconocida | t de Student | `t.ppf(1 - alfa/2, df=n-1)` |
| Proporción | Normal | `norm.ppf(1 - alfa/2)` |
| Diferencia de medias, varianzas conocidas | Normal | `norm.ppf(1 - alfa/2)` |
| Diferencia de medias, varianzas desconocidas | t de Student | `t.ppf(1 - alfa/2, df=gl)` |
| Cociente de varianzas | F | `f.ppf(alfa/2, n1-1, n2-1)` y `f.ppf(1-alfa/2, n1-1, n2-1)` |

Funciones de `scipy.stats` que se repiten: `.pdf` (densidad), `.cdf` (área acumulada), `.ppf` (valor que deja cierta área a la izquierda) y `.rvs` (números aleatorios).

---

## 2. Una muestra: media y proporción

```python
import numpy as np
from scipy.stats import norm, t

def calcular_intervalo_confianza(muestra, nivel_confianza=0.95,
                                 varianza_conocida=False, sigma=None):
    if not isinstance(muestra, (list, np.ndarray)):
        raise TypeError("La muestra debe ser una lista o un array de NumPy")

    muestra = np.array(muestra)
    n = len(muestra)
    alfa = 1 - nivel_confianza
    es_binaria = np.isin(muestra, [0, 1]).all()

    if es_binaria:                                  # proporción
        p = np.mean(muestra)
        error = np.sqrt(p * (1 - p) / n)
        critico = norm.ppf(1 - alfa / 2)
        return (p - critico * error, p + critico * error)

    media = np.mean(muestra)                        # media
    if varianza_conocida:
        if sigma is None:
            raise ValueError("Si la varianza es conocida hay que indicar sigma")
        error = sigma / np.sqrt(n)
        critico = norm.ppf(1 - alfa / 2)
    else:
        error = np.std(muestra, ddof=1) / np.sqrt(n)
        critico = t.ppf(1 - alfa / 2, df=n - 1)
    return (media - critico * error, media + critico * error)
```

Uso:

```python
calcular_intervalo_confianza(nivel_glucosa, 0.95, True, 15)   # media, sigma conocida
calcular_intervalo_confianza(nivel_glucosa, 0.95)             # media, sigma desconocida
calcular_intervalo_confianza(antecedentes)                    # proporción (datos 0/1)
calcular_intervalo_confianza(nivel_glucosa, 0.95, True)       # ValueError: falta sigma
```

Detalles importantes:

- `np.std(x, ddof=1)` calcula el desvío **muestral** (divide por `n-1`). Sin `ddof=1` divide por `n`.
- Una proporción es la media de una muestra de ceros y unos: `np.mean(antecedentes)`.
- Con una varianza desconocida se usa la t de Student, que tiene colas más pesadas que la normal y se acerca a ella al aumentar `df`.

### Estandarizar datos

```python
def estandarizar_datos(x, mu=None, sigma=None):
    x = np.array(x)
    if mu is None:
        mu = np.mean(x)
    if sigma is None:
        sigma = np.std(x, ddof=1)
    return (x - mu) / sigma
```

### Comprobar la cobertura por simulación

Se generan muchas muestras de una población con media conocida y se cuenta en cuántas el intervalo contiene la media verdadera. Debería ser cercano al nivel de confianza.

```python
mu, sigma = 90, 15
N, n = 1000, 150
contador = 0

for _ in range(N):
    muestra = norm.rvs(loc=mu, scale=sigma, size=n)
    inf, sup = calcular_intervalo_confianza(muestra, 0.95, True, sigma)
    if inf <= mu <= sup:
        contador += 1

print(f"Porcentaje aproximado: {contador / N * 100:.2f}%")   # ≈ 95 %
```

---

## 3. Dos muestras: diferencias y cociente de varianzas

### Diferencia de medias

```python
def calcular_intervalo_confianza_diferencia(muestra1, muestra2, nivel_confianza=0.95,
                                            varianza_conocida=False, sigma1=None,
                                            sigma2=None, varianzas_iguales=False):
    for m in (muestra1, muestra2):
        if not isinstance(m, (list, np.ndarray)):
            raise TypeError("Las muestras deben ser listas o arrays de NumPy")

    x1, x2 = np.array(muestra1), np.array(muestra2)
    n1, n2 = len(x1), len(x2)
    alfa = 1 - nivel_confianza
    diferencia = np.mean(x1) - np.mean(x2)

    if varianza_conocida:
        if sigma1 is None or sigma2 is None:
            raise ValueError("Hay que indicar sigma1 y sigma2")
        error = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
        critico = norm.ppf(1 - alfa / 2)
    else:
        s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
        if varianzas_iguales:                         # varianza combinada
            gl = n1 + n2 - 2
            sp2 = ((n1 - 1) * s1 + (n2 - 1) * s2) / gl
            error = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        else:                                         # Welch (varianzas diferentes)
            error = np.sqrt(s1 / n1 + s2 / n2)
            gl = error**4 / ((s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1))
        critico = t.ppf(1 - alfa / 2, df=gl)

    return (diferencia - critico * error, diferencia + critico * error)
```

```python
calcular_intervalo_confianza_diferencia(A, B, 0.95, True, 15, 10)        # sigmas conocidas
calcular_intervalo_confianza_diferencia(A, B)                            # desconocidas, distintas
calcular_intervalo_confianza_diferencia(A, B, varianzas_iguales=True)    # desconocidas, iguales
```

Si el intervalo **no contiene el 0**, hay evidencia de que las medias son distintas.

### ¿Varianzas iguales o distintas? Cociente de varianzas

```python
from scipy.stats import f

def ic_cociente_varianzas(x1, x2, nivel_confianza=0.95):
    n1, n2 = len(x1), len(x2)
    alfa = 1 - nivel_confianza
    cociente = np.var(x1, ddof=1) / np.var(x2, ddof=1)
    f_der = f.ppf(1 - alfa / 2, n1 - 1, n2 - 1)
    f_izq = f.ppf(alfa / 2, n1 - 1, n2 - 1)
    return (cociente / f_der, cociente / f_izq)
```

Si el intervalo **contiene el 1**, es razonable suponer varianzas iguales.

### Diferencia de proporciones

```python
def ic_diferencia_proporciones(x1, x2, nivel_confianza=0.95):
    n1, n2 = len(x1), len(x2)
    p1, p2 = np.mean(x1), np.mean(x2)
    error = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = norm.ppf(1 - (1 - nivel_confianza) / 2)
    return (p1 - p2 - z * error, p1 - p2 + z * error)
```

---

## 4. Bootstrap

Cuando no se conoce la distribución del estadístico (por ejemplo, la mediana) se **remuestrea con reemplazo** la muestra original muchas veces y se calcula el estadístico en cada remuestra.

### Remuestreo con reemplazo

```python
from random import choice

def extraer_muestra_con_reposicion(datos, n):
    indices = list(range(len(datos)))
    indices_muestra = [choice(indices) for _ in range(n)]     # con reposición
    return [float(datos[i]) for i in indices_muestra]
```

Versión equivalente y más rápida con NumPy:

```python
def bootstrap(datos, estadistico=np.mean, N=1000, semilla=None):
    rng = np.random.default_rng(semilla)
    datos = np.asarray(datos)
    return np.array([
        estadistico(rng.choice(datos, size=len(datos), replace=True))
        for _ in range(N)
    ])
```

### Intervalo por percentiles

```python
def ic_bootstrap_percentiles(estadisticos, nivel_confianza=0.95):
    alfa = 1 - nivel_confianza
    inferior = np.percentile(estadisticos, 100 * alfa / 2)          # 2.5
    superior = np.percentile(estadisticos, 100 * (1 - alfa / 2))    # 97.5
    return (inferior, superior)

medianas = bootstrap(nivel_trigliceridos, np.median, N=1000)
ic_bootstrap_percentiles(medianas)
```

### Intervalo por error estándar bootstrap

```python
def ic_bootstrap_error_estandar(datos, estadisticos, nivel_confianza=0.95):
    z = norm.ppf(1 - (1 - nivel_confianza) / 2)
    error = np.std(estadisticos, ddof=1)        # desvío de los estadísticos bootstrap
    estimador = np.mean(datos)
    return (estimador - z * error, estimador + z * error)
```

El desvío estándar de las medias bootstrap debería parecerse al teórico `sigma / np.sqrt(n)`.

### Reproducibilidad

```python
import random
random.seed(42)      # con la misma semilla se obtiene siempre la misma muestra
```

---

## 5. Números pseudoaleatorios

Un computador no genera azar real: usa una fórmula determinista que parte de una **semilla**. Un generador congruencial lineal es:

```
x_{k+1} = (a · x_k + c) mod m
```

### Versión iterativa

```python
def generar_numeros_aleatorios_iterativo(x0, n, a=3, c=1, m=2**16):
    x, numeros = x0, []
    for _ in range(n):
        x = (a * x + c) % m
        numeros.append(x / m)          # dividir por m deja el número en [0, 1)
    return numeros
```

### Versión recursiva

```python
def generar_numeros_aleatorios_recursivo(x0, n, numeros=None, a=3, c=1, m=2**16):
    if numeros is None:
        numeros = []
    if n == 0:                         # caso base
        return numeros
    xn = (a * x0 + c) % m
    numeros.append(xn / m)
    return generar_numeros_aleatorios_recursivo(xn, n - 1, numeros, a, c, m)
```

Con `n = 2000` la versión recursiva falla con `RecursionError`, porque Python limita la profundidad de recursión (por defecto, 1000 llamadas). Para muchos números conviene la iterativa.

### Otras distribuciones a partir de la uniforme

```python
import time

def generar_numeros_aleatorios_uniforme(a=0, b=1, n=100, x0=None):
    if x0 is None:
        x0 = int(time.time())          # semilla basada en el reloj
    u = generar_numeros_aleatorios_iterativo(x0, n)
    return (b - a) * np.array(u) + a

def generar_numeros_aleatorios_normal(mu=0, sigma=1, n=100, x0=None):
    u = generar_numeros_aleatorios_uniforme(0, 1, n, x0)
    z = norm.ppf(u)                    # método de la transformada inversa
    return sigma * z + mu              # estandarizada → N(mu, sigma²)
```

Idea clave (transformada inversa): si `U ~ Uniforme(0, 1)`, entonces `F⁻¹(U)` tiene la distribución cuya acumulada es `F`.

---

## 6. Gráficos útiles

Histograma de una muestra con la densidad teórica encima:

```python
import matplotlib.pyplot as plt

plt.figure()
plt.hist(nivel_glucosa, bins=20, density=True, color="skyblue", edgecolor="black")

x = np.linspace(40, 140, 500)
plt.plot(x, norm.pdf(x, loc=90, scale=15), label="Densidad teórica")

plt.title("Distribución de los niveles de glucosa")
plt.xlabel("Nivel de glucosa")
plt.ylabel("Densidad")
plt.legend()
plt.show()
```

Comparar la t de Student con distintos grados de libertad:

```python
x = np.linspace(-5, 5, 500)
for gl in [1, 2, 5, 10]:
    plt.plot(x, t.pdf(x, df=gl), label=f"t, gl={gl}")
plt.plot(x, norm.pdf(x), "k--", label="Normal")
plt.legend()
plt.show()
```

Boxplot para comparar dos grupos:

```python
plt.boxplot([nivel_glucosa_A, nivel_glucosa_B], tick_labels=["Clínica A", "Clínica B"])
plt.ylabel("Nivel de glucosa (mg/dL)")
plt.show()
```

---

