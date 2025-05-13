"Ejercicio 1"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del circuito
R = 1000  # Resistencia en ohmios
C = 0.001  # Capacitancia en faradios
V_fuente = 5  # Voltaje de la fuente en volts

# Definición de la ecuación diferencial
def f(t, V):
    return (1 / (R * C)) * (V_fuente - V)

# Condiciones iniciales
t0 = 0  # Tiempo inicial
V0 = 0  # Voltaje inicial
tf = 5  # Tiempo final
n = 20  # Número de pasos
h = (tf - t0) / n  # Paso

# Inicialización de listas para almacenar resultados
t_vals = [t0]
V_vals = [V0]

# Método de Euler
t = t0
V = V0
for i in range(n):
    V = V + h * f(t, V)  # Actualización usando Euler
    t = t + h
    t_vals.append(t)
    V_vals.append(V)

# Comparación con la solución analítica
V_analitica = [V_fuente * (1 - np.exp(-t / (R * C))) for t in t_vals]

# Guardar resultados en archivo CSV
data = {
    "t": t_vals,
    "V_aproximado": V_vals,
    "V_analitico": V_analitica
}
df = pd.DataFrame(data)
csv_path = "circuito_rc_euler.csv"
df.to_csv(csv_path, index=False)

# Graficar la solución aproximada y analítica
plt.figure(figsize=(8, 5))
plt.plot(t_vals, V_vals, 'o-', label='Solución aproximada (Euler)', color='blue')
plt.plot(t_vals, V_analitica, '-', label='Solución analítica', color='red')
plt.title('Modelo de Voltaje en Circuito RC (Método de Euler)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()
plt.savefig("circuito_rc_solucion.png")
plt.show()

"Ejercicio 2"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del problema
g = 9.81  # Gravedad
m = 2  # Masa
k = 0.5  # Coeficiente de fricción
v0 = 0  # Velocidad inicial
t0 = 0  # Tiempo inicial
tf = 10  # Tiempo final
n = 50  # Número de pasos
h = (tf - t0) / n  # Tamaño del paso

# Definición de la ecuación diferencial
def f(t, v):
    return g - (k / m) * v

# Inicialización de listas para almacenar resultados
t_vals = [t0]
v_vals = [v0]

# Método de Euler
t = t0
v = v0
for i in range(n):
    v = v + h * f(t, v)  # Paso de Euler
    t = t + h
    t_vals.append(t)
    v_vals.append(v)

# Solución exacta
v_exacta = [(m * g / k) * (1 - np.exp(- (k / m) * t)) for t in t_vals]

# Guardar resultados en CSV
data = {
    "t": t_vals,
    "v_aproximado": v_vals,
    "v_exacta": v_exacta
}
df = pd.DataFrame(data)
df.to_csv("caida_con_friccion.csv", index=False)

# Graficar resultados
plt.figure(figsize=(8, 5))
plt.plot(t_vals, v_vals, 'o-', label="Euler", color="blue")
plt.plot(t_vals, v_exacta, '-', label="Solución exacta", color="red")
plt.title("Caída con fricción (Método de Euler)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.grid(True)
plt.legend()
plt.savefig("caida_con_friccion.png")
plt.show()

"Ejercicio 3"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del problema
T0 = 90  # Temperatura inicial en °C
T_amb = 25  # Temperatura ambiente en °C
k = 0.07  # Constante de enfriamiento
t0 = 0  # Tiempo inicial en minutos
tf = 30  # Tiempo final en minutos
n = 30  # Número de pasos
h = (tf - t0) / n  # Tamaño de paso

# Definición de la ecuación diferencial
def f(t, T):
    return -k * (T - T_amb)

# Inicialización de listas
t_vals = [t0]
T_vals = [T0]

# Método de Euler
t = t0
T = T0
for i in range(n):
    T = T + h * f(t, T)  # Paso de Euler
    t = t + h
    t_vals.append(t)
    T_vals.append(T)

# Solución exacta
T_exacta = [T_amb + (T0 - T_amb) * np.exp(-k * t) for t in t_vals]

# Guardar resultados en CSV
data = {
    "t": t_vals,
    "T_aproximado": T_vals,
    "T_exacto": T_exacta
}
df = pd.DataFrame(data)
df.to_csv("enfriamiento_newton.csv", index=False)

# Graficar resultados
plt.figure(figsize=(8, 5))
plt.plot(t_vals, T_vals, 'o-', label="Euler", color="blue")
plt.plot(t_vals, T_exacta, '-', label="Solución exacta", color="red")
plt.title("Ley de Enfriamiento de Newton (Método de Euler)")
plt.xlabel("Tiempo (min)")
plt.ylabel("Temperatura (°C)")
plt.grid(True)
plt.legend()
plt.savefig("enfriamiento_newton.png")
plt.show()
