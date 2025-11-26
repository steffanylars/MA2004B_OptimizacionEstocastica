**Método Congruencial Lineal**

Backend en C++ · Frontend en Streamlit (Python)

Este proyecto implementa el Método Congruencial Lineal (MCL) para la generación de números pseudoaleatorios.
La lógica del algoritmo está realizada en C++ (backend), mientras que la interfaz gráfica y visualización de resultados está hecha en Streamlit (Python).

El usuario puede ingresar parámetros, generar secuencias, visualizar la gráfica y descargar un archivo CSV con los resultados.


**Explicación del Método Congruencial Lineal**

El MCL genera una secuencia mediante la ecuación:

*X_{n+1} = (a X_n + c) \mod m*

donde:
	•	m = módulo
	•	a = multiplicador
	•	c = incremento
	•	X_0 = semilla
	•	X_{n+1} = siguiente valor pseudoaleatorio

La salida normalizada se obtiene como:

*U_n = \frac{X_n}{m-1}*

Este proyecto también incluye un modo no aleatorio, donde:

*X_{n+1} = (X_n + 1) \mod m*
