'''
    Steffany Mishell Lara Muy
    A00838589
    Optimización Estocástica
'''
# Importar Librerías
import streamlit as st
import random
import math
import pandas

# Comienza el código
class Montecarlo():
    def __init__(self, n_experimentos, a, b):
        '''
            Esta función inicializa los valores necesarios para la simulación de Montecarlo dentro de la función. Así todas las definiciones tienen acceso a ella.
            También crea las listas necesarias para almacenar los valores aleatorios, las alturas y las áreas.
        '''
        self.n_experimentos = n_experimentos
        self.a = a
        self.b = b
        self.valores_aleatorios = []
        self.alturas = []
        self.areas = []
        self.integral_estimacion = 0
        self.generar_valores_aleatorios()
        self.calcular_integral()
    
    def generar_valores_aleatorios(self):
        '''
            Genera n valores aleatorios en el rango [a, b] y calcula las alturas f(x) correspondientes.
        '''
        
        for i in range(self.n_experimentos):
            x = random.uniform(self.a, self.b)
            # f(x) = (2/π) * 1/(e^x + e^-x)
            y = (2 / math.pi) * (1 / (math.exp(x) + math.exp(-x)))
            self.valores_aleatorios.append(x)
            self.alturas.append(y)
            # El área individual no es necesaria guardarla, pero si quieres mostrarla:
            self.areas.append((self.b - self.a) * y)
    
    def calcular_integral(self):
        '''
            Calcula la estimación de la integral usando Montecarlo:
            Integral ≈ (b-a)/n * Σf(xi)
        '''
        suma_alturas = sum(self.alturas)
        self.integral_estimacion = ((self.b - self.a) / self.n_experimentos) * suma_alturas
    
    def resultados(self):
        '''
        Muestra los resultados obtenidos: 
            valores aleatorios generados
            las alturas
            las áreas 
            estimación en la integral.
        '''
        df = pandas.DataFrame({
            'Valores Aleatorios (x)': self.valores_aleatorios,
            'Alturas (f(x))': self.alturas,
            'Áreas': self.areas
        })
        return df, self.integral_estimacion

# ------------------------------------------------------------------Interfaz gráfica
st.title("Algoritmo de Montecarlo")
st.write("### Por: Steffany Mishell Lara Muy (A00838589)")
st.write("---")
st.write("La función seleccionada fue f(x) = 1 / (e^x + e^-x). Se utilizará el método de Montecarlo para estimar la integral definida de esta función en un rango [a, b] especificado por el usuario. La integral es  en el rango [1,b] de 2/pi * f(x) dx.")
st.write("#### Parámetros de la simulación: ")

n = st.number_input(
    "Número de simulaciones (n):",
    min_value=1,
    value=100,
    step=1
)

st.write("Rango de la integral [a,b]")

col1, col2 = st.columns(2)
with col1:
    a = st.number_input(
        "a:",
        value=-6,
        step=1
    )
with col2:
    b = st.number_input(
        "b:",
        value=6,
        step=1
    )

# Validación de que b > a
if b <= a:
    st.error("Error: b debe ser mayor que a")
else:
    enviar = st.button("Ejecutar Simulación")
    
    if enviar: #si el botón es presionado se calcula la simulación :)
        st.write("### Resultados de la Simulación:")
        simulacion = Montecarlo(n_experimentos=n, a=a, b=b) #acá crea un objeto que por supuesto, instancia la clase
        tablita, estimacion_integral = simulacion.resultados() #acá obtiene los resultados
        tablita.index += 1 #esto es para que la tabla comience en experimento 1 y no 0
        
        st.write(f"### Estimación de la integral: {round(estimacion_integral, 6)}") #muestro la estimación de la integral con 6 decimales
        
        st.write("---")
        st.write("### Tabla de valores generados:") #muestro la tabla de los valores generados.
        st.dataframe(tablita)