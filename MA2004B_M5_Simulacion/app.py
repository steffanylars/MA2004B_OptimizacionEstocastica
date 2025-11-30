#----------------------------------MONTECARLO APP----------------------------------#

import random
import math


class Montecarlo:
    def __init__(self, num_variables=5, n_experimentos=6, criterio_sorteo=4,limite_inferior=1000,limite_superior=5000): #paneles, experimentos, fallas, li,ls,media
        self.num_variables = num_variables
        self.n_experimentos = n_experimentos
        self.criterio_sorteo = criterio_sorteo
        self.limite_inferior = limite_inferior
        self.limite_superior = limite_superior
        self.media = (limite_inferior + limite_superior) / 2

        self.crear_experimentos()
        self.sortear()
        self.metricas()

        # experimentos, satelite (promedio de falla), desviacion estandar muestral, error estandar :)
        

    def crear_experimentos(self):
        ''' 
            Crea los experimentos n veces, cada experimento tiene num_variables variables aleatorias.
            "Cuenta" el tiempo hasta que falle cada panel (num_variables) paneles
        '''

        self.experimentos=[]
        for i in range(self.n_experimentos):
            experimento_n =[]
            for j in range(self.num_variables):
                tiempo_falla_panel=random.randint(self.limite_inferior,self.limite_superior)
                experimento_n.append(tiempo_falla_panel)
            self.experimentos.append(experimento_n)

      
    def sortear(self): 
        '''
            Es la que toma el criteio_sorteo y elige a partir de cual experimento falla el sistema (4 por defecto)
            y eso lo guarda en la variable satelite
        '''
        satelite = []
        for experimento in self.experimentos:
            experimento.sort()
            satelite.append(experimento[self.criterio_sorteo-1])

        self.satelite= satelite

    def metricas(self):
        '''
            Calcula las metricas de los experimentos realizados
                promedio de tiempo de falla del sistema (satelite)
                desviacion estandar muestral
                standard error
        '''
        self.promedio_falla =sum(self.satelite)/len(self.satelite)
        self.desviacion_estandar_muestral = math.sqrt(
            sum((x - self.media)**2 for x in self.satelite) / (len(self.satelite)-1))
        
        self.standard_error = self.desviacion_estandar_muestral /math.sqrt(len(self.satelite))

    def retornar_resultados(self):
        lista = [self.satelite,round(self.promedio_falla,2), round(self.desviacion_estandar_muestral,2) ,round(self.standard_error,2)]

        return self.experimentos,lista,self.satelite

simulacion1 = Montecarlo() #los parametros finales de li,ls y media son opcionales segun yo para el area de parametrizacion pero no entendí bien la tarea jaja
#print(experimentos)


print(simulacion1.retornar_resultados())



#----------------------------------STREAMLIT APP----------------------------------#
import streamlit as st
import pandas as pd

st.title("Simulación Montecarlo")
st.write("Esta aplicación tiene como finalidad resolver el problema del tiempo de falla (MTTF) de un satelite utilizando simulación Montecarlo.")

st.write("Problema: Supongamos que tenemos un satélite, que para su funcionamiento depende  de que al menos 2 paneles solares de los 5 que tiene disponibles estén en  funcionamiento, y queremos calcular φ la vida útil esperada del satélite (el  tiempo promedio de funcionamiento hasta que falla, usualmente conocido  en la literatura como MTTF - Mean Time To Failure). Supongamos que cada panel solar tiene una vida útil que es aleatoria, y  está uniformemente distribuída en el rango [1000 hs, 5000 hs] (valor  promedio: 3000 hs). Para estimar por Monte Carlo el valor de φ, haremos n experimentos, cada  uno de los cuales consistirá en sortear el tiempo de falla de cada uno de los  paneles solares del satélite, y observar cual es el momento en el cuál han  fallado 4 de los mismos, esta es la variable aleatoria cuya esperanza es el  tiempo promedio de funcionamiento del satélite. El valor promedio de las n observaciones nos proporciona una estimación  de φ.")

st.write("### Parámetros de la simulación: ")

n = st.number_input(
    "Número de simulaciones (n):",
    min_value=1,
    value=6,
    step=1
)

paneles = st.number_input(
    "Número de paneles: ",
    min_value=1,
    value=5
)

panel_para_fallar = st.number_input(
    "Número de paneles necesarios para que deje de funcionar el satélite",
    min_value=1,
    value=4
)



st.write("Cada panel tiene una vida útil uniformemente distribuída en el rango [1000 hs, 5000 hs]. Pero puedes variar el límite si gustas.")
col1, col2 = st.columns(2)
with col1:
    limite_inferior = st.number_input(
        "Límite inferior de la vida útil (hs):",
        min_value=0,
        value=1000,
        step=100
    )
with col2:
    limite_superior = st.number_input(
        "Límite superior de la vida útil (hs):",
        min_value=limite_inferior + 1,
        value=5000,
        step=100
    )


enviar=st.button("Ejecutar Simulación")
if enviar:
    st.write("### Simulación:")
    simulacion = Montecarlo(num_variables=paneles, n_experimentos=n, criterio_sorteo=panel_para_fallar,limite_inferior=limite_inferior,limite_superior=limite_superior)
    experimentos, resultados,satelite = simulacion.retornar_resultados() # resultados son: lista = promedio falla, desviacion estandar muestral y error estandar
 
    conti=0
    for i in experimentos:
        i.append(satelite[conti]) 
        conti+=1

    df=pd.DataFrame(experimentos, columns=[f'Panel {i+1}' for i in range(paneles)]+['Satelite (xi)'])
    df.index +=1 
    st.table(df)

    for i, texto in enumerate(["Promedio de tiempo de falla del sistema (satelite):", "Desviación estándar muestral:", "Error estándar:"]):
        st.write(f"**{texto}** {resultados[i+1]} hs")
    

