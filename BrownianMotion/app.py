"""
Simulador de Movimiento Browniano - Streamlit App
Ma2004b - Optimización Estocástica - ITESM
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Optional
import io

# ============================================================================
# CLASES DEL SIMULADOR
# ============================================================================

@dataclass
class SimulationResult:
    """Clase que almacena los resultados de una simulación de Movimiento Browniano"""
    dimensions: int
    steps: int
    repetitions: int
    max_distances: List[float] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    @property
    def average_max_distance(self) -> float:
        return np.mean(self.max_distances) if self.max_distances else 0.0
    
    @property
    def overall_max_distance(self) -> float:
        return np.max(self.max_distances) if self.max_distances else 0.0
    
    @property
    def overall_min_distance(self) -> float:
        return np.min(self.max_distances) if self.max_distances else 0.0
    
    @property
    def standard_deviation(self) -> float:
        return np.std(self.max_distances, ddof=1) if len(self.max_distances) > 1 else 0.0


class BrownianMotion:
    """Clase que implementa la simulación del Movimiento Browniano"""
    
    def __init__(self, dimensions: int, steps: int, seed: Optional[int] = None):
        if dimensions < 1:
            raise ValueError("Las dimensiones deben ser al menos 1")
        if steps < 1:
            raise ValueError("Los pasos deben ser al menos 1")
        
        self.dimensions = dimensions
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros(dimensions)
    
    def reset(self):
        """Reinicia la posición de la partícula al origen"""
        self.position = np.zeros(self.dimensions)
    
    def get_distance_from_origin(self) -> float:
        """Calcula la distancia Euclidiana desde el origen"""
        return np.sqrt(np.sum(self.position ** 2))
    
    def step(self):
        """Ejecuta un paso del movimiento Browniano"""
        dimension = self.rng.integers(0, self.dimensions)
        direction = 1 if self.rng.random() >= 0.5 else -1
        self.position[dimension] += direction
    
    def execute_walk(self) -> float:
        """Ejecuta una caminata completa y retorna la distancia máxima alcanzada"""
        self.reset()
        max_distance = 0.0
        
        for _ in range(self.steps):
            self.step()
            current_distance = self.get_distance_from_origin()
            if current_distance > max_distance:
                max_distance = current_distance
        
        return max_distance
    
    def execute_walk_with_trajectory(self) -> tuple:
        """Ejecuta una caminata y retorna la trayectoria completa"""
        self.reset()
        trajectory = [self.position.copy()]
        distances = [0.0]
        max_distance = 0.0
        
        for _ in range(self.steps):
            self.step()
            trajectory.append(self.position.copy())
            current_distance = self.get_distance_from_origin()
            distances.append(current_distance)
            if current_distance > max_distance:
                max_distance = current_distance
        
        return np.array(trajectory), np.array(distances), max_distance


class SimulationEngine:
    """Motor de simulación que ejecuta múltiples experimentos"""
    
    @staticmethod
    def run_simulation(dimensions: int, steps: int, repetitions: int, 
                       progress_callback=None) -> SimulationResult:
        """Ejecuta una simulación con los parámetros especificados"""
        result = SimulationResult(
            dimensions=dimensions,
            steps=steps,
            repetitions=repetitions
        )
        
        start_time = time.perf_counter()
        brownian = BrownianMotion(dimensions, steps)
        
        for i in range(repetitions):
            max_distance = brownian.execute_walk()
            result.max_distances.append(max_distance)
            if progress_callback and (i + 1) % max(1, repetitions // 100) == 0:
                progress_callback((i + 1) / repetitions)
        
        end_time = time.perf_counter()
        result.execution_time_ms = (end_time - start_time) * 1000
        
        return result
    
    @staticmethod
    def run_full_battery(progress_callback=None) -> List[SimulationResult]:
        """Ejecuta una batería completa de simulaciones"""
        results = []
        step_options = [100, 1000, 10000]
        repetition_options = [100, 1000, 10000]
        dimension_options = [1, 2]
        
        total = len(dimension_options) * len(step_options) * len(repetition_options)
        current = 0
        
        for dim in dimension_options:
            for steps in step_options:
                for reps in repetition_options:
                    result = SimulationEngine.run_simulation(dim, steps, reps)
                    results.append(result)
                    current += 1
                    if progress_callback:
                        progress_callback(current / total)
        
        return results


#visualización

def plot_trajectory_1d(trajectory: np.ndarray, steps: int) -> go.Figure:
    """Genera gráfica de trayectoria en 1D"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(trajectory))),
        y=trajectory[:, 0],
        mode='lines',
        name='Posición',
        line=dict(color='blue', width=1)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Origen")
    fig.update_layout(
        title=f'Trayectoria 1D - {steps} pasos',
        xaxis_title='Paso',
        yaxis_title='Posición',
        template='plotly_white'
    )
    return fig


def plot_trajectory_2d(trajectory: np.ndarray, steps: int) -> go.Figure:
    """Genera gráfica de trayectoria en 2D"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        mode='lines',
        name='Trayectoria',
        line=dict(color='blue', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        name='Origen',
        marker=dict(color='green', size=12, symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=[trajectory[-1, 0]], y=[trajectory[-1, 1]],
        mode='markers',
        name='Final',
        marker=dict(color='red', size=12, symbol='x')
    ))
    fig.update_layout(
        title=f'Trayectoria 2D - {steps} pasos',
        xaxis_title='X',
        yaxis_title='Y',
        template='plotly_white',
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    return fig


def plot_distance_evolution(distances: np.ndarray, steps: int) -> go.Figure:
    """Genera gráfica de evolución de distancia"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(distances))),
        y=distances,
        mode='lines',
        name='Distancia',
        line=dict(color='purple', width=1)
    ))
    max_idx = np.argmax(distances)
    fig.add_trace(go.Scatter(
        x=[max_idx],
        y=[distances[max_idx]],
        mode='markers',
        name=f'Máximo: {distances[max_idx]:.2f}',
        marker=dict(color='red', size=10)
    ))
    fig.update_layout(
        title=f'Distancia desde el origen - {steps} pasos',
        xaxis_title='Paso',
        yaxis_title='Distancia Euclidiana',
        template='plotly_white'
    )
    return fig


def plot_max_distances_histogram(max_distances: List[float]) -> go.Figure:
    """Genera histograma de distancias máximas"""
    fig = px.histogram(
        x=max_distances,
        nbins=50,
        title='Distribución de Distancias Máximas',
        labels={'x': 'Distancia Máxima', 'y': 'Frecuencia'}
    )
    fig.add_vline(x=np.mean(max_distances), line_dash="dash", line_color="red",
                  annotation_text=f"Promedio: {np.mean(max_distances):.2f}")
    fig.update_layout(template='plotly_white')
    return fig


def plot_convergence(results_df: pd.DataFrame) -> go.Figure:
    """Genera gráfica de convergencia"""
    fig = go.Figure()
    
    for dim in results_df['Dimensiones'].unique():
        for steps in results_df['Pasos'].unique():
            subset = results_df[(results_df['Dimensiones'] == dim) & (results_df['Pasos'] == steps)]
            fig.add_trace(go.Scatter(
                x=subset['Repeticiones'],
                y=subset['Promedio Dist. Máx.'],
                mode='lines+markers',
                name=f'{dim}D, {steps} pasos'
            ))
    
    fig.update_layout(
        title='Convergencia del Promedio de Distancia Máxima',
        xaxis_title='Número de Repeticiones',
        yaxis_title='Promedio de Distancia Máxima',
        xaxis_type='log',
        template='plotly_white'
    )
    return fig


#funciones de utilidad 

def results_to_dataframe(results: List[SimulationResult]) -> pd.DataFrame:
    """Convierte lista de resultados a DataFrame"""
    data = []
    for r in results:
        data.append({
            'Dimensiones': r.dimensions,
            'Pasos': r.steps,
            'Repeticiones': r.repetitions,
            'Promedio Dist. Máx.': round(r.average_max_distance, 4),
            'Desv. Estándar': round(r.standard_deviation, 4),
            'Dist. Máx. Global': round(r.overall_max_distance, 4),
            'Dist. Mín. Global': round(r.overall_min_distance, 4),
            'Tiempo (ms)': round(r.execution_time_ms, 2)
        })
    return pd.DataFrame(data)


def generate_conclusions(results_df: pd.DataFrame) -> str:
    """Genera conclusiones sobre convergencia"""
    conclusions = """
## Análisis de Convergencia del Movimiento Browniano

### 1. Fundamento Teórico
Para un movimiento Browniano en d dimensiones con n pasos:
- La distancia esperada desde el origen es proporcional a √n
- La distancia máxima esperada crece aproximadamente como √(n × log(n))

### 2. Observaciones de los Resultados
"""
    
    for dim in sorted(results_df['Dimensiones'].unique()):
        conclusions += f"\n**{dim} Dimensión(es):**\n"
        subset = results_df[results_df['Dimensiones'] == dim]
        for _, row in subset.iterrows():
            theoretical = np.sqrt(row['Pasos'])
            ratio = row['Promedio Dist. Máx.'] / theoretical
            conclusions += f"- {int(row['Pasos'])} pasos, {int(row['Repeticiones'])} reps: "
            conclusions += f"Promedio={row['Promedio Dist. Máx.']:.2f}, Teórico≈{theoretical:.2f}, Ratio={ratio:.2f}\n"
    
    conclusions += """
### 3. Conclusión sobre Convergencia

si existe convergencia
- A medida que aumentan las repeticiones, el promedio de la distancia máxima se estabiliza
- La desviación estándar disminuye con más repeticiones, confirmando la **Ley de los Grandes Números**
- Los resultados convergen a valores proporcionales a √n, consistente con la teoría del movimiento Browniano
- En 2D las distancias máximas son ligeramente mayores que en 1D debido a la mayor libertad de movimiento
"""
    return conclusions


#streamlit

def main():
    st.set_page_config(
        page_title="Simulador Movimiento Browniano",
        page_icon="*",
        layout="wide"
    )
    
    st.title("Simulador de Movimiento Browniano")
    st.markdown("""
    **Ma2004b - Optimización Estocástica | ITESM**
    
    Simulación de caminata aleatoria en n dimensiones con pasos discretos unitarios.
    """)
    
    # Inicializar estado de sesión
    if 'all_results' not in st.session_state:
        st.session_state.all_results = []
    
    # Sidebar para navegación
    st.sidebar.title("Menú")
    option = st.sidebar.radio(
        "Seleccione una opción:",
        ["Simulación Personalizada", 
         "Batería Completa", 
         "Visualizar Trayectoria",
         "Ver Resultados",
         "Análisis de Convergencia"]
    )
    
    #simulacion personalizada
    if option == "Simulación Personalizada":
        st.header("Simulación Personalizada")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            dimensions = st.number_input("Dimensiones", min_value=1, max_value=10, value=2)
        with col2:
            steps = st.number_input("Pasos por caminata", min_value=10, max_value=100000, value=1000)
        with col3:
            repetitions = st.number_input("Repeticiones", min_value=10, max_value=50000, value=1000)
        
        if st.button("▶️ Ejecutar Simulación", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(p):
                progress_bar.progress(p)
                status_text.text(f"Progreso: {p*100:.1f}%")
            
            with st.spinner("Ejecutando simulación..."):
                result = SimulationEngine.run_simulation(
                    dimensions, steps, repetitions, update_progress
                )
                st.session_state.all_results.append(result)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("Simulación completada!")
            
            # Mostrar resultados
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Promedio Dist. Máx.", f"{result.average_max_distance:.4f}")
            col2.metric("Desv. Estándar", f"{result.standard_deviation:.4f}")
            col3.metric("Máximo Global", f"{result.overall_max_distance:.4f}")
            col4.metric("Tiempo (ms)", f"{result.execution_time_ms:.2f}")
            
            # Histograma
            st.plotly_chart(plot_max_distances_histogram(result.max_distances), use_container_width=True)
            
            # Mostrar primeras distancias
            with st.expander("Ver primeras 20 distancias máximas"):
                df_distances = pd.DataFrame({
                    'Caminata': range(1, min(21, len(result.max_distances)+1)),
                    'Distancia Máxima': result.max_distances[:20]
                })
                st.dataframe(df_distances, use_container_width=True)
    
    # ========================================================================
    # BATERÍA COMPLETA
    # ========================================================================
    elif option == "Batería Completa":
        st.header("Batería Completa de Simulaciones")
        
        st.info("""
        Se ejecutarán simulaciones para:
        - **Dimensiones:** 1D y 2D
        - **Pasos:** 100, 1000, 10000
        - **Repeticiones:** 100, 1000, 10000
        
        **Total:** 18 configuraciones diferentes
        """)
        
        if st.button("▶️ Ejecutar Batería Completa", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(p):
                progress_bar.progress(p)
                status_text.text(f"Progreso: {p*100:.1f}%")
            
            with st.spinner("Ejecutando batería de simulaciones..."):
                results = SimulationEngine.run_full_battery(update_progress)
                st.session_state.all_results.extend(results)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("Batería completada!")
            
            # Mostrar tabla de resultados
            df = results_to_dataframe(results)
            st.dataframe(df, use_container_width=True)
            
            # Gráfica de convergencia
            st.plotly_chart(plot_convergence(df), use_container_width=True)
    
    # ========================================================================
    # VISUALIZAR TRAYECTORIA
    # ========================================================================
    elif option == "Visualizar Trayectoria":
        st.header("Visualizar Trayectoria Individual")
        
        col1, col2 = st.columns(2)
        with col1:
            vis_dimensions = st.selectbox("Dimensiones", [1, 2], index=1)
        with col2:
            vis_steps = st.number_input("Número de pasos", min_value=10, max_value=10000, value=500)
        
        if st.button("Generar Trayectoria", type="primary"):
            brownian = BrownianMotion(vis_dimensions, vis_steps)
            trajectory, distances, max_dist = brownian.execute_walk_with_trajectory()
            
            st.metric("Distancia Máxima Alcanzada", f"{max_dist:.4f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if vis_dimensions == 1:
                    st.plotly_chart(plot_trajectory_1d(trajectory, vis_steps), use_container_width=True)
                else:
                    st.plotly_chart(plot_trajectory_2d(trajectory, vis_steps), use_container_width=True)
            
            with col2:
                st.plotly_chart(plot_distance_evolution(distances, vis_steps), use_container_width=True)
    
    # ========================================================================
    # VER RESULTADOS
    # ========================================================================
    elif option == "Ver Resultados":
        st.header("Todos los Resultados Almacenados")
        
        if not st.session_state.all_results:
            st.warning("No hay resultados almacenados. Ejecute una simulación primero.")
        else:
            df = results_to_dataframe(st.session_state.all_results)
            st.dataframe(df, use_container_width=True)
            
            # Descargar como CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name="resultados_browniano.csv",
                mime="text/csv"
            )
            
            # Gráfica de convergencia si hay suficientes datos
            if len(df) > 1:
                st.plotly_chart(plot_convergence(df), use_container_width=True)
            
            # Botón para limpiar resultados
            if st.button("Limpiar todos los resultados"):
                st.session_state.all_results = []
                st.rerun()
    
    #análisis de convergencia
    elif option == "Análisis de Convergencia":
        st.header("Análisis de Convergencia")
        
        if not st.session_state.all_results:
            st.warning("No hay resultados para analizar. Ejecute una simulación primero.")
        else:
            df = results_to_dataframe(st.session_state.all_results)
            conclusions = generate_conclusions(df)
            st.markdown(conclusions)
            
            # Gráfica de convergencia
            if len(df) > 1:
                st.plotly_chart(plot_convergence(df), use_container_width=True)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Información:**
    - Resultados almacenados: {}
    """.format(len(st.session_state.all_results)))


if __name__ == "__main__":
    main()