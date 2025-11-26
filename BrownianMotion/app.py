"""
Simulador de Movimiento Browniano - Streamlit App
Ma2004b - Optimizacion Estocastica - ITESM
Version optimizada con vectorizacion y paralelismo
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

# Importar joblib si esta disponible, sino usar fallback
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ============================================================================
# CLASES DEL SIMULADOR
# ============================================================================

@dataclass
class SimulationResult:
    """Clase que almacena los resultados de una simulacion de Movimiento Browniano"""
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
    """Clase que implementa la simulacion del Movimiento Browniano (vectorizada)"""
    
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
        """Reinicia la posicion de la particula al origen"""
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
        """Ejecuta una caminata completa VECTORIZADA y retorna la distancia maxima"""
        # Generar todos los pasos de una vez (vectorizado)
        dims_chosen = self.rng.integers(0, self.dimensions, size=self.steps)
        directions = self.rng.choice([-1, 1], size=self.steps)
        
        # Construir matriz de movimientos
        moves = np.zeros((self.steps, self.dimensions))
        moves[np.arange(self.steps), dims_chosen] = directions
        
        # Calcular posiciones acumuladas
        positions = np.cumsum(moves, axis=0)
        
        # Calcular distancias desde el origen
        distances = np.sqrt(np.sum(positions ** 2, axis=1))
        
        return float(np.max(distances))
    
    def execute_walk_with_trajectory(self) -> tuple:
        """Ejecuta una caminata VECTORIZADA y retorna la trayectoria completa"""
        # Generar todos los pasos de una vez
        dims_chosen = self.rng.integers(0, self.dimensions, size=self.steps)
        directions = self.rng.choice([-1, 1], size=self.steps)
        
        # Construir matriz de movimientos
        moves = np.zeros((self.steps, self.dimensions))
        moves[np.arange(self.steps), dims_chosen] = directions
        
        # Calcular posiciones acumuladas (incluir origen)
        positions = np.vstack([np.zeros(self.dimensions), np.cumsum(moves, axis=0)])
        
        # Calcular distancias desde el origen
        distances = np.sqrt(np.sum(positions ** 2, axis=1))
        max_distance = float(np.max(distances))
        
        return positions, distances, max_distance


def _run_single_walk(dimensions: int, steps: int) -> float:
    """Funcion auxiliar para ejecutar una caminata (usada en paralelismo)"""
    brownian = BrownianMotion(dimensions, steps)
    return brownian.execute_walk()


class SimulationEngine:
    """Motor de simulacion que ejecuta multiples experimentos con paralelismo"""
    
    @staticmethod
    def run_simulation(dimensions: int, steps: int, repetitions: int, 
                       progress_callback=None, use_parallel: bool = True) -> SimulationResult:
        """Ejecuta una simulacion con los parametros especificados (con paralelismo opcional)"""
        result = SimulationResult(
            dimensions=dimensions,
            steps=steps,
            repetitions=repetitions
        )
        
        start_time = time.perf_counter()
        
        if use_parallel and JOBLIB_AVAILABLE and repetitions >= 100:
            # Usar paralelismo para muchas repeticiones
            n_jobs = -1  # Usar todos los cores disponibles
            max_distances = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(_run_single_walk)(dimensions, steps) 
                for _ in range(repetitions)
            )
            result.max_distances = list(max_distances)
            if progress_callback:
                progress_callback(1.0)
        else:
            # Ejecucion secuencial (vectorizada, sigue siendo rapida)
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
    def run_multiple_trajectories(dimensions: int, steps: int, n_experiments: int,
                                   progress_callback=None) -> List[tuple]:
        """Ejecuta multiples experimentos y retorna todas las trayectorias"""
        trajectories = []
        
        for i in range(n_experiments):
            brownian = BrownianMotion(dimensions, steps)
            trajectory, distances, max_dist = brownian.execute_walk_with_trajectory()
            trajectories.append((trajectory, distances, max_dist))
            if progress_callback:
                progress_callback((i + 1) / n_experiments)
        
        return trajectories
    
    @staticmethod
    def run_full_battery(progress_callback=None) -> List[SimulationResult]:
        """Ejecuta una bateria completa de simulaciones (con paralelismo)"""
        results = []
        step_options = [100, 1000, 10000]
        repetition_options = [100, 1000, 10000]
        dimension_options = [1, 2]
        
        total = len(dimension_options) * len(step_options) * len(repetition_options)
        current = 0
        
        for dim in dimension_options:
            for steps in step_options:
                for reps in repetition_options:
                    result = SimulationEngine.run_simulation(dim, steps, reps, use_parallel=True)
                    results.append(result)
                    current += 1
                    if progress_callback:
                        progress_callback(current / total)
        
        return results


# ============================================================================
# FUNCIONES DE VISUALIZACION
# ============================================================================

def plot_trajectory_1d(trajectory: np.ndarray, steps: int, label: str = "Posicion") -> go.Figure:
    """Genera grafica de trayectoria en 1D"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(trajectory))),
        y=trajectory[:, 0],
        mode='lines',
        name=label,
        line=dict(width=1)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Origen")
    fig.update_layout(
        title=f'Trayectoria 1D - {steps} pasos',
        xaxis_title='Paso',
        yaxis_title='Posicion',
        template='plotly_white'
    )
    return fig


def plot_multiple_trajectories_1d(trajectories: List[tuple], steps: int) -> go.Figure:
    """Genera grafica con multiples trayectorias en 1D"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, (trajectory, distances, max_dist) in enumerate(trajectories):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=list(range(len(trajectory))),
            y=trajectory[:, 0],
            mode='lines',
            name=f'Exp {i+1} (max: {max_dist:.2f})',
            line=dict(width=1.5, color=color),
            opacity=0.8
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Origen")
    fig.update_layout(
        title=f'Multiples Trayectorias 1D - {steps} pasos ({len(trajectories)} experimentos)',
        xaxis_title='Paso',
        yaxis_title='Posicion',
        template='plotly_white',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02)
    )
    return fig


def plot_trajectory_2d(trajectory: np.ndarray, steps: int) -> go.Figure:
    """Genera grafica de trayectoria en 2D"""
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


def plot_multiple_trajectories_2d(trajectories: List[tuple], steps: int) -> go.Figure:
    """Genera grafica con multiples trayectorias en 2D"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    # Agregar origen
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        name='Origen',
        marker=dict(color='green', size=14, symbol='circle'),
        showlegend=True
    ))
    
    for i, (trajectory, distances, max_dist) in enumerate(trajectories):
        color = colors[i % len(colors)]
        
        # Trayectoria
        fig.add_trace(go.Scatter(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            mode='lines',
            name=f'Exp {i+1} (max: {max_dist:.2f})',
            line=dict(width=1.5, color=color),
            opacity=0.7
        ))
        
        # Punto final
        fig.add_trace(go.Scatter(
            x=[trajectory[-1, 0]], 
            y=[trajectory[-1, 1]],
            mode='markers',
            name=f'Final {i+1}',
            marker=dict(color=color, size=10, symbol='x'),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f'Multiples Trayectorias 2D - {steps} pasos ({len(trajectories)} experimentos)',
        xaxis_title='X',
        yaxis_title='Y',
        template='plotly_white',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02)
    )
    return fig


def plot_trajectory_3d(trajectory: np.ndarray, steps: int) -> go.Figure:
    """Genera grafica de trayectoria en 3D"""
    fig = go.Figure()
    
    # Trayectoria
    fig.add_trace(go.Scatter3d(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        z=trajectory[:, 2],
        mode='lines',
        name='Trayectoria',
        line=dict(color='blue', width=2)
    ))
    
    # Origen
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        name='Origen',
        marker=dict(color='green', size=8, symbol='circle')
    ))
    
    # Punto final
    fig.add_trace(go.Scatter3d(
        x=[trajectory[-1, 0]], 
        y=[trajectory[-1, 1]],
        z=[trajectory[-1, 2]],
        mode='markers',
        name='Final',
        marker=dict(color='red', size=8, symbol='x')
    ))
    
    fig.update_layout(
        title=f'Trayectoria 3D - {steps} pasos',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        template='plotly_white'
    )
    return fig


def plot_multiple_trajectories_3d(trajectories: List[tuple], steps: int) -> go.Figure:
    """Genera grafica con multiples trayectorias en 3D"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    # Agregar origen
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        name='Origen',
        marker=dict(color='green', size=10, symbol='circle'),
        showlegend=True
    ))
    
    for i, (trajectory, distances, max_dist) in enumerate(trajectories):
        color = colors[i % len(colors)]
        
        # Trayectoria
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines',
            name=f'Exp {i+1} (max: {max_dist:.2f})',
            line=dict(width=3, color=color),
            opacity=0.7
        ))
        
        # Punto final
        fig.add_trace(go.Scatter3d(
            x=[trajectory[-1, 0]], 
            y=[trajectory[-1, 1]],
            z=[trajectory[-1, 2]],
            mode='markers',
            name=f'Final {i+1}',
            marker=dict(color=color, size=6, symbol='x'),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f'Multiples Trayectorias 3D - {steps} pasos ({len(trajectories)} experimentos)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02)
    )
    return fig


def plot_final_positions_3d(trajectories: List[tuple]) -> go.Figure:
    """Genera scatter plot 3D de posiciones finales (visualiza la 'esfera' de dispersion)"""
    # Extraer posiciones finales
    final_positions_x = [traj[0][-1, 0] for traj in trajectories]
    final_positions_y = [traj[0][-1, 1] for traj in trajectories]
    final_positions_z = [traj[0][-1, 2] for traj in trajectories]
    
    # Calcular estadisticas
    mean_x = np.mean(final_positions_x)
    mean_y = np.mean(final_positions_y)
    mean_z = np.mean(final_positions_z)
    distances_from_origin = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in 
                             zip(final_positions_x, final_positions_y, final_positions_z)]
    mean_radius = np.mean(distances_from_origin)
    
    fig = go.Figure()
    
    # Posiciones finales
    fig.add_trace(go.Scatter3d(
        x=final_positions_x,
        y=final_positions_y,
        z=final_positions_z,
        mode='markers',
        name='Posiciones Finales',
        marker=dict(color='blue', size=8, opacity=0.7),
        text=[f'Exp {i+1}: ({x:.1f}, {y:.1f}, {z:.1f})' for i, (x, y, z) in 
              enumerate(zip(final_positions_x, final_positions_y, final_positions_z))],
        hoverinfo='text'
    ))
    
    # Origen
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        name='Origen',
        marker=dict(color='green', size=10, symbol='circle')
    ))
    
    # Centro de masa
    fig.add_trace(go.Scatter3d(
        x=[mean_x], y=[mean_y], z=[mean_z],
        mode='markers',
        name=f'Centro de masa ({mean_x:.2f}, {mean_y:.2f}, {mean_z:.2f})',
        marker=dict(color='red', size=10, symbol='diamond')
    ))
    
    # Esfera de radio promedio (aproximada con puntos)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sphere_x = mean_radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = mean_radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = mean_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        opacity=0.2,
        colorscale=[[0, 'orange'], [1, 'orange']],
        showscale=False,
        name=f'Radio promedio: {mean_radius:.2f}'
    ))
    
    fig.update_layout(
        title=f'Distribucion de Posiciones Finales 3D (Radio promedio: {mean_radius:.2f})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def plot_final_positions_circle(trajectories: List[tuple]) -> go.Figure:
    """Genera scatter plot de posiciones finales (visualiza el 'circulo' de dispersion)"""
    # Extraer posiciones finales
    final_positions_x = [traj[0][-1, 0] for traj in trajectories]
    final_positions_y = [traj[0][-1, 1] for traj in trajectories]
    
    # Calcular estadisticas
    mean_x = np.mean(final_positions_x)
    mean_y = np.mean(final_positions_y)
    distances_from_origin = [np.sqrt(x**2 + y**2) for x, y in zip(final_positions_x, final_positions_y)]
    mean_radius = np.mean(distances_from_origin)
    
    fig = go.Figure()
    
    # Posiciones finales
    fig.add_trace(go.Scatter(
        x=final_positions_x,
        y=final_positions_y,
        mode='markers',
        name='Posiciones Finales',
        marker=dict(color='blue', size=10, opacity=0.7),
        text=[f'Exp {i+1}: ({x:.1f}, {y:.1f})' for i, (x, y) in enumerate(zip(final_positions_x, final_positions_y))],
        hoverinfo='text'
    ))
    
    # Origen
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        name='Origen',
        marker=dict(color='green', size=14, symbol='circle')
    ))
    
    # Centro de masa
    fig.add_trace(go.Scatter(
        x=[mean_x], y=[mean_y],
        mode='markers',
        name=f'Centro de masa ({mean_x:.2f}, {mean_y:.2f})',
        marker=dict(color='red', size=12, symbol='diamond')
    ))
    
    # Circulo de radio promedio
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = mean_radius * np.cos(theta)
    circle_y = mean_radius * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode='lines',
        name=f'Radio promedio: {mean_radius:.2f}',
        line=dict(color='orange', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title='Distribucion de Posiciones Finales (Circulo de Dispersion)',
        xaxis_title='X',
        yaxis_title='Y',
        template='plotly_white',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True
    )
    
    return fig


def plot_distance_evolution(distances: np.ndarray, steps: int) -> go.Figure:
    """Genera grafica de evolucion de distancia"""
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
        name=f'Maximo: {distances[max_idx]:.2f}',
        marker=dict(color='red', size=10)
    ))
    fig.update_layout(
        title=f'Distancia desde el origen - {steps} pasos',
        xaxis_title='Paso',
        yaxis_title='Distancia Euclidiana',
        template='plotly_white'
    )
    return fig


def plot_multiple_distance_evolution(trajectories: List[tuple], steps: int) -> go.Figure:
    """Genera grafica de evolucion de distancia para multiples experimentos"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, (trajectory, distances, max_dist) in enumerate(trajectories):
        color = colors[i % len(colors)]
        max_idx = np.argmax(distances)
        
        fig.add_trace(go.Scatter(
            x=list(range(len(distances))),
            y=distances,
            mode='lines',
            name=f'Exp {i+1}',
            line=dict(width=1.5, color=color),
            opacity=0.7
        ))
        
        # Punto maximo
        fig.add_trace(go.Scatter(
            x=[max_idx],
            y=[distances[max_idx]],
            mode='markers',
            name=f'Max {i+1}: {distances[max_idx]:.2f}',
            marker=dict(color=color, size=8, symbol='star'),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f'Evolucion de Distancia - {steps} pasos ({len(trajectories)} experimentos)',
        xaxis_title='Paso',
        yaxis_title='Distancia Euclidiana',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02)
    )
    return fig


def plot_max_distances_histogram(max_distances: List[float]) -> go.Figure:
    """Genera histograma de distancias maximas"""
    fig = px.histogram(
        x=max_distances,
        nbins=50,
        title='Distribucion de Distancias Maximas',
        labels={'x': 'Distancia Maxima', 'y': 'Frecuencia'}
    )
    fig.add_vline(x=np.mean(max_distances), line_dash="dash", line_color="red",
                  annotation_text=f"Promedio: {np.mean(max_distances):.2f}")
    fig.update_layout(template='plotly_white')
    return fig


def plot_convergence(results_df: pd.DataFrame) -> go.Figure:
    """Genera grafica de convergencia"""
    fig = go.Figure()
    
    for dim in results_df['Dimensiones'].unique():
        for steps in results_df['Pasos'].unique():
            subset = results_df[(results_df['Dimensiones'] == dim) & (results_df['Pasos'] == steps)]
            fig.add_trace(go.Scatter(
                x=subset['Repeticiones'],
                y=subset['Promedio Dist. Max.'],
                mode='lines+markers',
                name=f'{dim}D, {steps} pasos'
            ))
    
    fig.update_layout(
        title='Convergencia del Promedio de Distancia Maxima',
        xaxis_title='Numero de Repeticiones',
        yaxis_title='Promedio de Distancia Maxima',
        xaxis_type='log',
        template='plotly_white'
    )
    return fig


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def results_to_dataframe(results: List[SimulationResult]) -> pd.DataFrame:
    """Convierte lista de resultados a DataFrame"""
    data = []
    for r in results:
        data.append({
            'Dimensiones': r.dimensions,
            'Pasos': r.steps,
            'Repeticiones': r.repetitions,
            'Promedio Dist. Max.': round(r.average_max_distance, 4),
            'Desv. Estandar': round(r.standard_deviation, 4),
            'Dist. Max. Global': round(r.overall_max_distance, 4),
            'Dist. Min. Global': round(r.overall_min_distance, 4),
            'Tiempo (ms)': round(r.execution_time_ms, 2)
        })
    return pd.DataFrame(data)


def trajectories_to_dataframe(trajectories: List[tuple]) -> pd.DataFrame:
    """Convierte trayectorias a DataFrame con informacion resumida"""
    data = []
    for i, (trajectory, distances, max_dist) in enumerate(trajectories):
        final_pos = trajectory[-1]
        final_dist = distances[-1]
        data.append({
            'Experimento': i + 1,
            'Distancia Maxima': round(max_dist, 4),
            'Distancia Final': round(final_dist, 4),
            'Posicion Final': str(np.round(final_pos, 2).tolist()),
            'Paso del Maximo': int(np.argmax(distances))
        })
    return pd.DataFrame(data)


def generate_conclusions(results_df: pd.DataFrame) -> str:
    """Genera conclusiones sobre convergencia"""
    conclusions = """
## Analisis de Convergencia del Movimiento Browniano

### 1. Fundamento Teorico
Para un movimiento Browniano en d dimensiones con n pasos:
- La distancia esperada desde el origen es proporcional a sqrt(n)
- La distancia maxima esperada crece aproximadamente como sqrt(n * log(n))

### 2. Observaciones de los Resultados
"""
    
    for dim in sorted(results_df['Dimensiones'].unique()):
        conclusions += f"\n**{dim} Dimension(es):**\n"
        subset = results_df[results_df['Dimensiones'] == dim]
        for _, row in subset.iterrows():
            theoretical = np.sqrt(row['Pasos'])
            ratio = row['Promedio Dist. Max.'] / theoretical
            conclusions += f"- {int(row['Pasos'])} pasos, {int(row['Repeticiones'])} reps: "
            conclusions += f"Promedio={row['Promedio Dist. Max.']:.2f}, Teorico={theoretical:.2f}, Ratio={ratio:.2f}\n"
    
    conclusions += """
### 3. Conclusion sobre Convergencia

**Si existe convergencia:**
- A medida que aumentan las repeticiones, el promedio de la distancia maxima se estabiliza
- La desviacion estandar disminuye con mas repeticiones, confirmando la **Ley de los Grandes Numeros**
- Los resultados convergen a valores proporcionales a sqrt(n), consistente con la teoria del movimiento Browniano
- En 2D las distancias maximas son ligeramente mayores que en 1D debido a la mayor libertad de movimiento
"""
    return conclusions


def save_results_auto(df: pd.DataFrame, prefix: str = "resultados"):
    """Guarda resultados automaticamente con timestamp"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    return df.to_csv(index=False), filename


# ============================================================================
# APLICACION STREAMLIT - FLUJO UNIFICADO
# ============================================================================

def main():
    st.set_page_config(
        page_title="Simulador Movimiento Browniano",
        page_icon=None,
        layout="wide"
    )
    
    st.title("Simulador de Movimiento Browniano")
    st.markdown("""
    **Ma2004b - Optimizacion Estocastica | ITESM**
    
    Simulacion de caminata aleatoria en n dimensiones con pasos discretos unitarios.
    
    ---
    """)
    
    # Inicializar estado de sesion
    if 'all_results' not in st.session_state:
        st.session_state.all_results = []
    if 'last_trajectories' not in st.session_state:
        st.session_state.last_trajectories = None
    if 'last_simulation_result' not in st.session_state:
        st.session_state.last_simulation_result = None
    
    # ========================================================================
    # PASO 1: CONFIGURACION Y SIMULACION
    # ========================================================================
    with st.expander("PASO 1: Configuracion y Simulacion", expanded=True):
        st.subheader("Parametros de Entrada")
        
        st.markdown("""
        Configure los parametros para la simulacion del movimiento Browniano:
        - **Dimensiones**: Numero de ejes del sistema de coordenadas (1-5)
        - **Pasos**: Numero de movimientos por caminata
        - **Repeticiones**: Numero de veces que se repite el experimento
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            dimensions = st.number_input(
                "Dimensiones", 
                min_value=1, 
                max_value=5,  # Limite: >2 no se visualiza bien
                value=2,
                help="Numero de dimensiones del sistema de coordenadas (max 5)"
            )
        with col2:
            steps = st.number_input(
                "Pasos por caminata", 
                min_value=10, 
                max_value=50000,  # Limite de seguridad
                value=1000,
                help="Numero de pasos que da la particula (max 50000)"
            )
        with col3:
            repetitions = st.number_input(
                "Repeticiones (n)", 
                min_value=10, 
                max_value=20000,  # Limite para evitar lentitud
                value=1000,
                help="Numero de veces que se repite el experimento (max 20000)"
            )
        
        # Warning para repeticiones altas
        if repetitions > 1000:
            st.warning("Advertencia: Repeticiones mayores a 1000 pueden tardar mas tiempo. Se utilizara procesamiento paralelo para optimizar.")
        
        if dimensions > 2:
            st.info("Nota: Para dimensiones mayores a 2, la visualizacion de trayectorias no estara disponible.")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Ejecutar Simulacion Personalizada", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(p):
                    progress_bar.progress(min(p, 1.0))
                    status_text.text(f"Progreso: {p*100:.1f}%")
                
                with st.spinner("Ejecutando simulacion con procesamiento paralelo..."):
                    result = SimulationEngine.run_simulation(
                        dimensions, steps, repetitions, update_progress, use_parallel=True
                    )
                    st.session_state.all_results.append(result)
                    st.session_state.last_simulation_result = result
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"Simulacion completada: {repetitions} experimentos en {result.execution_time_ms:.2f} ms")
        
        with col_btn2:
            if st.button("Ejecutar Bateria Completa (18 configuraciones)"):
                st.info("Ejecutando bateria: 1D y 2D, pasos [100, 1000, 10000], repeticiones [100, 1000, 10000]")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(p):
                    progress_bar.progress(min(p, 1.0))
                    status_text.text(f"Progreso bateria: {p*100:.1f}%")
                
                with st.spinner("Ejecutando bateria de simulaciones con paralelismo..."):
                    results = SimulationEngine.run_full_battery(update_progress)
                    st.session_state.all_results.extend(results)
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("Bateria completada: 18 configuraciones ejecutadas")
    
    # ========================================================================
    # PASO 2: RESULTADOS DE SIMULACION
    # ========================================================================
    with st.expander("PASO 2: Resultados de Simulacion", expanded=bool(st.session_state.all_results)):
        if not st.session_state.all_results:
            st.info("Ejecute una simulacion en el Paso 1 para ver los resultados aqui.")
        else:
            st.subheader("Tabla de Resultados")
            
            df = results_to_dataframe(st.session_state.all_results)
            st.dataframe(df, use_container_width=True)
            
            # Mostrar metricas del ultimo resultado
            if st.session_state.last_simulation_result:
                result = st.session_state.last_simulation_result
                st.subheader("Ultima Simulacion")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Promedio Dist. Max.", f"{result.average_max_distance:.4f}")
                col2.metric("Desv. Estandar", f"{result.standard_deviation:.4f}")
                col3.metric("Maximo Global", f"{result.overall_max_distance:.4f}")
                col4.metric("Tiempo (ms)", f"{result.execution_time_ms:.2f}")
                
                # Histograma
                st.plotly_chart(plot_max_distances_histogram(result.max_distances), use_container_width=True, key="histogram_paso2")
            
            # Grafica de convergencia
            if len(df) > 1:
                st.subheader("Grafica de Convergencia")
                st.plotly_chart(plot_convergence(df), use_container_width=True, key="convergence_paso2")
            
            # Descargas CSV
            st.subheader("Descargar Datos")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                csv_results = df.to_csv(index=False)
                st.download_button(
                    label="Descargar Tabla de Resultados (CSV)",
                    data=csv_results,
                    file_name="resultados_browniano.csv",
                    mime="text/csv"
                )
            
            with col_dl2:
                if st.session_state.last_simulation_result:
                    result = st.session_state.last_simulation_result
                    df_distances = pd.DataFrame({
                        'Caminata': range(1, len(result.max_distances)+1),
                        'Distancia_Maxima': result.max_distances
                    })
                    csv_dist = df_distances.to_csv(index=False)
                    st.download_button(
                        label="Descargar Distancias Maximas (CSV)",
                        data=csv_dist,
                        file_name="distancias_maximas.csv",
                        mime="text/csv"
                    )
            
            # Boton limpiar
            if st.button("Limpiar todos los resultados"):
                st.session_state.all_results = []
                st.session_state.last_simulation_result = None
                st.rerun()
    
    # ========================================================================
    # PASO 3: VISUALIZACION DE TRAYECTORIAS
    # ========================================================================
    with st.expander("PASO 3: Visualizacion de Trayectorias", expanded=False):
        st.subheader("Generar y Visualizar Trayectorias")
        
        st.markdown("""
        Visualice multiples experimentos con sus trayectorias superpuestas.
        Cada experimento se muestra en un color diferente.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            vis_dimensions = st.selectbox(
                "Dimensiones para visualizacion", 
                [1, 2, 3], 
                index=1,
                help="1D, 2D o 3D disponibles para visualizacion"
            )
        with col2:
            vis_steps = st.number_input(
                "Numero de pasos", 
                min_value=10, 
                max_value=10000, 
                value=500,
                help="Pasos por caminata para visualizacion"
            )
        with col3:
            n_experiments = st.number_input(
                "Repeticiones (n) para visualizar", 
                min_value=1, 
                max_value=10000, 
                value=5,
                help="Numero de experimentos a visualizar (max 20 para claridad)"
            )
        
        if st.button("Generar Trayectorias"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(p):
                progress_bar.progress(min(p, 1.0))
                status_text.text(f"Generando experimento {int(p * n_experiments)}/{n_experiments}...")
            
            with st.spinner(f"Generando {n_experiments} trayectorias..."):
                start_time = time.perf_counter()
                trajectories = SimulationEngine.run_multiple_trajectories(
                    vis_dimensions, vis_steps, n_experiments, update_progress
                )
                execution_time = (time.perf_counter() - start_time) * 1000
                st.session_state.last_trajectories = trajectories
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"{n_experiments} experimentos generados en {execution_time:.2f} ms")
        
        # Mostrar trayectorias si existen
        if st.session_state.last_trajectories:
            trajectories = st.session_state.last_trajectories
            max_distances = [t[2] for t in trajectories]
            
            # Metricas
            st.subheader("Estadisticas de Trayectorias")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Promedio Dist. Max.", f"{np.mean(max_distances):.4f}")
            col2.metric("Desv. Estandar", f"{np.std(max_distances, ddof=1):.4f}" if len(max_distances) > 1 else "N/A")
            col3.metric("Maximo de Maximos", f"{np.max(max_distances):.4f}")
            col4.metric("Minimo de Maximos", f"{np.min(max_distances):.4f}")
            
            # Graficas de trayectorias
            st.subheader("Visualizacion de Trayectorias")
            
            # Detectar dimensiones de las trayectorias
            traj_dims = trajectories[0][0].shape[1]
            
            if traj_dims == 3:
                # Para 3D, mostrar la grafica en ancho completo
                st.plotly_chart(plot_multiple_trajectories_3d(trajectories, vis_steps), 
                                use_container_width=True, key="traj_3d_paso3")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_multiple_distance_evolution(trajectories, vis_steps), 
                                    use_container_width=True, key="dist_evolution_paso3")
                with col2:
                    # Esfera de dispersion para 3D
                    st.plotly_chart(plot_final_positions_3d(trajectories), 
                                    use_container_width=True, key="sphere_paso3")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    if traj_dims == 1:
                        st.plotly_chart(plot_multiple_trajectories_1d(trajectories, vis_steps), 
                                        use_container_width=True, key="traj_1d_paso3")
                    else:
                        st.plotly_chart(plot_multiple_trajectories_2d(trajectories, vis_steps), 
                                        use_container_width=True, key="traj_2d_paso3")
                
                with col2:
                    st.plotly_chart(plot_multiple_distance_evolution(trajectories, vis_steps), 
                                    use_container_width=True, key="dist_evolution_paso3")
            
            # Grafica del circulo de dispersion (solo para 2D)
            if traj_dims == 2:
                st.subheader("Circulo de Dispersion - Posiciones Finales")
                st.markdown("Esta grafica muestra donde terminan las particulas, formando un patron circular caracteristico del movimiento Browniano.")
                st.plotly_chart(plot_final_positions_circle(trajectories), use_container_width=True, key="circle_paso3")
            
            # Tabla de resumen
            st.subheader("Resumen de Experimentos")
            df_summary = trajectories_to_dataframe(trajectories)
            st.dataframe(df_summary, use_container_width=True)
            
            # Histograma
            if len(trajectories) >= 3:
                st.subheader("Distribucion de Distancias Maximas")
                st.plotly_chart(plot_max_distances_histogram(max_distances), use_container_width=True, key="histogram_paso3")
            
            # Descarga CSV
            csv_summary = df_summary.to_csv(index=False)
            st.download_button(
                label="Descargar Resumen de Trayectorias (CSV)",
                data=csv_summary,
                file_name="trayectorias_resumen.csv",
                mime="text/csv"
            )
    
    # ========================================================================
    # PASO 4: ANALISIS DE CONVERGENCIA
    # ========================================================================
    with st.expander("PASO 4: Analisis de Convergencia y Conclusiones", expanded=False):
        st.subheader("Analisis de Convergencia")
        
        if not st.session_state.all_results:
            st.info("Ejecute simulaciones en el Paso 1 para generar el analisis de convergencia.")
        else:
            df = results_to_dataframe(st.session_state.all_results)
            
            # Conclusiones
            conclusions = generate_conclusions(df)
            st.markdown(conclusions)
            
            # Grafica de convergencia
            if len(df) > 1:
                st.plotly_chart(plot_convergence(df), use_container_width=True, key="convergence_paso4")
            
            # Tabla de datos
            st.subheader("Datos Utilizados para el Analisis")
            st.dataframe(df, use_container_width=True)
            
            # Descarga del analisis
            csv_analysis = df.to_csv(index=False)
            st.download_button(
                label="Descargar Datos de Analisis (CSV)",
                data=csv_analysis,
                file_name="analisis_convergencia.csv",
                mime="text/csv"
            )
    
    # ========================================================================
    # SIDEBAR - INFORMACION
    # ========================================================================
    st.sidebar.title("Informacion")
    st.sidebar.markdown(f"""
    **Estado de la Sesion:**
    - Simulaciones almacenadas: {len(st.session_state.all_results)}
    - Trayectorias en memoria: {'Si' if st.session_state.last_trajectories else 'No'}
    
    ---
    
    **Descripcion del Modelo:**
    - Distancia: Euclidiana desde el origen
    - Pasos: Discretos unitarios
    - Direccion: Uniforme al azar
    - Probabilidad: 0.5 incremento/decremento
    
    ---
    
    **Limites de Entrada:**
    - Dimensiones: 1-5
    - Pasos: 10-50,000
    - Repeticiones: 10-20,000
    
    ---
    
    **Optimizaciones:**
    - Vectorizacion NumPy: Activa
    - Paralelismo joblib: {'Activo' if JOBLIB_AVAILABLE else 'No disponible'}
    """)


if __name__ == "__main__":
    main()
