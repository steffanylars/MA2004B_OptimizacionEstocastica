using System;
using System.Collections.Generic;
using System.Linq;

namespace BrownianMotionSimulation
{
    /// <summary>
    /// Clase que almacena los resultados de una simulación de Movimiento Browniano
    /// </summary>
    public class SimulationResult
    {
        public int Dimensions { get; set; }
        public int Steps { get; set; }
        public int Repetitions { get; set; }
        public List<double> MaxDistances { get; set; }
        public double ExecutionTimeMs { get; set; }

        public SimulationResult()
        {
            MaxDistances = new List<double>();
        }

        /// <summary>
        /// Calcula el promedio de las distancias máximas
        /// </summary>
        public double AverageMaxDistance => MaxDistances.Count > 0 ? MaxDistances.Average() : 0;

        /// <summary>
        /// Obtiene la distancia máxima de todas las caminatas
        /// </summary>
        public double OverallMaxDistance => MaxDistances.Count > 0 ? MaxDistances.Max() : 0;

        /// <summary>
        /// Obtiene la distancia mínima de las máximas de cada caminata
        /// </summary>
        public double OverallMinDistance => MaxDistances.Count > 0 ? MaxDistances.Min() : 0;

        /// <summary>
        /// Calcula la desviación estándar de las distancias máximas
        /// </summary>
        public double StandardDeviation
        {
            get
            {
                if (MaxDistances.Count < 2) return 0;
                double avg = AverageMaxDistance;
                double sumSquaredDiff = MaxDistances.Sum(d => Math.Pow(d - avg, 2));
                return Math.Sqrt(sumSquaredDiff / (MaxDistances.Count - 1));
            }
        }

        /// <summary>
        /// Genera un resumen de los resultados en formato texto
        /// </summary>
        public override string ToString()
        {
            return $"Dimensiones: {Dimensions}, Pasos: {Steps}, Repeticiones: {Repetitions}\n" +
                   $"  Promedio Distancia Máxima: {AverageMaxDistance:F4}\n" +
                   $"  Desviación Estándar: {StandardDeviation:F4}\n" +
                   $"  Distancia Máxima Global: {OverallMaxDistance:F4}\n" +
                   $"  Distancia Mínima Global: {OverallMinDistance:F4}\n" +
                   $"  Tiempo de Ejecución: {ExecutionTimeMs:F2} ms";
        }
    }
}
