using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace BrownianMotionSimulation
{
    /// <summary>
    /// Motor de simulación que ejecuta múltiples experimentos de Movimiento Browniano
    /// </summary>
    public class SimulationEngine
    {
        /// <summary>
        /// Ejecuta una simulación con los parámetros especificados
        /// </summary>
        /// <param name="dimensions">Número de dimensiones</param>
        /// <param name="steps">Número de pasos por caminata</param>
        /// <param name="repetitions">Número de repeticiones del experimento</param>
        /// <returns>Objeto SimulationResult con los resultados</returns>
        public SimulationResult RunSimulation(int dimensions, int steps, int repetitions)
        {
            var result = new SimulationResult
            {
                Dimensions = dimensions,
                Steps = steps,
                Repetitions = repetitions
            };

            var stopwatch = Stopwatch.StartNew();
            var brownian = new BrownianMotion(dimensions, steps);

            for (int i = 0; i < repetitions; i++)
            {
                double maxDistance = brownian.ExecuteWalk();
                result.MaxDistances.Add(maxDistance);
            }

            stopwatch.Stop();
            result.ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds;

            return result;
        }

        /// <summary>
        /// Ejecuta una batería completa de simulaciones según los requisitos del ejercicio
        /// </summary>
        /// <returns>Lista de todos los resultados</returns>
        public List<SimulationResult> RunFullBattery()
        {
            var results = new List<SimulationResult>();
            int[] stepOptions = { 100, 1000, 10000 };
            int[] repetitionOptions = { 100, 1000, 10000 };
            int[] dimensionOptions = { 1, 2 };

            foreach (int dim in dimensionOptions)
            {
                foreach (int steps in stepOptions)
                {
                    foreach (int reps in repetitionOptions)
                    {
                        Console.WriteLine($"Ejecutando: {dim}D, {steps} pasos, {reps} repeticiones...");
                        var result = RunSimulation(dim, steps, reps);
                        results.Add(result);
                    }
                }
            }

            return results;
        }

        /// <summary>
        /// Genera una tabla formateada con los resultados
        /// </summary>
        /// <param name="results">Lista de resultados</param>
        /// <returns>Tabla en formato texto</returns>
        public string GenerateResultsTable(List<SimulationResult> results)
        {
            string separator = new string('-', 100);
            string header = string.Format("{0,-12} {1,-10} {2,-12} {3,-18} {4,-15} {5,-15}",
                "Dimensiones", "Pasos", "Repeticiones", "Promedio Dist Max", "Desv. Estándar", "Tiempo (ms)");

            var lines = new List<string>
            {
                separator,
                "TABLA DE RESULTADOS - SIMULACIÓN DE MOVIMIENTO BROWNIANO",
                separator,
                header,
                separator
            };

            foreach (var result in results)
            {
                string line = string.Format("{0,-12} {1,-10} {2,-12} {3,-18:F4} {4,-15:F4} {5,-15:F2}",
                    result.Dimensions,
                    result.Steps,
                    result.Repetitions,
                    result.AverageMaxDistance,
                    result.StandardDeviation,
                    result.ExecutionTimeMs);
                lines.Add(line);
            }

            lines.Add(separator);
            return string.Join("\n", lines);
        }
    }
}
