using System;

namespace BrownianMotionSimulation
{
    /// <summary>
    /// Clase que implementa la simulación del Movimiento Browniano
    /// en n dimensiones con pasos discretos unitarios.
    /// </summary>
    public class BrownianMotion
    {
        private readonly int _dimensions;
        private readonly int _steps;
        private readonly Random _random;
        private double[] _position;

        public int Dimensions => _dimensions;
        public int Steps => _steps;

        /// <summary>
        /// Constructor de la clase BrownianMotion
        /// </summary>
        /// <param name="dimensions">Número de dimensiones del sistema de coordenadas</param>
        /// <param name="steps">Número de pasos a realizar en la caminata</param>
        /// <param name="seed">Semilla opcional para el generador aleatorio</param>
        public BrownianMotion(int dimensions, int steps, int? seed = null)
        {
            if (dimensions < 1)
                throw new ArgumentException("Las dimensiones deben ser al menos 1", nameof(dimensions));
            if (steps < 1)
                throw new ArgumentException("Los pasos deben ser al menos 1", nameof(steps));

            _dimensions = dimensions;
            _steps = steps;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            _position = new double[dimensions];
        }

        /// <summary>
        /// Reinicia la posición de la partícula al origen
        /// </summary>
        public void Reset()
        {
            _position = new double[_dimensions];
        }

        /// <summary>
        /// Calcula la distancia Euclidiana desde el origen
        /// </summary>
        /// <returns>Distancia desde el origen</returns>
        public double GetDistanceFromOrigin()
        {
            double sumSquares = 0;
            for (int i = 0; i < _dimensions; i++)
            {
                sumSquares += _position[i] * _position[i];
            }
            return Math.Sqrt(sumSquares);
        }

        /// <summary>
        /// Ejecuta un paso del movimiento Browniano
        /// Selecciona una dimensión al azar y se mueve +1 o -1 con probabilidad 0.5
        /// </summary>
        public void Step()
        {
            // Seleccionar dimensión al azar
            int dimension = _random.Next(_dimensions);
            
            // Mover +1 o -1 con probabilidad 0.5
            int direction = _random.NextDouble() < 0.5 ? -1 : 1;
            
            _position[dimension] += direction;
        }

        /// <summary>
        /// Ejecuta una caminata completa y retorna la distancia máxima alcanzada
        /// </summary>
        /// <returns>Distancia máxima desde el origen durante la caminata</returns>
        public double ExecuteWalk()
        {
            Reset();
            double maxDistance = 0;

            for (int i = 0; i < _steps; i++)
            {
                Step();
                double currentDistance = GetDistanceFromOrigin();
                if (currentDistance > maxDistance)
                {
                    maxDistance = currentDistance;
                }
            }

            return maxDistance;
        }

        /// <summary>
        /// Obtiene la posición actual de la partícula
        /// </summary>
        /// <returns>Array con las coordenadas actuales</returns>
        public double[] GetCurrentPosition()
        {
            return (double[])_position.Clone();
        }
    }
}
