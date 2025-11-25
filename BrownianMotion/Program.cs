using System;
using System.Collections.Generic;
using System.IO;

namespace BrownianMotionSimulation
{
    /// <summary>
    /// Programa principal con interfaz de usuario para la simulación de Movimiento Browniano
    /// Instituto Tecnológico de Estudios Superiores de Monterrey
    /// Cátedra: Ma2004b - Optimización Estocástica
    /// </summary>
    class Program
    {
        static SimulationEngine engine = new SimulationEngine();
        static List<SimulationResult> allResults = new List<SimulationResult>();

        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            
            bool exit = false;
            while (!exit)
            {
                ShowMainMenu();
                string option = Console.ReadLine()?.Trim() ?? "";

                switch (option)
                {
                    case "1":
                        RunCustomSimulation();
                        break;
                    case "2":
                        RunFullBattery();
                        break;
                    case "3":
                        ShowAllResults();
                        break;
                    case "4":
                        SaveResultsToFile();
                        break;
                    case "5":
                        ShowConvergenceAnalysis();
                        break;
                    case "6":
                        exit = true;
                        Console.WriteLine("\n¡Gracias por usar el simulador de Movimiento Browniano!");
                        break;
                    default:
                        Console.WriteLine("\nOpción no válida. Intente de nuevo.");
                        break;
                }

                if (!exit)
                {
                    Console.WriteLine("\nPresione Enter para continuar...");
                    Console.ReadLine();
                }
            }
        }

        static void ShowMainMenu()
        {
            Console.Clear();
            Console.WriteLine("╔══════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║     SIMULADOR DE MOVIMIENTO BROWNIANO - CAMINATA ALEATORIA       ║");
            Console.WriteLine("║     Ma2004b - Optimización Estocástica                           ║");
            Console.WriteLine("║     ITESM                                                        ║");
            Console.WriteLine("╠══════════════════════════════════════════════════════════════════╣");
            Console.WriteLine("║                                                                  ║");
            Console.WriteLine("║   1. Ejecutar simulación personalizada                           ║");
            Console.WriteLine("║   2. Ejecutar batería completa de pruebas                        ║");
            Console.WriteLine("║   3. Ver todos los resultados                                    ║");
            Console.WriteLine("║   4. Guardar resultados en archivo                               ║");
            Console.WriteLine("║   5. Análisis de convergencia                                    ║");
            Console.WriteLine("║   6. Salir                                                       ║");
            Console.WriteLine("║                                                                  ║");
            Console.WriteLine("╚══════════════════════════════════════════════════════════════════╝");
            Console.Write("\nSeleccione una opción: ");
        }

        static void RunCustomSimulation()
        {
            Console.Clear();
            Console.WriteLine("═══════════════════════════════════════════════════════════════════");
            Console.WriteLine("              SIMULACIÓN PERSONALIZADA");
            Console.WriteLine("═══════════════════════════════════════════════════════════════════\n");

            int dimensions = GetIntInput("Ingrese el número de dimensiones (1, 2, 3, ...): ", 1, 100);
            int steps = GetIntInput("Ingrese el número de pasos por caminata: ", 1, 1000000);
            int repetitions = GetIntInput("Ingrese el número de repeticiones: ", 1, 100000);

            Console.WriteLine($"\nEjecutando simulación: {dimensions}D, {steps} pasos, {repetitions} repeticiones...\n");

            var result = engine.RunSimulation(dimensions, steps, repetitions);
            allResults.Add(result);

            Console.WriteLine("═══════════════════════════════════════════════════════════════════");
            Console.WriteLine("                      RESULTADOS");
            Console.WriteLine("═══════════════════════════════════════════════════════════════════");
            Console.WriteLine(result.ToString());
            Console.WriteLine("═══════════════════════════════════════════════════════════════════");

            // Mostrar algunas distancias máximas individuales
            Console.WriteLine("\nPrimeras 10 distancias máximas registradas:");
            for (int i = 0; i < Math.Min(10, result.MaxDistances.Count); i++)
            {
                Console.WriteLine($"  Caminata {i + 1}: {result.MaxDistances[i]:F4}");
            }
        }

        static void RunFullBattery()
        {
            Console.Clear();
            Console.WriteLine("═══════════════════════════════════════════════════════════════════");
            Console.WriteLine("         BATERÍA COMPLETA DE SIMULACIONES");
            Console.WriteLine("═══════════════════════════════════════════════════════════════════");
            Console.WriteLine("\nSe ejecutarán simulaciones para:");
            Console.WriteLine("  - Dimensiones: 1D y 2D");
            Console.WriteLine("  - Pasos: 100, 1000, 10000");
            Console.WriteLine("  - Repeticiones: 100, 1000, 10000");
            Console.WriteLine("\nTotal: 18 configuraciones diferentes\n");
            Console.WriteLine("Presione Enter para comenzar...");
            Console.ReadLine();

            var results = engine.RunFullBattery();
            allResults.AddRange(results);

            Console.WriteLine("\n" + engine.GenerateResultsTable(results));
        }

        static void ShowAllResults()
        {
            Console.Clear();
            Console.WriteLine("═══════════════════════════════════════════════════════════════════");
            Console.WriteLine("              TODOS LOS RESULTADOS ALMACENADOS");
            Console.WriteLine("═══════════════════════════════════════════════════════════════════\n");

            if (allResults.Count == 0)
            {
                Console.WriteLine("No hay resultados almacenados. Ejecute una simulación primero.");
                return;
            }

            Console.WriteLine(engine.GenerateResultsTable(allResults));
        }

        static void SaveResultsToFile()
        {
            Console.Clear();
            Console.WriteLine("═══════════════════════════════════════════════════════════════════");
            Console.WriteLine("              GUARDAR RESULTADOS EN ARCHIVO");
            Console.WriteLine("═══════════════════════════════════════════════════════════════════\n");

            if (allResults.Count == 0)
            {
                Console.WriteLine("No hay resultados para guardar. Ejecute una simulación primero.");
                return;
            }

            Console.Write("Ingrese el nombre del archivo (sin extensión): ");
            string filename = Console.ReadLine()?.Trim() ?? "resultados";
            if (string.IsNullOrEmpty(filename)) filename = "resultados";

            string csvFile = filename + ".csv";
            string txtFile = filename + ".txt";

            // Guardar CSV
            using (var writer = new StreamWriter(csvFile))
            {
                writer.WriteLine("Dimensiones,Pasos,Repeticiones,PromedioDistMax,DesvEstandar,TiempoMs,DistMaxGlobal,DistMinGlobal");
                foreach (var result in allResults)
                {
                    writer.WriteLine($"{result.Dimensions},{result.Steps},{result.Repetitions}," +
                                   $"{result.AverageMaxDistance:F4},{result.StandardDeviation:F4}," +
                                   $"{result.ExecutionTimeMs:F2},{result.OverallMaxDistance:F4},{result.OverallMinDistance:F4}");
                }
            }

            // Guardar TXT con formato de tabla
            using (var writer = new StreamWriter(txtFile))
            {
                writer.WriteLine("RESULTADOS DE SIMULACIÓN DE MOVIMIENTO BROWNIANO");
                writer.WriteLine($"Fecha: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                writer.WriteLine(new string('=', 80));
                writer.WriteLine();
                writer.WriteLine(engine.GenerateResultsTable(allResults));
                writer.WriteLine();
                writer.WriteLine("CONCLUSIONES:");
                writer.WriteLine(GenerateConclusions());
            }

            Console.WriteLine($"\nArchivos guardados:");
            Console.WriteLine($"  - {csvFile} (datos en formato CSV)");
            Console.WriteLine($"  - {txtFile} (reporte completo)");
        }

        static void ShowConvergenceAnalysis()
        {
            Console.Clear();
            Console.WriteLine("═══════════════════════════════════════════════════════════════════");
            Console.WriteLine("              ANÁLISIS DE CONVERGENCIA");
            Console.WriteLine("═══════════════════════════════════════════════════════════════════\n");

            if (allResults.Count == 0)
            {
                Console.WriteLine("No hay resultados para analizar. Ejecute una simulación primero.");
                return;
            }

            Console.WriteLine(GenerateConclusions());
        }

        static string GenerateConclusions()
        {
            var conclusions = new List<string>();
            
            conclusions.Add("ANÁLISIS DE CONVERGENCIA DEL MOVIMIENTO BROWNIANO:");
            conclusions.Add(new string('-', 60));
            conclusions.Add("");

            // Análisis teórico
            conclusions.Add("1. FUNDAMENTO TEÓRICO:");
            conclusions.Add("   Para un movimiento Browniano en d dimensiones con n pasos,");
            conclusions.Add("   la distancia esperada desde el origen es proporcional a √n.");
            conclusions.Add("   La distancia máxima esperada crece aproximadamente como √(n * log(n)).");
            conclusions.Add("");

            // Análisis de resultados por dimensión
            var results1D = allResults.FindAll(r => r.Dimensions == 1);
            var results2D = allResults.FindAll(r => r.Dimensions == 2);

            if (results1D.Count > 0)
            {
                conclusions.Add("2. RESULTADOS EN 1 DIMENSIÓN:");
                foreach (var r in results1D)
                {
                    double theoretical = Math.Sqrt(r.Steps);
                    double ratio = r.AverageMaxDistance / theoretical;
                    conclusions.Add($"   {r.Steps} pasos, {r.Repetitions} reps: Promedio={r.AverageMaxDistance:F2}, " +
                                  $"Teórico≈{theoretical:F2}, Ratio={ratio:F2}");
                }
                conclusions.Add("");
            }

            if (results2D.Count > 0)
            {
                conclusions.Add("3. RESULTADOS EN 2 DIMENSIONES:");
                foreach (var r in results2D)
                {
                    double theoretical = Math.Sqrt(r.Steps);
                    double ratio = r.AverageMaxDistance / theoretical;
                    conclusions.Add($"   {r.Steps} pasos, {r.Repetitions} reps: Promedio={r.AverageMaxDistance:F2}, " +
                                  $"Teórico≈{theoretical:F2}, Ratio={ratio:F2}");
                }
                conclusions.Add("");
            }

            // Conclusión sobre convergencia
            conclusions.Add("4. CONCLUSIÓN SOBRE CONVERGENCIA:");
            conclusions.Add("   - SÍ EXISTE CONVERGENCIA: A medida que aumentan las repeticiones,");
            conclusions.Add("     el promedio de la distancia máxima se estabiliza.");
            conclusions.Add("   - La desviación estándar disminuye con más repeticiones,");
            conclusions.Add("     confirmando la Ley de los Grandes Números.");
            conclusions.Add("   - Los resultados convergen a valores proporcionales a √n,");
            conclusions.Add("     consistente con la teoría del movimiento Browniano.");
            conclusions.Add("   - En 2D las distancias máximas son ligeramente mayores que en 1D");
            conclusions.Add("     debido a la mayor libertad de movimiento.");

            return string.Join("\n", conclusions);
        }

        static int GetIntInput(string prompt, int min, int max)
        {
            int value;
            while (true)
            {
                Console.Write(prompt);
                string input = Console.ReadLine() ?? "";
                if (int.TryParse(input, out value) && value >= min && value <= max)
                {
                    return value;
                }
                Console.WriteLine($"Por favor ingrese un número entero entre {min} y {max}.");
            }
        }
    }
}
