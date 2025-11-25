# Simulador de Movimiento Browniano
## Ma2004b - Optimización Estocástica - ITESM

### Descripción
Este programa simula el Movimiento Browniano (caminata aleatoria) en n dimensiones.
Una partícula comienza en el origen y en cada paso se mueve +1 o -1 en una dimensión
seleccionada aleatoriamente.

### Archivos del Proyecto

1. **BrownianMotion.cs** - Clase principal que implementa la simulación
   - Constructor: `BrownianMotion(int dimensions, int steps, int? seed)`
   - Métodos: `ExecuteWalk()`, `Step()`, `GetDistanceFromOrigin()`, `Reset()`

2. **SimulationResult.cs** - Clase para almacenar resultados
   - Propiedades: Dimensions, Steps, Repetitions, MaxDistances, ExecutionTimeMs
   - Propiedades calculadas: AverageMaxDistance, StandardDeviation, OverallMaxDistance

3. **SimulationEngine.cs** - Motor de simulación
   - `RunSimulation(dimensions, steps, repetitions)` - Ejecuta una simulación
   - `RunFullBattery()` - Ejecuta todas las combinaciones requeridas
   - `GenerateResultsTable()` - Genera tabla formateada de resultados

4. **Program.cs** - Interfaz de usuario con menú interactivo
   - Simulación personalizada
   - Batería completa de pruebas
   - Guardado de resultados en CSV y TXT
   - Análisis de convergencia

5. **BrownianMotionSimulation.csproj** - Archivo de proyecto .NET

### Requisitos
- .NET 6.0 SDK o superior

### Compilación y Ejecución

```bash
# Compilar el proyecto
dotnet build

# Ejecutar el programa
dotnet run

# O crear ejecutable
dotnet publish -c Release -o ./publish
```

### Uso del Programa

El programa presenta un menú interactivo con las siguientes opciones:

1. **Simulación personalizada**: Permite especificar dimensiones, pasos y repeticiones
2. **Batería completa**: Ejecuta automáticamente todas las combinaciones:
   - Dimensiones: 1D y 2D
   - Pasos: 100, 1000, 10000
   - Repeticiones: 100, 1000, 10000
3. **Ver resultados**: Muestra tabla con todos los resultados almacenados
4. **Guardar resultados**: Exporta a archivos CSV y TXT
5. **Análisis de convergencia**: Muestra conclusiones sobre la convergencia

### Parámetros de Entrada
- **Dimensiones**: Número de dimensiones del sistema de coordenadas (1, 2, 3, ...)
- **Pasos**: Número de pasos por caminata aleatoria
- **Repeticiones**: Número de veces que se repite el experimento

### Parámetros de Salida
- Distancias máximas de cada caminata
- Promedio de distancia máxima
- Desviación estándar
- Tiempo de ejecución en milisegundos
- Análisis de convergencia

### Fundamento Teórico
Para un movimiento Browniano en d dimensiones con n pasos:
- La distancia esperada desde el origen es proporcional a √n
- La desviación estándar disminuye con más repeticiones (Ley de Grandes Números)
- Los resultados convergen a valores estables conforme aumentan las repeticiones
