#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "LinearCongruentialGenerator.h"

int main(int argc, char* argv[]) {
    // Esperamos:
    // mcl m a c seed n modo
    // modo: 1 = pseudoaleatorio (MCL), 0 = no aleatorio
    if (argc != 7) {
        std::cerr << "Uso: ./mcl m a c seed n modo\n";
        std::cerr << "  m    : modulo (entero > 1)\n";
        std::cerr << "  a    : multiplicador\n";
        std::cerr << "  c    : incremento\n";
        std::cerr << "  seed : semilla X0\n";
        std::cerr << "  n    : cantidad de numeros a generar\n";
        std::cerr << "  modo : 1 = pseudoaleatorio (MCL), 0 = no aleatorio\n";
        return 1;
    }

    try {
        unsigned long long m    = std::stoull(argv[1]);
        unsigned long long a    = std::stoull(argv[2]);
        unsigned long long c    = std::stoull(argv[3]);
        unsigned long long seed = std::stoull(argv[4]);
        int n                   = std::stoi(argv[5]);
        bool randomMode         = (std::stoi(argv[6]) == 1);

        if (m <= 1 || n <= 0) {
            std::cerr << "Error: m debe ser > 1 y n > 0.\n";
            return 1;
        }

        LinearCongruentialGenerator gen(m, a, c, seed, randomMode);

        // Generamos los valores
        std::vector<double> values(n);
        gen.generateMany(values.data(), values.size());

        // Escribimos CSV en el directorio actual
        std::ofstream out("output.csv");
        if (!out.is_open()) {
            std::cerr << "Error: no se pudo abrir output.csv para escritura.\n";
            return 1;
        }

        out << "index,value\n";
        for (int i = 0; i < n; ++i) {
            out << i << "," << values[i] << "\n";
        }
        out.close();

        return 0;
    catch (const std::exception& e) {
        std::cerr << "Error de conversion de argumentos: " << e.what() << "\n";
        return 1;
    }
}