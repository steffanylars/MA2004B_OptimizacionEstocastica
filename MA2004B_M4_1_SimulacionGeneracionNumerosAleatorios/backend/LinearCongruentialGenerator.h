#ifndef LINEAR_CONGRUENTIAL_GENERATOR_H
#define LINEAR_CONGRUENTIAL_GENERATOR_H

#include <cstddef> // size_t

class LinearCongruentialGenerator {
public:
    // Parámetros del generador 
    unsigned long long modulus;       // m
    unsigned long long multiplier;    // a
    unsigned long long increment;     // c
    unsigned long long state;         // Xn (semilla / estado actual)
    bool randomMode;                  // true = MCL, false = no aleatorio

    // Constructor
    LinearCongruentialGenerator(
        unsigned long long m,
        unsigned long long a,
        unsigned long long c,
        unsigned long long seed,
        bool random_mode
    );

    // Genera el siguiente número en [0,1]
    double next();

    // Genera n números y los guarda en un arreglo externo
    void generateMany(double* outputArray, std::size_t n);
};

#endif 