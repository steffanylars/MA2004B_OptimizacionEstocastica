#include "LinearCongruentialGenerator.h"

// Constructor
LinearCongruentialGenerator::LinearCongruentialGenerator(
    unsigned long long m,
    unsigned long long a,
    unsigned long long c,
    unsigned long long seed,
    bool random_mode
)
{
    modulus     = m;
    multiplier  = a;
    increment   = c;
    state       = seed;
    randomMode  = random_mode;
}

// Genera el siguiente número en [0,1]
double LinearCongruentialGenerator::next() {
    if (modulus <= 1) {
        // Evitar división entre cero o módulo inválido.
        // Regresa 0.0 como valor seguro.
        return 0.0;
    }

    if (randomMode) {
        // Modo pseudoaleatorio: Método Congruencial Lineal
        // X_{n+1} = (a * X_n + c) mod m
        state = (multiplier * state + increment) % modulus;
    } else {
        // Modo NO aleatorio: secuencia muy simple y predecible
        // X_{n+1} = (X_n + 1) mod m
        state = (state + 1) % modulus;
    }

    // Normalizamos a [0, 1]
    double u = static_cast<double>(state) / static_cast<double>(modulus - 1);

    return u;
}

// Genera n números y los guarda en outputArray
void LinearCongruentialGenerator::generateMany(double* outputArray, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        outputArray[i] = next();
    }
}