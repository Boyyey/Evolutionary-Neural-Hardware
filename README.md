# NEAT (NeuroEvolution of Augmenting Topologies) in C

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://codecov.io/gh/boyyey/evolutionary-neural-hardware/branch/main/graph/badge.svg)](https://codecov.io/gh/boyyey/evolutionary-neural-hardware)

A high-performance, feature-rich implementation of the NeuroEvolution of Augmenting Topologies (NEAT) algorithm in pure C, designed for both research and production use.

## 🌟 Features

- **Advanced NEAT Implementation**: Complete implementation of the NEAT algorithm with support for complex topologies
- **HyperNEAT & CPPN**: Support for HyperNEAT and Compositional Pattern-Producing Networks (CPPN)
- **Parallel Evolution**: Multi-threaded fitness evaluation using pthreads and OpenMP
- **SIMD Acceleration**: Optimized math operations using AVX2, SSE4.2, and FMA instructions
- **Visualization**: Real-time visualization of neural networks and evolution using SDL2
- **Novelty Search**: Implementation of novelty search and other advanced evolutionary strategies
- **Multi-objective Optimization**: Support for multi-objective optimization with Pareto front visualization
- **Interactive Tools**: Interactive genome editor and visualization tools
- **Extensible Architecture**: Modular design for easy extension and customization
- **Cross-platform**: Works on Linux, macOS, and Windows (with MinGW)

## 🚀 Getting Started

### Prerequisites

- C compiler with C11 support (GCC, Clang, or MSVC)
- CMake 3.12+
- SDL2 and SDL2_ttf development libraries
- (Optional) Doxygen for building documentation

### Installation

#### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake libsdl2-dev libsdl2-ttf-dev

# Clone the repository
git clone https://github.com/boyyey/evolutionary-neural-hardware.git

# Build the project
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

#### Windows (MinGW)

1. Install MSYS2: https://www.msys2.org/
2. Open MSYS2 MinGW 64-bit terminal and run:
   ```bash
   pacman -Syu
   pacman -S --needed git mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake \
       mingw-w64-x86_64-SDL2 mingw-w64-x86_64-SDL2_ttf \
       mingw-w64-x86_64-toolchain
   ```
3. Clone and build the project as shown in the Linux instructions

### Building with Make

```bash
# Build in release mode (default)
make

# Build with debug symbols and sanitizers
make debug

# Build and run tests
make test

# Build examples
make examples

# Build and run benchmarks
make bench

# Generate documentation (requires Doxygen)
make docs

# Clean build artifacts
make clean
```

## 🧠 Core Components

- **neat.c/h**: Core NEAT algorithm implementation
- **genome.c/h**: Genome representation and genetic operations
- **population.c/h**: Population management and speciation
- **species.c/h**: Species handling and fitness sharing
- **network.c/h**: Neural network evaluation
- **innovation.c/h**: Innovation tracking and historical markings
- **utils.c/h**: Utility functions and helpers
- **rng.c/h**: Fast, high-quality random number generation
- **simd_math.c/h**: SIMD-accelerated math operations
- **visualization.c/h**: Interactive visualization using SDL2
- **hyperneat.c/h**: HyperNEAT and CPPN implementation
- **novelty.c/h**: Novelty search implementation
- **multiobjective.c/h**: Multi-objective optimization
- **io.c/h**: File I/O and serialization

## 📊 Visualization

The project includes a powerful visualization system that can display:

- Real-time evolution of neural networks
- Fitness and population statistics
- Neural network topologies
- Species distribution
- Novelty search metrics
- Multi-objective optimization fronts

![Visualization Demo](docs/images/neat_visualization.png)

## 🧪 Examples

### XOR Problem

```c
#include <neat.h>
#include <stdio.h>

int main() {
    // Create a NEAT configuration
    neat_config_t config = neat_get_default_config();
    config.population_size = 150;
    config.num_inputs = 2;
    config.num_outputs = 1;
    
    // Create a population
    neat_population_t* pop = neat_create_population(&config);
    
    // Evolution loop
    for (int gen = 0; gen < 100; gen++) {
        // Evaluate each genome
        for (size_t i = 0; i < pop->genome_count; i++) {
            neat_genome_t* genome = pop->genomes[i];
            float fitness = evaluate_xor(genome);  // Your fitness function
            genome->fitness = fitness;
        }
        
        // Evolve to next generation
        neat_evolve(pop);
        
        // Print progress
        printf("Generation %d: Best fitness = %.4f\n", 
               gen, neat_get_best_fitness(pop));
    }
    
    // Cleanup
    neat_free_population(pop);
    return 0;
}
```

### HyperNEAT

```c
#include <hyperneat.h>

void hyperneat_demo() {
    // Create a HyperNEAT configuration
    hyperneat_config_t config = hyperneat_get_default_config();
    config.cppn_inputs = 4;  // x1, y1, x2, y2
    config.cppn_outputs = 1; // Weight
    
    // Create a HyperNEAT instance
    hyperneat_t* hyperneat = hyperneat_create(&config);
    
    // Train the HyperNEAT network
    hyperneat_train(hyperneat, training_data, num_samples);
    
    // Query the network
    float weight = hyperneat_query(hyperneat, x1, y1, x2, y2);
    
    // Cleanup
    hyperneat_free(hyperneat);
}
```

## 📚 Documentation

Detailed documentation is available in the `docs` directory. To build the documentation:

```bash
# Install Doxygen (if not already installed)
sudo apt-get install -y doxygen graphviz

# Generate documentation
make docs

# Open documentation in browser
xdg-open docs/html/index.html  # Linux
start docs/html/index.html     # Windows
```

## 📊 Benchmarks

Performance benchmarks are available in the `benchmarks` directory. To run them:

```bash
make bench
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting Topologies. Evolutionary Computation, 10(2), 99-127.
2. Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks. Artificial Life, 15(2), 185-212.
3. Lehman, J., & Stanley, K. O. (2011). Abandoning Objectives: Evolution Through the Search for Novelty Alone. Evolutionary Computation, 19(2), 189-223.

---

<div align="center">
  <p>Made with ❤️ and C</p>
  <p>© 2023 NEAT-C Contributors</p>
</div>
