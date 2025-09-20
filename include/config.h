#ifndef NEAT_CONFIG_H
#define NEAT_CONFIG_H

/* Memory allocation */
#define NEAT_DEFAULT_ALLOC_SIZE 128
#define NEAT_GROWTH_FACTOR 2

/* Network parameters */
#define NEAT_MAX_NODES 1000000
#define NEAT_MAX_CONNECTIONS 1000000
#define NEAT_MAX_ACTIVATION_FUNCS 10
#define NEAT_MAX_INPUTS 1000
#define NEAT_MAX_OUTPUTS 100

/* Evolution parameters */
#define NEAT_DEFAULT_POPULATION_SIZE 150
#define NEAT_DEFAULT_ELITISM 2
#define NEAT_DEFAULT_STAGNATION_LIMIT 15

/* Speciation parameters */
#define NEAT_COMPATIBILITY_THRESHOLD 3.0
#define NEAT_COMPATIBILITY_MOD 0.3
#define NEAT_EXCESS_COEFF 1.0
#define NEAT_DISJOINT_COEFF 1.0
#define NEAT_WEIGHT_COEFF 0.4

/* Mutation rates */
#define NEAT_MUTATE_CONNECTION_RATE 0.25
#define NEAT_MUTATE_NODE_RATE 0.03
#define NEAT_MUTATE_LINK_RATE 0.05
#define NEAT_MUTATE_WEIGHT_RATE 0.8
#define NEAT_MUTATE_TOGGLE_LINK_RATE 0.1
#define NEAT_MUTATE_ACTIVATION_RATE 0.1

/* Weight mutation parameters */
#define NEAT_WEIGHT_MUTATION_POWER 2.5
#define NEAT_WEIGHT_RANDOM_STRENGTH 1.0

/* Species parameters */
#define NEAT_SPECIES_DROPOFF_AGE 15
#define NEAT_SPECIES_AGE_PENALTY 0.5

/* Activation functions */
typedef double (*activation_func_t)(double);

typedef enum {
    NEAT_ACTIVATION_SIGMOID,
    NEAT_ACTIVATION_TANH,
    NEAT_ACTIVATION_RELU,
    NEAT_ACTIVATION_LEAKY_RELU,
    NEAT_ACTIVATION_LINEAR,
    NEAT_ACTIVATION_STEP,
    NEAT_ACTIVATION_SOFTSIGN,
    NEAT_ACTIVATION_SIN,
    NEAT_ACTIVATION_GAUSSIAN,
    NEAT_ACTIVATION_ABS
} neat_activation_type_t;

/* Node types */
typedef enum {
    NEAT_NODE_INPUT,
    NEAT_NODE_HIDDEN,
    NEAT_NODE_OUTPUT,
    NEAT_NODE_BIAS
} neat_node_type_t;

/* Node placement in network */
typedef enum {
    NEAT_PLACEMENT_INPUT,
    NEAT_PLACEMENT_HIDDEN,
    NEAT_PLACEMENT_OUTPUT
} neat_node_placement_t;

#endif /* NEAT_CONFIG_H */
