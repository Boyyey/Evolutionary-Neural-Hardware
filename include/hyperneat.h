#ifndef HYPERNEAT_H
#define HYPERNEAT_H

#include "neat.h"
#include <stddef.h>

/* Forward declarations */
struct hyperneat_population;
typedef struct hyperneat_population hyperneat_population_t;

/* Substrate node structure */
typedef struct {
    float x, y, z;                  /* 3D coordinates */
    float* activations;             /* Activation values */
    int layer;                      /* Layer index */
    int node_type;                  /* Node type (0=input, 1=hidden, 2=output, 3=bias) */
} substrate_node_t;

/* Substrate connection structure */
typedef struct {
    int from_node;                  /* Source node index */
    int to_node;                    /* Target node index */
    float weight;                   /* Connection weight */
    int enabled;                    /* Whether the connection is enabled */
} substrate_connection_t;

/* Substrate structure */
typedef struct {
    substrate_node_t* nodes;        /* Array of nodes */
    int node_count;                 /* Number of nodes */
    substrate_connection_t* connections; /* Array of connections */
    int connection_count;           /* Number of connections */
    int* layer_sizes;               /* Number of nodes in each layer */
    int layer_count;                /* Number of layers */
    float min_x, max_x, min_y, max_y, min_z, max_z; /* Bounding box */
} substrate_t;

/* HyperNEAT individual structure */
typedef struct {
    neat_genome_t* cppn;            /* The CPPN that defines the mapping */
    substrate_t* substrate;         /* The substrate network */
    float fitness;                  /* Fitness of the individual */
    float* objectives;              /* For multi-objective optimization */
    int objective_count;            /* Number of objectives */
    float* novelty;                 /* Novelty score */
    int novelty_dimensions;         /* Dimensionality of novelty space */
    float* activation_pattern;      /* Activation pattern for visualization */
    int pattern_size;               /* Size of activation pattern */
} hyperneat_individual_t;

/* HyperNEAT configuration structure */
typedef struct {
    /* Substrate configuration */
    int substrate_input_width;      /* Width of input substrate */
    int substrate_input_height;     /* Height of input substrate */
    int substrate_output_width;     /* Width of output substrate */
    int substrate_output_height;    /* Height of output substrate */
    int substrate_hidden_layers;    /* Number of hidden substrate layers */
    
    /* CPPN configuration */
    int cppn_inputs;                /* Number of CPPN inputs (x1,y1,x2,y2, etc.) */
    int cppn_outputs;               /* Number of CPPN outputs (weights, etc.) */
    
    /* HyperNEAT parameters */
    float weight_range;             /* Range for weight values */
    float activation_prob;          /* Probability of activating a connection */
    float weight_mutate_power;      /* Power of weight mutations */
    float weight_mutate_rate;       /* Rate of weight mutations */
    float weight_replace_rate;      /* Rate of weight replacement */
    
    /* Substrate connection parameters */
    float connection_density;       /* Density of connections in the substrate */
    int max_weight;                 /* Maximum absolute weight value */
    
    /* Compatibility parameters */
    float compatibility_threshold;  /* Compatibility threshold for speciation */
    float compatibility_change;     /* Rate of compatibility threshold change */
    
    /* Novelty search parameters */
    int k_nearest;                  /* Number of nearest neighbors for novelty */
    float novelty_threshold;        /* Threshold for adding to archive */
    
    /* Visualization parameters */
    int visualize;                  /* Whether to visualize the process */
    int visualization_interval;     /* How often to update visualization */
} hyperneat_config_t;

/* HyperNEAT population structure */
struct hyperneat_population {
    hyperneat_individual_t* individuals;    /* Array of individuals */
    int population_size;                    /* Number of individuals */
    int generation;                         /* Current generation */
    
    hyperneat_config_t config;              /* Configuration */
    neat_population_t* cppn_population;     /* Population of CPPNs */
    
    /* Substrates */
    substrate_t* input_substrate;           /* Input substrate */
    substrate_t* output_substrate;          /* Output substrate */
    substrate_t** hidden_substrates;        /* Hidden substrates */
    int hidden_substrate_count;             /* Number of hidden substrates */
    
    /* Novelty search */
    float** archive;                        /* Archive of novel individuals */
    int archive_size;                       /* Current size of archive */
    int archive_capacity;                   /* Maximum size of archive */
    
    /* Statistics */
    float* best_fitness_history;            /* Best fitness over generations */
    float* avg_fitness_history;             /* Average fitness over generations */
    int* species_count_history;             /* Species count over generations */
    int max_generations;                    /* Maximum number of generations */
    
};

/* Function declarations */

/* Initialization and cleanup */
hyperneat_config_t hyperneat_get_default_config(void);
hyperneat_population_t* hyperneat_create_population(const hyperneat_config_t* config, size_t population_size);
void hyperneat_free_population(hyperneat_population_t* pop);

/* Substrate operations */
substrate_t substrate_create(int num_layers, const int* layer_sizes, 
                           float min_x, float max_x, 
                           float min_y, float max_y, 
                           float min_z, float max_z);
void substrate_free(substrate_t* substrate);
void substrate_connect_layers(substrate_t* substrate, int from_layer, int to_layer, 
                             float density, int max_connections);

/* Individual operations */
hyperneat_individual_t* hyperneat_create_individual(neat_genome_t* cppn, 
                                                   const hyperneat_config_t* config);
void hyperneat_free_individual(hyperneat_individual_t* individual);
void hyperneat_build_phenotype(hyperneat_individual_t* individual, 
                              const hyperneat_config_t* config);
void hyperneat_activate(hyperneat_individual_t* individual, const float* inputs, 
                       float* outputs);

/* Population operations */
void hyperneat_evolve(hyperneat_population_t* pop, 
                     float (*fitness_function)(hyperneat_individual_t*));
void hyperneat_evaluate(hyperneat_population_t* pop, 
                       float (*fitness_function)(hyperneat_individual_t*));
void hyperneat_speciate(hyperneat_population_t* pop);
void hyperneat_reproduce(hyperneat_population_t* pop);

/* Novelty search */
void hyperneat_update_novelty(hyperneat_individual_t* individual, 
                             hyperneat_population_t* pop);
void hyperneat_add_to_archive(hyperneat_individual_t* individual, 
                             hyperneat_population_t* pop);
float hyperneat_calculate_novelty(hyperneat_individual_t* individual, 
                                 const hyperneat_individual_t** population,
                                 size_t population_size,
                                 size_t k_nearest);

/* Visualization */
void hyperneat_visualize_individual(hyperneat_individual_t* individual, 
                                   const char* filename);
void hyperneat_visualize_population(hyperneat_population_t* pop, 
                                   const char* directory);
void hyperneat_visualize_substrate(const substrate_t* substrate, 
                                  const char* filename);

/* Utility functions */
float hyperneat_sigmoid(float x);
float hyperneat_gaussian(float x, float y, float sigma);
float hyperneat_distance(const float* a, const float* b, int n);
void hyperneat_mutate(hyperneat_individual_t* individual, 
                     const hyperneat_config_t* config);
void hyperneat_crossover(hyperneat_individual_t* child, 
                        const hyperneat_individual_t* parent1, 
                        const hyperneat_individual_t* parent2);

/* Multi-objective optimization */
void hyperneat_nsga2_selection(hyperneat_population_t* pop);
void hyperneat_calculate_pareto_front(hyperneat_individual_t** front, 
                                     int* front_size, 
                                     hyperneat_individual_t** individuals, 
                                     int count);

/* Save/load */
int hyperneat_save_individual(const hyperneat_individual_t* individual, 
                             const char* filename);
int hyperneat_save_population(const hyperneat_population_t* pop, 
                             const char* filename);
hyperneat_individual_t* hyperneat_load_individual(const char* filename);
hyperneat_population_t* hyperneat_load_population(const char* filename, 
                                                 const hyperneat_config_t* config);

/* Advanced features */
void hyperneat_evolve_substrate(hyperneat_population_t* pop);
void hyperneat_evolve_plasticity(hyperneat_individual_t* individual, 
                                const hyperneat_config_t* config);
void hyperneat_evolve_learning_rule(hyperneat_individual_t* individual, 
                                   const hyperneat_config_t* config);

/* Interactive evolution */
typedef float (*hyperneat_interactive_fitness_t)(hyperneat_individual_t*, void*);
void hyperneat_interactive_evolution(hyperneat_population_t* pop, 
                                    hyperneat_interactive_fitness_t fitness_func, 
                                    void* userdata);

/* Coevolution */
struct hyperneat_population_t; /* Forward declaration */
void hyperneat_coevolve(hyperneat_population_t** populations, 
                        int num_populations, 
                        int generations);

/* Modularity and pattern generation */
void hyperneat_evoke_pattern(hyperneat_individual_t* individual, 
                            float* pattern, 
                            int pattern_size);
void hyperneat_evoke_modularity(hyperneat_individual_t* individual, 
                               float* module1, int size1, 
                               float* module2, int size2);

/* Developmental encoding */
void hyperneat_developmental_encoding(hyperneat_individual_t* individual, 
                                     int num_phases);

/* Adaptive parameters */
void hyperneat_adaptive_mutation(hyperneat_population_t* pop);
void hyperneat_adaptive_crossover(hyperneat_population_t* pop);

/* Ensemble methods */
void hyperneat_create_ensemble(hyperneat_individual_t** individuals, 
                              int count, 
                              hyperneat_individual_t* ensemble);

/* Transfer learning */
void hyperneat_transfer_learning(hyperneat_individual_t* source, 
                                hyperneat_individual_t* target, 
                                float transfer_rate);

/* Multi-task learning */
void hyperneat_multitask_learning(hyperneat_individual_t* individual, 
                                 float** inputs, 
                                 float** outputs, 
                                 int num_tasks);

/* Memory and attention mechanisms */
void hyperneat_add_memory(hyperneat_individual_t* individual, 
                         int memory_size);
void hyperneat_add_attention(hyperneat_individual_t* individual, 
                            int attention_heads);

/* Neuroevolution of augmenting topologies with memory (NEAT-M) */
void hyperneat_add_memory_module(hyperneat_individual_t* individual, 
                                int memory_size);

/* HyperNEAT with indirect encoding extensions */
void hyperneat_indirect_encoding(hyperneat_individual_t* individual, 
                                int encoding_depth);

/* Meta-learning extensions */
void hyperneat_metalearning(hyperneat_population_t* pop, 
                           float (*meta_fitness)(hyperneat_individual_t*));

/* Visualization extensions */
void hyperneat_visualize_development(hyperneat_individual_t* individual, 
                                    const char* directory);
void hyperneat_visualize_evolution(hyperneat_population_t* pop, 
                                  const char* directory);

#endif /* HYPERNEAT_H */
