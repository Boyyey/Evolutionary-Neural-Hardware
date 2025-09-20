#ifndef NEAT_H
#define NEAT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include "config.h"

/* Forward declarations */
struct neat_innovation;
struct neat_innovation_table;
struct neat_node;
struct neat_connection;
struct neat_genome;
struct neat_species;
struct neat_population;

typedef struct neat_innovation neat_innovation_t;
typedef struct neat_innovation_table neat_innovation_table_t;
typedef struct neat_node neat_node_t;
typedef struct neat_connection neat_connection_t;
typedef struct neat_genome neat_genome_t;
typedef struct neat_species neat_species_t;
typedef struct neat_population neat_population_t;

/* Node structure */
struct neat_node {
    int id;                     /* Unique identifier for the node */
    neat_node_type_t type;      /* Type of node (input, hidden, output, bias) */
    neat_node_placement_t placement; /* Placement in network (input/hidden/output) */
    neat_activation_type_t activation_type; /* Activation function type */
    double value;               /* Current activation value */
    double bias;                /* Bias value for the node */
    bool active;                /* Whether the node is active in the current evaluation */
    int x_pos;                  /* Used for visualization and network placement */
};

/* Connection structure */
struct neat_connection {
    int innovation;             /* Innovation number */
    int in_node;                /* Input node ID */
    int out_node;               /* Output node ID */
    double weight;              /* Connection weight */
    bool enabled;               /* Whether the connection is active */
};

/* Genome structure - represents a neural network's genetic encoding */
struct neat_genome {
    int id;                     /* Unique identifier for the genome */
    neat_node_t *nodes;         /* Array of nodes */
    size_t node_count;          /* Number of nodes */
    size_t node_capacity;       /* Capacity of nodes array */
    
    neat_connection_t *connections; /* Array of connections */
    size_t connection_count;    /* Number of connections */
    size_t connection_capacity; /* Capacity of connections array */
    
    double fitness;             /* Raw fitness score */
    double adjusted_fitness;    /* Fitness adjusted for species sharing */
    int global_rank;            /* Global rank in population */
    int species_id;             /* ID of the species this genome belongs to */
    
    /* For network evaluation */
    int *evaluation_order;      /* Order in which to evaluate nodes */
    size_t evaluation_order_size; /* Size of evaluation_order */
};

/* Species structure - groups similar genomes */
typedef struct neat_species {
    int id;                     /* Unique identifier for the species */
    struct neat_genome **members;    /* Array of pointers to genomes in this species */
    size_t member_count;        /* Number of genomes in this species */
    size_t member_capacity;     /* Capacity of members array */
    
    struct neat_genome *champion;    /* Best genome in this species */
    double best_fitness;        /* Best fitness in this species */
    double average_fitness;     /* Average fitness of the species */
    int staleness;              /* Generations without improvement */
    int age;                    /* Age of the species */
    
    /* For speciation */
    struct neat_genome *representative; /* Representative genome for compatibility */
} neat_species_t;

/* Population structure - contains all genomes and species */
typedef struct neat_population {
    struct neat_genome **genomes;    /* Array of all genomes */
    size_t genome_count;        /* Number of genomes */
    size_t genome_capacity;     /* Capacity of genomes array */
    
    struct neat_species **species;   /* Array of species */
    size_t species_count;       /* Number of species */
    size_t species_capacity;    /* Capacity of species array */
    
    struct neat_innovation_table *innovation_table; /* Global innovation tracker */
    
    /* Configuration */
    size_t population_size;     /* Target population size */
    int generation;             /* Current generation number */
    int max_fitness_achieved;   /* Best fitness achieved so far */
    
    /* Callback for evaluating genomes */
    double (*evaluate_genome)(struct neat_genome *genome, void *user_data);
    void *evaluate_user_data;   /* User data passed to evaluate_genome */
} neat_population_t;

/* Innovation structure - tracks historical markings */
typedef struct neat_innovation {
    int innovation_id;          /* Global innovation ID */
    int in_node;                /* Input node ID */
    int out_node;               /* Output node ID */
    int innovation_number;      /* Innovation number for this connection */
    bool is_new_node;           /* Whether this innovation created a new node */
    int node_id;                /* If new node, the ID of the new node */
    double weight;              /* Initial weight for the connection */
} neat_innovation_t;

/* Innovation table - tracks all innovations in the population */
struct neat_innovation_table {
    neat_innovation_t *innovations; /* Array of innovations */
    size_t count;               /* Number of innovations */
    size_t capacity;            /* Capacity of innovations array */
    int next_innovation;        /* Next available innovation number */
    int next_node_id;           /* Next available node ID */
    int next_species_id;        /* Next available species ID */
};

/* Function prototypes */
/* Memory management */
void* neat_malloc(size_t size);
void neat_free(void *ptr);
void* neat_calloc(size_t nmemb, size_t size);
void* neat_realloc(void *ptr, size_t size);

/* Node functions */
neat_node_t neat_create_node(int id, neat_node_type_t type, neat_node_placement_t placement);
void neat_free_node(neat_node_t *node);

/* Connection functions */
neat_connection_t neat_create_connection(int innovation, int in_node, int out_node, 
                                         double weight, bool enabled);
void neat_free_connection(neat_connection_t *conn);

/* Genome functions */
neat_genome_t* neat_load_genome(FILE* fp);
void neat_free_genome(neat_genome_t *genome);
neat_genome_t* neat_clone_genome(const neat_genome_t *genome);
int neat_add_node(neat_genome_t *genome, neat_node_type_t type, 
                 neat_node_placement_t placement);
int neat_add_connection(neat_genome_t *genome, int in_node, int out_node, 
                       double weight, bool enabled);
void neat_mutate(neat_genome_t *genome, neat_innovation_table_t *table);
void neat_mutate_weights(neat_genome_t *genome);
void neat_mutate_add_connection(neat_genome_t *genome, neat_innovation_table_t *table);
void neat_mutate_add_node(neat_genome_t *genome, neat_innovation_table_t *table);
void neat_mutate_toggle_connection(neat_genome_t *genome);
void neat_mutate_activation(neat_genome_t *genome);
neat_genome_t* neat_crossover(const neat_genome_t *parent1, const neat_genome_t *parent2);
double neat_compatibility_distance(const neat_genome_t *genome1, 
                                 const neat_genome_t *genome2);
void neat_evaluate(neat_genome_t *genome, const double *inputs, double *outputs);
void neat_update_network(neat_genome_t *genome);

/* Species functions */
neat_species_t* neat_create_species(int id);
void neat_free_species(neat_species_t *species);
void neat_add_genome_to_species(neat_species_t *species, neat_genome_t *genome);
void neat_remove_genome_from_species(neat_species_t *species, const neat_genome_t *genome);
void neat_adjust_fitness(neat_species_t *species);

/* Population functions */
neat_population_t* neat_create_population(size_t input_size, size_t output_size, 
                                         size_t population_size);
void neat_free_population(neat_population_t *pop);
void neat_evolve(neat_population_t *pop);
void neat_speciate(neat_population_t *pop);
void neat_remove_stale_species(neat_population_t *pop);
void neat_remove_weak_species(neat_population_t *pop);
void neat_reproduce(neat_population_t *pop);

/* Innovation table functions */
neat_innovation_table_t* neat_create_innovation_table(void);
void neat_free_innovation_table(neat_innovation_table_t *table);
int neat_get_innovation(neat_innovation_table_t *table, int in_node, int out_node, 
                       bool is_new_node, int node_id, double weight);

/* Utility functions */
double neat_sigmoid(double x);
double neat_tanh(double x);
double neat_relu(double x);
double neat_leaky_relu(double x);
double neat_linear(double x);
double neat_step(double x);
double neat_softsign(double x);
double neat_sin(double x);
double neat_gaussian(double x);
double neat_abs(double x);

double neat_random_uniform(double min, double max);
double neat_random_normal(double mean, double stddev);
int neat_random_int(int min, int max);

/* Activation functions */
activation_func_t neat_get_activation_function(neat_activation_type_t type);
const char* neat_activation_name(neat_activation_type_t type);

/* Activation functions and types are defined in config.h */

#endif /* NEAT_H */
