#include "../include/hyperneat.h"
#include "../include/neat.h"
#include "../include/simd_math.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdio.h>  // For FILE, fopen, fclose, etc.

/* Node type definitions */
#define NEAT_NODE_INPUT  0
#define NEAT_NODE_HIDDEN 1
#define NEAT_NODE_OUTPUT 2
#define NEAT_NODE_BIAS   3

/* Default HyperNEAT configuration */
hyperneat_config_t hyperneat_get_default_config(void) {
    hyperneat_config_t config = {0};
    
    /* Substrate configuration */
    config.substrate_input_width = 3;
    config.substrate_input_height = 3;
    config.substrate_output_width = 2;
    config.substrate_output_height = 2;
    config.substrate_hidden_layers = 1;
    
    /* CPPN configuration */
    config.cppn_inputs = 4;  /* x1, y1, x2, y2 */
    config.cppn_outputs = 1; /* weight */
    
    /* HyperNEAT parameters */
    config.weight_range = 6.0f;
    config.activation_prob = 0.7f;
    config.weight_mutate_power = 2.5f;
    config.weight_mutate_rate = 0.8f;
    config.weight_replace_rate = 0.1f;
    
    /* Substrate connection parameters */
    config.connection_density = 0.3f;
    config.max_weight = 8.0f;
    
    /* Compatibility parameters */
    config.compatibility_threshold = 3.0f;
    config.compatibility_change = 0.3f;
    
    /* Novelty search parameters */
    config.k_nearest = 15;
    config.novelty_threshold = 6.0f;
    
    /* Visualization parameters */
    config.visualize = 1;
    config.visualization_interval = 5;
    
    return config;
}

/* Create a new substrate node */
substrate_node_t substrate_node_create(float x, float y, float z, int layer, int node_type) {
    substrate_node_t node;
    node.x = x;
    node.y = y;
    node.z = z;
    node.layer = layer;
    node.node_type = node_type;
    node.activations = NULL;
    return node;
}

/* Free a substrate node */
void substrate_node_free(substrate_node_t* node) {
    if (node) {
        if (node->activations) {
            free(node->activations);
            node->activations = NULL;
        }
    }
}

/* Create a new substrate connection */
substrate_connection_t substrate_connection_create(int from_node, int to_node, float weight, int enabled) {
    substrate_connection_t conn;
    conn.from_node = from_node;
    conn.to_node = to_node;
    conn.weight = weight;
    conn.enabled = enabled;
    return conn;
}

/* Initialize a substrate with the given dimensions */
substrate_t substrate_create(int num_layers, const int* layer_sizes, 
                            float min_x, float max_x, 
                            float min_y, float max_y, 
                            float min_z, float max_z) {
    substrate_t substrate = {0};
    
    /* Calculate total number of nodes */
    int total_nodes = 0;
    for (int i = 0; i < num_layers; i++) {
        total_nodes += layer_sizes[i];
    }
    
    /* Allocate memory for nodes */
    substrate.nodes = (substrate_node_t*)calloc(total_nodes, sizeof(substrate_node_t));
    if (!substrate.nodes) {
        return substrate;  /* Return empty substrate on allocation failure */
    }
    substrate.node_count = total_nodes;
    
    /* Allocate memory for layer sizes */
    substrate.layer_sizes = (int*)calloc(num_layers, sizeof(int));
    if (!substrate.layer_sizes) {
        free(substrate.nodes);
        return substrate;
    }
    memcpy(substrate.layer_sizes, layer_sizes, num_layers * sizeof(int));
    substrate.layer_count = num_layers;
    
    /* Initialize nodes */
    int node_index = 0;
    float z_step = (max_z - min_z) / (num_layers > 1 ? (num_layers - 1) : 1);
    
    for (int l = 0; l < num_layers; l++) {
        int nodes_in_layer = layer_sizes[l];
        float z = min_z + l * z_step;
        
        /* Determine node type */
        int node_type;
        if (l == 0) node_type = NEAT_NODE_INPUT;
        else if (l == num_layers - 1) node_type = NEAT_NODE_OUTPUT;
        else node_type = NEAT_NODE_HIDDEN;
        
        /* Position nodes in a grid */
        int grid_size = (int)ceilf(sqrtf((float)nodes_in_layer));
        float x_step = (max_x - min_x) / (grid_size + 1);
        float y_step = (max_y - min_y) / (grid_size + 1);
        
        for (int i = 0; i < nodes_in_layer; i++) {
            int row = i / grid_size;
            int col = i % grid_size;
            float x = min_x + (col + 1) * x_step;
            float y = min_y + (row + 1) * y_step;
            
            substrate.nodes[node_index] = substrate_node_create(x, y, z, l, node_type);
            node_index++;
        }
    }
    
    /* Initialize bounds */
    substrate.min_x = min_x;
    substrate.max_x = max_x;
    substrate.min_y = min_y;
    substrate.max_y = max_y;
    substrate.min_z = min_z;
    substrate.max_z = max_z;
    
    return substrate;
}

/* Free a substrate and all its resources */
void substrate_free(substrate_t* substrate) {
    if (!substrate) return;
    
    /* Free all nodes */
    if (substrate->nodes) {
        for (int i = 0; i < substrate->node_count; i++) {
            substrate_node_free(&substrate->nodes[i]);
        }
        free(substrate->nodes);
        substrate->nodes = NULL;
    }
    
    /* Free connections */
    if (substrate->connections) {
        free(substrate->connections);
        substrate->connections = NULL;
    }
    
    /* Free layer sizes */
    if (substrate->layer_sizes) {
        free(substrate->layer_sizes);
        substrate->layer_sizes = NULL;
    }
    
    /* Reset counts */
    substrate->node_count = 0;
    substrate->connection_count = 0;
    substrate->layer_count = 0;
}

/* Connect layers in a substrate */
void substrate_connect_layers(substrate_t* substrate, int from_layer, int to_layer, 
                             float density, int max_connections) {
    if (!substrate || from_layer < 0 || to_layer < 0 || 
        from_layer >= substrate->layer_count || to_layer >= substrate->layer_count) {
        return;
    }
    
    /* Find the range of nodes in each layer */
    int from_start = 0, from_end = 0;
    int to_start = 0, to_end = 0;
    
    for (int i = 0; i <= from_layer; i++) {
        from_end += (i == 0 && substrate->layer_count > 2) ? 
                   (substrate->layer_sizes[i] + 1) : substrate->layer_sizes[i];
    }
    from_start = from_end - ((from_layer == 0 && substrate->layer_count > 2) ? 
                            (substrate->layer_sizes[from_layer] + 1) : 
                            substrate->layer_sizes[from_layer]);
    
    for (int i = 0; i <= to_layer; i++) {
        to_end += (i == 0 && substrate->layer_count > 2) ? 
                 (substrate->layer_sizes[i] + 1) : substrate->layer_sizes[i];
    }
    to_start = to_end - ((to_layer == 0 && substrate->layer_count > 2) ? 
                        (substrate->layer_sizes[to_layer] + 1) : 
                        substrate->layer_sizes[to_layer]);
    
    int from_count = from_end - from_start;
    int to_count = to_end - to_start;
    
    if (from_count == 0 || to_count == 0) {
        return;  /* No nodes to connect */
    }
    
    /* Calculate number of connections to create */
    int total_possible = from_count * to_count;
    int num_connections = (int)(density * total_possible);
    
    if (max_connections > 0 && num_connections > max_connections) {
        num_connections = max_connections;
    }
    
    /* Create connections */
    for (int i = 0; i < num_connections; i++) {
        /* Simple strategy: connect random nodes */
        int from_idx = from_start + (rand() % from_count);
        int to_idx = to_start + (rand() % to_count);
        
        /* Skip if connection already exists */
        int exists = 0;
        for (int j = 0; j < substrate->connection_count; j++) {
            if (substrate->connections[j].from_node == from_idx && 
                substrate->connections[j].to_node == to_idx) {
                exists = 1;
                break;
            }
        }
        
        if (!exists) {
            /* Create a new connection */
            substrate_connection_t conn = substrate_connection_create(
                from_idx, to_idx, 
                ((float)rand() / RAND_MAX) * 4.0f - 2.0f,  /* Weight between -2 and 2 */
                1  /* Enabled */
            );
            
            /* Add to connections array */
            substrate->connection_count++;
            substrate_connection_t* new_connections = (substrate_connection_t*)realloc(
                substrate->connections, 
                substrate->connection_count * sizeof(substrate_connection_t)
            );
            
            if (new_connections) {
                substrate->connections = new_connections;
                substrate->connections[substrate->connection_count - 1] = conn;
            } else {
                substrate->connection_count--;  // Revert the count if realloc failed
            }
        }
    }
}

/* Create a population of HyperNEAT individuals */
hyperneat_population_t* hyperneat_create_population(const hyperneat_config_t* config, 
                                                   size_t population_size) {
    if (!config || population_size == 0) return NULL;

    hyperneat_population_t* pop = (hyperneat_population_t*)calloc(1, sizeof(hyperneat_population_t));
    if (!pop) return NULL;

    /* Create NEAT population for CPPNs */
    pop->cppn_population = neat_create_population(
        config->cppn_inputs,  /* Inputs: x1, y1, x2, y2 */
        config->cppn_outputs, /* Outputs: weight */
        population_size
    );

    if (!pop->cppn_population) {
        free(pop);
        return NULL;
    }

    /* Allocate memory for individuals */
    pop->individuals = (hyperneat_individual_t*)calloc(population_size, sizeof(hyperneat_individual_t));
    if (!pop->individuals) {
        neat_free_population(pop->cppn_population);
        free(pop);
        return NULL;
    }

    /* Create individuals */
    for (size_t i = 0; i < population_size; i++) {
        /* Initialize the individual */
        pop->individuals[i].cppn = pop->cppn_population->genomes[i];
        pop->individuals[i].fitness = 0.0f;
        pop->individuals[i].objectives = NULL;
        pop->individuals[i].objective_count = 0;
        pop->individuals[i].novelty = NULL;
        pop->individuals[i].novelty_dimensions = 0;
        pop->individuals[i].activation_pattern = NULL;
        pop->individuals[i].pattern_size = 0;

        /* Create substrate for the individual */
        pop->individuals[i].substrate = (substrate_t*)calloc(1, sizeof(substrate_t));
        if (!pop->individuals[i].substrate) {
            /* Clean up */
            for (size_t j = 0; j < i; j++) {
                if (pop->individuals[j].substrate) {
                    substrate_free(pop->individuals[j].substrate);
                    free(pop->individuals[j].substrate);
                }
            }
            free(pop->individuals);
            neat_free_population(pop->cppn_population);
            free(pop);
            return NULL;
        }

        /* Initialize substrate */
        int num_layers = 2 + config->substrate_hidden_layers;
        int* layer_sizes = (int*)calloc(num_layers, sizeof(int));
        if (!layer_sizes) {
            for (size_t j = 0; j <= i; j++) {
                if (pop->individuals[j].substrate) {
                    substrate_free(pop->individuals[j].substrate);
                    free(pop->individuals[j].substrate);
                }
            }
            free(pop->individuals);
            neat_free_population(pop->cppn_population);
            free(pop);
            return NULL;
        }

        /* Set layer sizes */
        layer_sizes[0] = config->substrate_input_width * config->substrate_input_height;
        for (int j = 1; j < num_layers - 1; j++) {
            layer_sizes[j] = (int)sqrtf(layer_sizes[0] * config->substrate_output_width * config->substrate_output_height);
        }
        layer_sizes[num_layers - 1] = config->substrate_output_width * config->substrate_output_height;

        /* Initialize substrate */
        *(pop->individuals[i].substrate) = substrate_create(num_layers, layer_sizes,
                                                         -1.0f, 1.0f,  /* x range */
                                                         -1.0f, 1.0f,  /* y range */
                                                         0.0f, (float)(num_layers - 1));  /* z range */

        free(layer_sizes);
    }

    /* Initialize population fields */
    pop->population_size = population_size;
    pop->generation = 0;
    pop->config = *config;
    pop->archive = NULL;
    pop->archive_size = 0;
    pop->archive_capacity = 1000;  /* Default archive size */

    return pop;
}

/* Free a HyperNEAT individual */
void hyperneat_free_individual(hyperneat_individual_t* individual) {
    if (!individual) return;
    
    /* Free CPPN */
    if (individual->cppn) {
        neat_free_genome(individual->cppn);
        individual->cppn = NULL;
    }
    
    /* Free substrate */
    if (individual->substrate) {
        substrate_free(individual->substrate);
        free(individual->substrate);
        individual->substrate = NULL;
    }
    
    /* Free other fields */
    if (individual->objectives) {
        free(individual->objectives);
        individual->objectives = NULL;
    }
    if (individual->novelty) {
        free(individual->novelty);
        individual->novelty = NULL;
    }
    if (individual->activation_pattern) {
        free(individual->activation_pattern);
        individual->activation_pattern = NULL;
    }
    
    /* Reset fields */
    individual->fitness = 0.0f;
    individual->objective_count = 0;
    individual->novelty_dimensions = 0;
    individual->pattern_size = 0;
}

/* Free a HyperNEAT population */
void hyperneat_free_population(hyperneat_population_t* pop) {
    if (!pop) return;

    /* Free individuals */
    for (int i = 0; i < pop->population_size; i++) {
        hyperneat_free_individual(&pop->individuals[i]);
    }
    free(pop->individuals);

    /* Free NEAT population */
    neat_free_population(pop->cppn_population);

    /* Free other fields */
    if (pop->archive) {
        free(pop->archive);
        pop->archive = NULL;
    }

    /* Reset fields */
    pop->population_size = 0;
    pop->generation = 0;
    pop->archive_size = 0;
    pop->archive_capacity = 0;
}

/* Load a population from a file */
hyperneat_population_t* hyperneat_load_population(const char* filename, 
                                                 const hyperneat_config_t* config) {
    if (!filename || !config) return NULL;
    /* Implementation would be similar to hyperneat_load_individual but for a population */
    /* For brevity, this is left as an exercise */
    (void)filename;
    (void)config;
    return NULL;
}

/* Save a population to a file */
int hyperneat_save_population(const hyperneat_population_t* pop, 
                             const char* filename) {
    if (!pop || !filename) return 0;
    /* Implementation would save the entire population to a file */
    /* For brevity, this is left as an exercise */
    (void)pop;
    (void)filename;
    return 0;
}
