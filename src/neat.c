#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include "neat.h"
#include "config.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Define NEAT_ALLOW_RECURRENT if not already defined */
#ifndef NEAT_ALLOW_RECURRENT
#define NEAT_ALLOW_RECURRENT 0
#endif

/* Global random seed for deterministic behavior */
static unsigned long g_random_seed = 1;

/* Memory management */
void* neat_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void* neat_calloc(size_t nmemb, size_t size) {
    void *ptr = calloc(nmemb, size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void* neat_realloc(void *ptr, size_t size) {
    void *new_ptr = realloc(ptr, size);
    if (!new_ptr && size > 0) {
        fprintf(stderr, "Memory reallocation failed!\n");
        exit(EXIT_FAILURE);
    }
    return new_ptr;
}

void neat_free(void *ptr) {
    if (ptr) {
        free(ptr);
    }
}

/* Random number generation */
void neat_srand(unsigned long seed) {
    g_random_seed = seed;
}

/* Simple XOR shift random number generator */
static unsigned long xorshift32() {
    unsigned long x = g_random_seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_random_seed = x;
    return x;
}

double neat_random_uniform(double min, double max) {
    return min + (max - min) * ((double)xorshift32() / (double)UINT32_MAX);
}

double neat_random_normal(double mean, double stddev) {
    /* Box-Muller transform */
    double u1 = neat_random_uniform(0, 1);
    double u2 = neat_random_uniform(0, 1);
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + stddev * z0;
}

int neat_random_int(int min, int max) {
    return min + (xorshift32() % (max - min + 1));
}

/* Activation functions */
static double neat_activation(neat_activation_type_t type, double x) {
    switch (type) {
        case NEAT_ACTIVATION_SIGMOID: return 1.0 / (1.0 + exp(-x));
        case NEAT_ACTIVATION_TANH: return tanh(x);
        case NEAT_ACTIVATION_RELU: return x > 0.0 ? x : 0.0;
        case NEAT_ACTIVATION_LEAKY_RELU: return x > 0.0 ? x : 0.01 * x;
        case NEAT_ACTIVATION_LINEAR: return x;
        case NEAT_ACTIVATION_STEP: return x > 0.0 ? 1.0 : 0.0;
        case NEAT_ACTIVATION_SOFTSIGN: return x / (1.0 + fabs(x));
        case NEAT_ACTIVATION_SIN: return sin(x);
        case NEAT_ACTIVATION_GAUSSIAN: return exp(-(x * x));
        case NEAT_ACTIVATION_ABS: return fabs(x);
        default: return 1.0 / (1.0 + exp(-x));  /* Default to sigmoid */
    }
}

double neat_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-4.9 * x));
}

double neat_tanh(double x) {
    return tanh(x);
}

double neat_relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double neat_leaky_relu(double x) {
    return x > 0.0 ? x : 0.01 * x;
}

double neat_linear(double x) {
    return x;
}

double neat_step(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double neat_softsign(double x) {
    return x / (1.0 + fabs(x));
}

double neat_sin(double x) {
    return sin(x);
}

double neat_gaussian(double x) {
    return exp(-(x * x));
}

double neat_abs(double x) {
    return fabs(x);
}

activation_func_t neat_get_activation_function(neat_activation_type_t type) {
    switch (type) {
        case NEAT_ACTIVATION_SIGMOID: return neat_sigmoid;
        case NEAT_ACTIVATION_TANH: return neat_tanh;
        case NEAT_ACTIVATION_RELU: return neat_relu;
        case NEAT_ACTIVATION_LEAKY_RELU: return neat_leaky_relu;
        case NEAT_ACTIVATION_LINEAR: return neat_linear;
        case NEAT_ACTIVATION_STEP: return neat_step;
        case NEAT_ACTIVATION_SOFTSIGN: return neat_softsign;
        case NEAT_ACTIVATION_SIN: return neat_sin;
        case NEAT_ACTIVATION_GAUSSIAN: return neat_gaussian;
        case NEAT_ACTIVATION_ABS: return neat_abs;
        default: return neat_sigmoid;  /* Default to sigmoid */
    }
}

const char* neat_activation_name(neat_activation_type_t type) {
    static const char* names[] = {
        "sigmoid", "tanh", "relu", "leaky_relu", "linear",
        "step", "softsign", "sin", "gaussian", "abs"
    };
    
    if (type >= 0 && type < NEAT_MAX_ACTIVATION_FUNCS) {
        return names[type];
    }
    return "unknown";
}

/* Node functions */
neat_node_t neat_create_node(int id, neat_node_type_t type, neat_node_placement_t placement) {
    neat_node_t node;
    node.id = id;
    node.type = type;
    node.placement = placement;
    node.activation_type = NEAT_ACTIVATION_SIGMOID;  /* Default activation */
    node.value = 0.0;
    node.bias = neat_random_normal(0.0, 1.0);
    node.active = true;
    node.x_pos = 0;  /* Will be set during network construction */
    return node;
}

void neat_free_node(neat_node_t *node) {
    if (node) {
        neat_free(node);
    }
}

/* Connection functions */
neat_connection_t neat_create_connection(int innovation, int in_node, int out_node, 
                                       double weight, bool enabled) {
    neat_connection_t conn;
    conn.innovation = innovation;
    conn.in_node = in_node;
    conn.out_node = out_node;
    conn.weight = weight;
    conn.enabled = enabled;
    return conn;
}

void neat_free_connection(neat_connection_t *conn) {
    if (conn) {
        neat_free(conn);
    }
}

/* Genome functions */
neat_genome_t* neat_create_genome(int id) {
    neat_genome_t* genome = (neat_genome_t*)neat_malloc(sizeof(neat_genome_t));
    genome->id = id;
    genome->node_count = 0;
    genome->node_capacity = NEAT_DEFAULT_ALLOC_SIZE;
    genome->nodes = (neat_node_t*)neat_malloc(genome->node_capacity * sizeof(neat_node_t));
    
    genome->connection_count = 0;
    genome->connection_capacity = NEAT_DEFAULT_ALLOC_SIZE;
    genome->connections = (neat_connection_t*)neat_malloc(genome->connection_capacity * sizeof(neat_connection_t));
    
    genome->fitness = 0.0;
    genome->adjusted_fitness = 0.0;
    genome->global_rank = 0;
    genome->species_id = -1;
    
    genome->evaluation_order = NULL;
    genome->evaluation_order_size = 0;
    
    return genome;
}

void neat_free_genome(neat_genome_t *genome) {
    if (!genome) return;
    
    /* Free nodes */
    for (size_t i = 0; i < genome->node_count; i++) {
        // No need to free individual nodes as they're part of the array
    }
    neat_free(genome->nodes);
    
    /* Free connections */
    // No need to free individual connections as they're part of the array
    neat_free(genome->connections);
    
    /* Free evaluation order */
    neat_free(genome->evaluation_order);
    
    /* Free the genome itself */
    neat_free(genome);
}

/* Genome manipulation functions */
int neat_add_node(neat_genome_t *genome, neat_node_type_t type, neat_node_placement_t placement) {
    /* Check if we need to grow the nodes array */
    if (genome->node_count >= genome->node_capacity) {
        genome->node_capacity *= NEAT_GROWTH_FACTOR;
        genome->nodes = (neat_node_t*)neat_realloc(
            genome->nodes, 
            genome->node_capacity * sizeof(neat_node_t)
        );
        if (!genome->nodes) return -1;
    }
    
    /* Initialize the node directly in the array */
    neat_node_t new_node = neat_create_node(genome->node_count, type, placement);
    genome->nodes[genome->node_count] = new_node;
    
    /* Get the ID before incrementing */
    int new_id = genome->node_count;
    
    /* Increment the node count */
    genome->node_count++;
    
    return new_id;
    
    /* Update evaluation order */
    if (genome->evaluation_order) {
        neat_free(genome->evaluation_order);
        genome->evaluation_order = NULL;
        genome->evaluation_order_size = 0;
    }
    
    return new_id;
}

int neat_add_connection(neat_genome_t *genome, int in_node, int out_node, 
                       double weight, bool enabled) {
    /* Check if connection already exists */
    for (size_t i = 0; i < genome->connection_count; i++) {
        neat_connection_t* conn = &genome->connections[i];
        if (conn->in_node == in_node && conn->out_node == out_node) {
            return -1;  /* Connection already exists */
        }
    }
    
    /* Check if we need to grow the connections array */
    if (genome->connection_count >= genome->connection_capacity) {
        genome->connection_capacity *= NEAT_GROWTH_FACTOR;
        genome->connections = (neat_connection_t*)neat_realloc(
            genome->connections, 
            genome->connection_capacity * sizeof(neat_connection_t)
        );
        if (!genome->connections) return -1;
    }
    
    /* Initialize the connection directly in the array */
    neat_connection_t conn = neat_create_connection(-1, in_node, out_node, weight, enabled);
    genome->connections[genome->connection_count] = conn;
    
    /* Increment the connection count */
    genome->connection_count++;
    
    /* Update evaluation order */
    if (genome->evaluation_order) {
        neat_free(genome->evaluation_order);
        genome->evaluation_order = NULL;
        genome->evaluation_order_size = 0;
    }
    
    return 0;  /* Return 1 to indicate a new connection was added */
}

/* Mutation functions */
void neat_mutate_weights(neat_genome_t *genome) {
    for (size_t i = 0; i < genome->connection_count; i++) {
        neat_connection_t* conn = &genome->connections[i];
        
        if (neat_random_uniform(0, 1) < NEAT_WEIGHT_MUTATION_POWER) {
            /* Perturb weight */
            conn->weight += neat_random_normal(0, NEAT_WEIGHT_RANDOM_STRENGTH);
            
            /* Occasionally reset weight to a completely new value */
            if (neat_random_uniform(0, 1) < 0.1) {
                conn->weight = neat_random_normal(0, NEAT_WEIGHT_RANDOM_STRENGTH);
            }
        }
    }
    
    /* Mutate node biases */
    for (size_t i = 0; i < genome->node_count; i++) {
        if (neat_random_uniform(0, 1) < NEAT_MUTATE_WEIGHT_RATE) {
            genome->nodes[i].bias += neat_random_normal(0, NEAT_WEIGHT_RANDOM_STRENGTH);
        }
    }
}

void neat_mutate_add_connection(neat_genome_t *genome, neat_innovation_table_t *table) {
    if (genome->node_count < 2) return;  /* Need at least 2 nodes to add a connection */
    
    /* Select two random nodes */
    int from_node_idx = neat_random_int(0, genome->node_count - 1);
    int to_node_idx = neat_random_int(0, genome->node_count - 1);
    
    /* Make sure we're not connecting a node to itself */
    if (from_node_idx == to_node_idx) {
        return;  /* Skip self-connections */
    }
    
    /* Get the actual nodes */
    neat_node_t* from_node = &genome->nodes[from_node_idx];
    neat_node_t* to_node = &genome->nodes[to_node_idx];
    
    /* Make sure the connection would not create a cycle (unless it's a recurrent network) */
    if (!NEAT_ALLOW_RECURRENT) {
        /* Check if the connection would create a cycle */
        if (from_node->x_pos >= to_node->x_pos) {
            return;  /* Skip this connection to avoid cycles */
        }
    }
    
    /* Check if connection already exists */
    for (size_t i = 0; i < genome->connection_count; i++) {
        neat_connection_t* conn = &genome->connections[i];
        if (conn->in_node == from_node->id && conn->out_node == to_node->id) {
            return;  /* Connection already exists */
        }
    }
    
    /* Create a new connection with a random weight */
    double weight = neat_random_normal(0, NEAT_WEIGHT_RANDOM_STRENGTH);
    int innovation = -1;
    
    /* Get innovation number if table is provided */
    if (table) {
        innovation = neat_get_innovation(table, from_node->id, to_node->id, false, -1, weight);
    }
    
    /* Add the connection */
    neat_add_connection(genome, from_node->id, to_node->id, weight, true);
    
    /* Set the innovation number if we have one */
    if (innovation != -1 && genome->connection_count > 0) {
        genome->connections[genome->connection_count - 1].innovation = innovation;
    }
}


void neat_mutate_add_node(neat_genome_t *genome, neat_innovation_table_t *table) {
    if (genome->connection_count == 0) {
        return;  /* Need at least one connection to split */
    }
    
    /* Find an enabled connection to split */
    neat_connection_t* conn_to_split = NULL;
    int conn_index = -1;
    int attempts = 0;
    
    while (attempts < 100) {
        conn_index = neat_random_int(0, genome->connection_count - 1);
        if (genome->connections[conn_index].enabled) {
            conn_to_split = &genome->connections[conn_index];
            break;
        }
        attempts++;
    }
    
    if (!conn_to_split || !conn_to_split->enabled) {
        return;  /* Couldn't find a suitable connection to split */
    }
    
    /* Disable the original connection */
    conn_to_split->enabled = false;
    
    /* Create a new node */
    neat_node_type_t new_node_type = NEAT_NODE_HIDDEN;
    neat_node_placement_t new_node_placement = NEAT_PLACEMENT_HIDDEN;
    int new_node_id = neat_add_node(genome, new_node_type, new_node_placement);
    
    /* Create new connections */
    double weight1 = 1.0;  /* Weight from input to new node */
    double weight2 = conn_to_split->weight;  /* Weight from new node to output */
    
    /* Get innovation numbers if table is provided */
    int innovation1 = -1;
    int innovation2 = -1;
    
    if (table) {
        innovation1 = neat_get_innovation(table, conn_to_split->in_node, new_node_id, false, -1, weight1);
        innovation2 = neat_get_innovation(table, new_node_id, conn_to_split->out_node, false, -1, weight2);
    }
    
    /* Add the new connections */
    neat_add_connection(genome, conn_to_split->in_node, new_node_id, weight1, true);
    neat_add_connection(genome, new_node_id, conn_to_split->out_node, weight2, true);
    
    /* Set the innovation numbers if we have them */
    if (innovation1 != -1 && genome->connection_count > 1) {
        genome->connections[genome->connection_count - 2].innovation = innovation1;
        genome->connections[genome->connection_count - 1].innovation = innovation2;
    }
}

void neat_mutate_toggle_connection(neat_genome_t *genome) {
    if (genome->connection_count == 0) {
        return;
    }
    
    /* Find an enabled connection to toggle */
    neat_connection_t* conn = NULL;
    int enabled_count = 0;
    
    for (size_t i = 0; i < genome->connection_count; i++) {
        if (genome->connections[i].enabled) {
            enabled_count++;
            if (neat_random_uniform(0, 1) < 1.0 / enabled_count) {
                conn = &genome->connections[i];
            }
        }
    }
    
    /* If no enabled connection found, enable a random disabled one */
    if (!conn) {
        int disabled_count = genome->connection_count - enabled_count;
        if (disabled_count > 0) {
            int target = neat_random_int(0, disabled_count - 1);
            int found = 0;
            
            for (size_t i = 0; i < genome->connection_count; i++) {
                if (!genome->connections[i].enabled) {
                    if (found == target) {
                        genome->connections[i].enabled = true;
                        return;
                    }
                    found++;
                }
            }
        }
    } else {
        /* Toggle the found connection */
        if (conn) {
            conn->enabled = !conn->enabled;
        }
    }
}

void neat_mutate_activation(neat_genome_t *genome) {
    if (genome->node_count == 0) {
        return;
    }
    
    /* Pick a random node (excluding input and bias nodes) */
    int node_idx = neat_random_int(0, genome->node_count - 1);
    
    if (genome->nodes[node_idx].type == NEAT_NODE_INPUT || 
        genome->nodes[node_idx].type == NEAT_NODE_BIAS) {
        return;  /* Don't change activation for input or bias nodes */
    }
    
    /* Change to a random activation function */
    genome->nodes[node_idx].activation_type = neat_random_int(0, NEAT_MAX_ACTIVATION_FUNCS - 1);
}

void neat_mutate(neat_genome_t *genome, neat_innovation_table_t *table) {
    /* Apply mutations based on probabilities */
    if (neat_random_uniform(0, 1) < NEAT_MUTATE_WEIGHT_RATE) {
        neat_mutate_weights(genome);
    }
    
    if (neat_random_uniform(0, 1) < NEAT_MUTATE_NODE_RATE) {
        neat_mutate_add_node(genome, table);
    }
    
    if (neat_random_uniform(0, 1) < NEAT_MUTATE_LINK_RATE) {
        neat_mutate_add_connection(genome, table);
    }
    
    if (neat_random_uniform(0, 1) < NEAT_MUTATE_TOGGLE_LINK_RATE) {
        neat_mutate_toggle_connection(genome);
    }
    
    if (neat_random_uniform(0, 1) < NEAT_MUTATE_ACTIVATION_RATE) {
        neat_mutate_activation(genome);
    }
}

/* Crossover functions */
neat_genome_t* neat_crossover(const neat_genome_t *parent1, const neat_genome_t *parent2) {
    /* Determine fitter parent */
    const neat_genome_t *fitter = (parent1->fitness >= parent2->fitness) ? parent1 : parent2;
    const neat_genome_t *other = (parent1->fitness >= parent2->fitness) ? parent2 : parent1;
    
    /* Create a new genome */
    neat_genome_t *child = neat_create_genome(-1);  /* ID will be set by population */
    
    /* Copy nodes from fitter parent */
    for (size_t i = 0; i < fitter->node_count; i++) {
        neat_node_t node = fitter->nodes[i];
        int new_node_id = neat_add_node(child, node.type, node.placement);
        child->nodes[new_node_id].bias = node.bias;
        child->nodes[new_node_id].activation_type = node.activation_type;
    }
    
    /* Track which connections we've added to avoid duplicates */
    int *connection_added = (int*)neat_calloc(fitter->connection_count + other->connection_count, sizeof(int));
    
    /* Add connections from both parents */
    for (size_t i = 0; i < fitter->connection_count; i++) {
        neat_connection_t conn = fitter->connections[i];
        neat_connection_t matching_conn;
        bool found_match = false;
        
        /* Look for matching innovation in other parent */
        for (size_t j = 0; j < other->connection_count; j++) {
            if (other->connections[j].innovation == conn.innovation) {
                matching_conn = other->connections[j];
                found_match = true;
                break;
            }
        }
        
        /* Choose which parent's connection to inherit */
        neat_connection_t inherited_conn = conn;
        if (found_match && neat_random_uniform(0, 1) < 0.5) {
            inherited_conn = matching_conn;
        }
        
        /* Add the connection to the child */
        neat_add_connection(child, inherited_conn.in_node, 
                          inherited_conn.out_node, 
                          inherited_conn.weight,
                          inherited_conn.enabled);
        connection_added[i] = 1;
    }
    
    /* Add any remaining connections from the other parent that weren't in the fitter parent */
    for (size_t i = 0; i < other->connection_count; i++) {
        neat_connection_t conn = other->connections[i];
        bool exists = false;
        
        /* Check if this innovation already exists in the fitter parent */
        for (size_t j = 0; j < fitter->connection_count; j++) {
            if (fitter->connections[j].innovation == conn.innovation) {
                exists = true;
                break;
            }
        }
        
        /* If not in fitter parent, maybe include it (50% chance) */
        if (!exists && neat_random_uniform(0, 1) < 0.5) {
            neat_add_connection(child, conn.in_node, conn.out_node, 
                              conn.weight, conn.enabled);
        }
    }
    
    neat_free(connection_added);
    return child;
}

/* Compatibility distance function */
double neat_compatibility_distance(const neat_genome_t *genome1, const neat_genome_t *genome2) {
    /* Count matching, disjoint, and excess genes */
    size_t matching = 0, disjoint = 0, excess = 0;
    double weight_diff_sum = 0.0;
    
    /* Compare connections */
    size_t i = 0, j = 0;
    while (i < genome1->connection_count && j < genome2->connection_count) {
        neat_connection_t conn1 = genome1->connections[i];
        neat_connection_t conn2 = genome2->connections[j];
        
        if (conn1.innovation == conn2.innovation) {
            /* Matching gene */
            matching++;
            weight_diff_sum += fabs(conn1.weight - conn2.weight);
            i++;
            j++;
        } else if (conn1.innovation < conn2.innovation) {
            /* Disjoint gene in genome1 */
            disjoint++;
            i++;
        } else {
            /* Disjoint gene in genome2 */
            disjoint++;
            j++;
        }
    }
    
    /* Count remaining excess genes */
    excess += (genome1->connection_count - i) + (genome2->connection_count - j);
    
    /* Calculate compatibility distance */
    double N = (double)((genome1->connection_count > genome2->connection_count) ? 
                       genome1->connection_count : genome2->connection_count);
    if (N < 20.0) N = 1.0;  /* Protect against divide by zero */
    
    double distance = (NEAT_EXCESS_COEFF * excess) / N +
                     (NEAT_DISJOINT_COEFF * disjoint) / N;
    
    /* Only add weight difference if there are matching genes */
    if (matching > 0) {
        distance += (NEAT_WEIGHT_COEFF * weight_diff_sum) / matching;
    }
    
    return distance;
}

/* Network evaluation */
void neat_update_network(neat_genome_t *genome) {
    /* If evaluation order hasn't been computed yet, compute it */
    if (genome->evaluation_order == NULL) {
        /* Simple implementation: evaluate nodes in order of their IDs */
        /* In a more complete implementation, this would perform a topological sort */
        genome->evaluation_order_size = genome->node_count;
        genome->evaluation_order = (int*)neat_malloc(genome->node_count * sizeof(int));
        
        for (size_t i = 0; i < genome->node_count; i++) {
            genome->evaluation_order[i] = i;
        }
    }
    
    /* Reset node values (except inputs) */
    for (size_t i = 0; i < genome->node_count; i++) {
        if (genome->nodes[i].type != NEAT_NODE_INPUT) {
            genome->nodes[i].value = 0.0;
        }
    }
    
    /* Evaluate nodes in order */
    for (size_t i = 0; i < genome->evaluation_order_size; i++) {
        int node_idx = genome->evaluation_order[i];
        neat_node_t *node = &genome->nodes[node_idx];
        
        /* Skip input nodes (they're set externally) */
        if (node->type == NEAT_NODE_INPUT) {
            continue;
        }
        
        /* Sum up weighted inputs */
        double sum = 0.0;
        
        for (size_t j = 0; j < genome->connection_count; j++) {
            neat_connection_t *conn = &genome->connections[j];
            
            if (conn->out_node == node->id && conn->enabled) {
                neat_node_t *in_node = &genome->nodes[conn->in_node];
                if (in_node && in_node->active) {
                    sum += in_node->value * conn->weight;
                }
            }
        }
        
        /* Add bias and apply activation function */
        sum += node->bias;
        node->value = neat_activation(node->activation_type, sum);
    }
}

void neat_evaluate(neat_genome_t *genome, const double *inputs, double *outputs) {
    /* Set input values */
    size_t input_count = 0;
    
    for (size_t i = 0; i < genome->node_count; i++) {
        neat_node_t *node = &genome->nodes[i];
        
        if (node->type == NEAT_NODE_INPUT) {
            if (input_count < NEAT_MAX_INPUTS) {
                node->value = inputs[input_count++];
            } else {
                node->value = 0.0;  /* Default to 0 if not enough inputs provided */
            }
        } else if (node->type == NEAT_NODE_BIAS) {
            node->value = 1.0;  /* Bias nodes always output 1.0 */
        } else {
            node->value = 0.0;  /* Reset other nodes */
        }
    }
    
    /* Update network */
    neat_update_network(genome);
    
    /* Collect output values */
    size_t output_count = 0;
    for (size_t i = 0; i < genome->node_count; i++) {
        neat_node_t *node = &genome->nodes[i];
        
        if (node->type == NEAT_NODE_OUTPUT) {
            if (output_count < NEAT_MAX_OUTPUTS) {
                outputs[output_count++] = node->value;
            }
        }
    }
}

/* Innovation table functions */
neat_innovation_table_t* neat_create_innovation_table(void) {
    neat_innovation_table_t* table = (neat_innovation_table_t*)neat_malloc(sizeof(neat_innovation_table_t));
    table->count = 0;
    table->capacity = NEAT_DEFAULT_ALLOC_SIZE;
    table->innovations = (neat_innovation_t*)neat_malloc(table->capacity * sizeof(neat_innovation_t));
    table->next_innovation = 1;
    table->next_node_id = 1;  /* Start node IDs from 1 */
    table->next_species_id = 1;
    return table;
}

void neat_free_innovation_table(neat_innovation_table_t *table) {
    if (table) {
        neat_free(table->innovations);
        neat_free(table);
    }
}

/* Innovation table functions */
int neat_get_innovation(neat_innovation_table_t *table, int in_node, int out_node, 
                       bool is_new_node, int node_id, double weight) {
    /* Check if this innovation already exists */
    for (size_t i = 0; i < table->count; i++) {
        neat_innovation_t *innov = &table->innovations[i];
        
        if (innov->in_node == in_node && 
            innov->out_node == out_node && 
            innov->is_new_node == is_new_node) {
            return innov->innovation_number;
        }
    }
    
    /* If not, create a new innovation */
    if (table->count >= table->capacity) {
        table->capacity *= NEAT_GROWTH_FACTOR;
        table->innovations = (neat_innovation_t*)neat_realloc(
            table->innovations, 
            table->capacity * sizeof(neat_innovation_t)
        );
    }
    
    neat_innovation_t *innov = &table->innovations[table->count++];
    innov->innovation_id = table->next_innovation++;
    innov->in_node = in_node;
    innov->out_node = out_node;
    innov->is_new_node = is_new_node;
    innov->node_id = is_new_node ? node_id : 0;
    innov->weight = weight;
    
    /* For new node innovations, we need to assign a new node ID */
    if (is_new_node) {
        innov->innovation_number = table->next_node_id++;
    } else {
        innov->innovation_number = table->next_innovation++;
    }
    
    return innov->innovation_number;
}

/* Species functions */
neat_species_t* neat_create_species(int id) {
    neat_species_t *species = (neat_species_t*)neat_malloc(sizeof(neat_species_t));
    species->id = id;
    species->member_count = 0;
    species->member_capacity = NEAT_DEFAULT_ALLOC_SIZE;
    species->members = (neat_genome_t**)neat_malloc(species->member_capacity * sizeof(neat_genome_t*));
    species->best_fitness = -1e10;
    species->average_fitness = 0.0;
    species->staleness = 0;
    species->age = 0;
    species->representative = NULL;
    return species;
}

void neat_free_species(neat_species_t *species) {
    if (species) {
        neat_free(species->members);
        neat_free(species);
    }
}

void neat_add_genome_to_species(neat_species_t *species, neat_genome_t *genome) {
    /* Check if we need to grow the members array */
    if (species->member_count >= species->member_capacity) {
        species->member_capacity *= NEAT_GROWTH_FACTOR;
        species->members = (neat_genome_t**)neat_realloc(
            species->members,
            species->member_capacity * sizeof(neat_genome_t*)
        );
    }
    
    /* Add the genome to the species */
    species->members[species->member_count++] = genome;
    
    /* Update best fitness if needed */
    if (genome->fitness > species->best_fitness) {
        species->best_fitness = genome->fitness;
        species->staleness = 0;
    }
    
    /* Update average fitness */
    species->average_fitness = 0.0;
    for (size_t i = 0; i < species->member_count; i++) {
        species->average_fitness += species->members[i]->fitness;
    }
    species->average_fitness /= species->member_count;
}

void neat_remove_genome_from_species(neat_species_t *species, const neat_genome_t *genome) {
    /* Find and remove the genome */
    for (size_t i = 0; i < species->member_count; i++) {
        if (species->members[i] == genome) {
            /* Shift remaining elements */
            for (size_t j = i; j < species->member_count - 1; j++) {
                species->members[j] = species->members[j + 1];
            }
            
            species->member_count--;
            
            /* Update average fitness */
            species->average_fitness = 0.0;
            for (size_t k = 0; k < species->member_count; k++) {
                species->average_fitness += species->members[k]->fitness;
            }
            
            if (species->member_count > 0) {
                species->average_fitness /= species->member_count;
            }
            
            break;
        }
    }
}

void neat_adjust_fitness(neat_species_t *species) {
    /* Simple fitness sharing: divide by species size */
    if (species->member_count > 0) {
        for (size_t i = 0; i < species->member_count; i++) {
            species->members[i]->adjusted_fitness = species->members[i]->fitness / species->member_count;
        }
    }
}

/* Population functions */
neat_population_t* neat_create_population(size_t input_size, size_t output_size, 
                                         size_t population_size) {
    /* Create population */
    neat_population_t *pop = (neat_population_t*)neat_malloc(sizeof(neat_population_t));
    pop->genome_count = 0;
    pop->genome_capacity = population_size;
    pop->genomes = (neat_genome_t**)neat_malloc(population_size * sizeof(neat_genome_t*));
    
    pop->species_count = 0;
    pop->species_capacity = NEAT_DEFAULT_ALLOC_SIZE;
    pop->species = (neat_species_t**)neat_malloc(pop->species_capacity * sizeof(neat_species_t*));
    
    pop->innovation_table = neat_create_innovation_table();
    pop->population_size = population_size;
    pop->generation = 0;
    pop->max_fitness_achieved = -1e10;
    pop->evaluate_genome = NULL;
    pop->evaluate_user_data = NULL;
    
    /* Create initial population */
    for (size_t i = 0; i < population_size; i++) {
        neat_genome_t *genome = neat_create_genome(i);
        
        /* Add input nodes */
        for (size_t j = 0; j < input_size; j++) {
            neat_add_node(genome, NEAT_NODE_INPUT, NEAT_PLACEMENT_INPUT);
        }
        
        /* Add bias node */
        neat_add_node(genome, NEAT_NODE_BIAS, NEAT_PLACEMENT_INPUT);
        
        /* Add output nodes */
        for (size_t j = 0; j < output_size; j++) {
            neat_add_node(genome, NEAT_NODE_OUTPUT, NEAT_PLACEMENT_OUTPUT);
        }
        
        /* Add initial connections (minimal structure) */
        /* Connect each input to each output */
        for (size_t in = 0; in < input_size + 1; in++) {  /* +1 for bias */
            for (size_t out = 0; out < output_size; out++) {
                int out_node_id = input_size + 1 + out;  /* Skip inputs and bias */
                double weight = neat_random_normal(0, 1.0);
                neat_add_connection(genome, in, out_node_id, weight, true);
                
                /* Set innovation number */
genome->connections[genome->connection_count - 1].innovation = 
                    neat_get_innovation(pop->innovation_table, in, out_node_id, false, 0, weight);
            }
        }
        
        pop->genomes[pop->genome_count++] = genome;
    }
    
    /* Create initial species */
    neat_speciate(pop);
    
    return pop;
}

void neat_free_population(neat_population_t *pop) {
    if (!pop) return;
    
    /* Free all genomes */
    for (size_t i = 0; i < pop->genome_count; i++) {
        neat_free_genome(pop->genomes[i]);
    }
    neat_free(pop->genomes);
    
    /* Free all species */
    for (size_t i = 0; i < pop->species_count; i++) {
        neat_free_species(pop->species[i]);
    }
    neat_free(pop->species);
    
    /* Free innovation table */
    neat_free_innovation_table(pop->innovation_table);
    
    neat_free(pop);
}

void neat_speciate(neat_population_t *pop) {
    /* Reset species */
    for (size_t i = 0; i < pop->species_count; i++) {
        neat_free_species(pop->species[i]);
    }
    pop->species_count = 0;
    
    /* If no genomes, nothing to do */
    if (pop->genome_count == 0) {
        return;
    }
    
    /* Create the first species with the first genome */
    neat_species_t *first_species = neat_create_species(pop->innovation_table->next_species_id++);
    first_species->representative = pop->genomes[0];
    neat_add_genome_to_species(first_species, pop->genomes[0]);
    pop->species[pop->species_count++] = first_species;
    
    /* For each remaining genome, find a compatible species or create a new one */
    for (size_t i = 1; i < pop->genome_count; i++) {
        neat_genome_t *genome = pop->genomes[i];
        int found_species = 0;
        
        /* Check against each existing species */
        for (size_t j = 0; j < pop->species_count; j++) {
            neat_species_t *species = pop->species[j];
            
            if (species->member_count > 0 && species->representative) {
                double distance = neat_compatibility_distance(genome, species->representative);
                
                if (distance < NEAT_COMPATIBILITY_THRESHOLD) {
                    /* Add to this species */
                    neat_add_genome_to_species(species, genome);
                    found_species = 1;
                    break;
                }
            }
        }
        
        /* If no compatible species found, create a new one */
        if (!found_species) {
            if (pop->species_count >= pop->species_capacity) {
                pop->species_capacity *= NEAT_GROWTH_FACTOR;
                pop->species = (neat_species_t**)neat_realloc(
                    pop->species,
                    pop->species_capacity * sizeof(neat_species_t*)
                );
            }
            
            neat_species_t *new_species = neat_create_species(pop->innovation_table->next_species_id++);
            new_species->representative = genome;
            neat_add_genome_to_species(new_species, genome);
            pop->species[pop->species_count++] = new_species;
        }
    }
    
    /* Remove empty species */
    size_t i = 0;
    while (i < pop->species_count) {
        if (pop->species[i]->member_count == 0) {
            neat_free_species(pop->species[i]);
            
            /* Shift remaining species */
            for (size_t j = i; j < pop->species_count - 1; j++) {
                pop->species[j] = pop->species[j + 1];
            }
            
            pop->species_count--;
        } else {
            i++;
        }
    }
}

void neat_remove_stale_species(neat_population_t *pop) {
    size_t i = 0;
    while (i < pop->species_count) {
        neat_species_t *species = pop->species[i];
        
        /* Increment staleness */
        species->staleness++;
        
        /* Check if this species has improved */
        if (species->best_fitness > pop->max_fitness_achieved) {
            species->staleness = 0;
            pop->max_fitness_achieved = species->best_fitness;
        } else {
            i++;
        }
    }
}

void neat_remove_weak_species(neat_population_t *pop) {
    /* Calculate total average fitness */
    double total_avg_fitness = 0.0;
    for (size_t i = 0; i < pop->species_count; i++) {
        neat_species_t *species = pop->species[i];
        total_avg_fitness += fabs(species->average_fitness);
    }
    
    /* Remove weak species */
    size_t i = 0;
    while (i < pop->species_count) {
        neat_species_t *species = pop->species[i];
        
        /* Calculate how many offspring this species should have */
        double fitness_ratio = fabs(species->average_fitness) / total_avg_fitness;
        int offspring_count = (int)(fitness_ratio * pop->population_size);
        
        /* Remove species with no offspring */
        if (offspring_count < 1) {
            neat_free_species(species);
            
            /* Shift remaining species */
            for (size_t j = i; j < pop->species_count - 1; j++) {
                pop->species[j] = pop->species[j + 1];
            }
            
            pop->species_count--;
        } else {
            i++;
        }
    }
}

void neat_reproduce(neat_population_t *pop) {
    /* Calculate total average fitness */
    double total_avg_fitness = 0.0;
    for (size_t i = 0; i < pop->species_count; i++) {
        neat_species_t *species = pop->species[i];
        total_avg_fitness += fabs(species->average_fitness);
    }
    
    /* Create new population */
    neat_genome_t **new_genomes = (neat_genome_t**)neat_malloc(pop->population_size * sizeof(neat_genome_t*));
    size_t new_genome_count = 0;
    
    /* Carry over elites */
    for (size_t i = 0; i < pop->species_count && new_genome_count < pop->population_size; i++) {
        neat_species_t *species = pop->species[i];
        
        /* Sort species by fitness (descending) */
        for (size_t j = 0; j < species->member_count - 1; j++) {
            for (size_t k = j + 1; k < species->member_count; k++) {
                if (species->members[j]->fitness < species->members[k]->fitness) {
                    neat_genome_t *temp = species->members[j];
                    species->members[j] = species->members[k];
                    species->members[k] = temp;
                }
            }
        }
        
        /* Carry over the best genome as elite */
        if (species->member_count > 0 && new_genome_count < pop->population_size) {
            new_genomes[new_genome_count++] = neat_clone_genome(species->members[0]);
        }
    }
    
    /* Fill the rest of the population with offspring */
    while (new_genome_count < pop->population_size) {
        /* Select a species with probability proportional to its average fitness */
        double r = neat_random_uniform(0, total_avg_fitness);
        neat_species_t *selected_species = NULL;
        
        for (size_t i = 0; i < pop->species_count; i++) {
            neat_species_t *species = pop->species[i];
            r -= fabs(species->average_fitness);
            
            if (r <= 0.0 || i == pop->species_count - 1) {
                selected_species = species;
                break;
            }
        }
        
        if (!selected_species || selected_species->member_count == 0) {
            continue;  /* Shouldn't happen, but just in case */
        }
        
        /* Select parent(s) and create offspring */
        neat_genome_t *parent1 = NULL;
        neat_genome_t *parent2 = NULL;
        
        /* Select first parent (tournament selection) */
        for (int i = 0; i < 3; i++) {
            int idx = neat_random_int(0, selected_species->member_count - 1);
            if (!parent1 || selected_species->members[idx]->fitness > parent1->fitness) {
                parent1 = selected_species->members[idx];
            }
        }
        
        /* Select second parent (with some probability) */
        if (neat_random_uniform(0, 1) < 0.3) {  /* 30% chance of sexual reproduction */
            for (int i = 0; i < 3; i++) {
                int idx = neat_random_int(0, selected_species->member_count - 1);
                if (!parent2 || selected_species->members[idx]->fitness > parent2->fitness) {
                    /* Make sure we don't select the same parent twice */
                    if (selected_species->members[idx] != parent1) {
                        parent2 = selected_species->members[idx];
                    }
                }
            }
        }
        
        /* Create offspring */
        neat_genome_t *offspring = NULL;
        
        if (parent2) {
            /* Sexual reproduction */
            offspring = neat_crossover(parent1, parent2);
        } else {
            /* Asexual reproduction */
            offspring = neat_clone_genome(parent1);
        }
        
        /* Mutate the offspring */
        neat_mutate(offspring, pop->innovation_table);
        
        /* Add to new population */
        if (new_genome_count < pop->population_size) {
            new_genomes[new_genome_count++] = offspring;
        } else {
            neat_free_genome(offspring);
        }
    }
    
    /* Free old genomes */
    for (size_t i = 0; i < pop->genome_count; i++) {
        neat_free_genome(pop->genomes[i]);
    }
    neat_free(pop->genomes);
    
    /* Update population */
    pop->genomes = new_genomes;
    pop->genome_count = new_genome_count;
    
    /* Increment generation */
    pop->generation++;
}

void neat_evolve(neat_population_t *pop) {
    /* Evaluate all genomes */
    if (pop->evaluate_genome) {
        for (size_t i = 0; i < pop->genome_count; i++) {
            pop->genomes[i]->fitness = pop->evaluate_genome(pop->genomes[i], pop->evaluate_user_data);
            
            /* Update max fitness */
            if (pop->genomes[i]->fitness > pop->max_fitness_achieved) {
                pop->max_fitness_achieved = pop->genomes[i]->fitness;
            }
        }
    }
    
    /* Speciate */
    neat_speciate(pop);
    
    /* Adjust fitness within species */
    for (size_t i = 0; i < pop->species_count; i++) {
        neat_adjust_fitness(pop->species[i]);
    }
    
    /* Remove stale species */
    neat_remove_stale_species(pop);
    
    /* Remove weak species */
    neat_remove_weak_species(pop);
    
    /* Reproduce to create next generation */
    neat_reproduce(pop);
}
