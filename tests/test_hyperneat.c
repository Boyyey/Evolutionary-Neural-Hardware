#include "../include/hyperneat.h"
#include "../include/neat.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Test function for XOR problem */
float xor_fitness(hyperneat_individual_t* ind) {
    float inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float expected[4] = {0, 1, 1, 0};
    float outputs[1];
    float error = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        hyperneat_activate(ind, inputs[i], outputs);
        float diff = outputs[0] - expected[i];
        error += diff * diff;
    }
    
    /* Convert to fitness (higher is better) */
    return 1.0f / (1.0f + error);
}

/* Test function for visualization */
void visualize_hyperneat(hyperneat_individual_t* ind, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    
    /* Write DOT format header */
    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  rankdir=LR;\n");
    fprintf(fp, "  node [shape=circle, style=filled];\n");
    
    /* Add nodes */
    for (int i = 0; i < ind->substrate->node_count; i++) {
        const char* color = "white";
        const char* shape = "circle";
        
        switch (ind->substrate->nodes[i]->node_type) {
            case NEAT_NODE_INPUT: 
                color = "lightblue";
                shape = "box";
                break;
            case NEAT_NODE_HIDDEN: 
                color = "lightgray";
                break;
            case NEAT_NODE_OUTPUT: 
                color = "lightgreen";
                shape = "box";
                break;
            case NEAT_NODE_BIAS: 
                color = "pink";
                shape = "diamond";
                break;
        }
        
        fprintf(fp, "  n%d [label=\"\", shape=%s, fillcolor=%s];\n", 
                i, shape, color);
    }
    
    /* Add connections */
    for (int i = 0; i < ind->substrate->connection_count; i++) {
        substrate_connection_t* conn = ind->substrate->connections[i];
        if (conn->enabled) {
            const char* color = conn->weight > 0 ? "blue" : "red";
            float width = fminf(3.0f, 0.1f + fabsf(conn->weight) / 2.0f);
            
            fprintf(fp, "  n%d -> n%d [color=\"%s\", penwidth=%.2f];\n",
                    conn->from_node, conn->to_node, color, width);
        }
    }
    
    fprintf(fp, "}\n");
    fclose(fp);
}

int main() {
    /* Initialize random seed */
    srand((unsigned int)time(NULL));
    
    /* Create HyperNEAT configuration */
    hyperneat_config_t config = hyperneat_get_default_config();
    config.substrate_input_width = 2;  /* XOR has 2 inputs */
    config.substrate_input_height = 1;
    config.substrate_output_width = 1; /* XOR has 1 output */
    config.substrate_output_height = 1;
    config.substrate_hidden_layers = 1;  /* One hidden layer */
    
    /* Create population */
    printf("Creating HyperNEAT population...\n");
    hyperneat_population_t* pop = hyperneat_create_population(&config, 50);
    if (!pop) {
        fprintf(stderr, "Failed to create population\n");
        return 1;
    }
    
    /* Evolution loop */
    const int max_generations = 100;
    float best_fitness = 0.0f;
    
    for (int gen = 0; gen < max_generations; gen++) {
        /* Evolve population */
        hyperneat_evolve(pop, xor_fitness);
        
        /* Find best individual */
        float gen_best = 0.0f;
        for (size_t i = 0; i < pop->population_size; i++) {
            if (pop->individuals[i]->fitness > gen_best) {
                gen_best = pop->individuals[i]->fitness;
            }
        }
        
        /* Update overall best */
        if (gen_best > best_fitness) {
            best_fitness = gen_best;
            
            /* Save visualization of the best individual */
            char filename[256];
            snprintf(filename, sizeof(filename), "best_gen_%03d.dot", gen);
            
            /* Find the best individual */
            hyperneat_individual_t* best = NULL;
            for (size_t i = 0; i < pop->population_size; i++) {
                if (pop->individuals[i]->fitness == gen_best) {
                    best = pop->individuals[i];
                    break;
                }
            }
            
            if (best) {
                visualize_hyperneat(best, filename);
                printf("Saved visualization to %s\n", filename);
            }
        }
        
        /* Print progress */
        printf("Generation %d: best fitness = %.4f\n", gen + 1, gen_best);
        
        /* Early stopping if we've solved XOR */
        if (gen_best > 0.95f) {
            printf("\nSolved XOR in %d generations!\n", gen + 1);
            break;
        }
    }
    
    /* Test the best individual */
    hyperneat_individual_t* best = NULL;
    for (size_t i = 0; i < pop->population_size; i++) {
        if (!best || pop->individuals[i]->fitness > best->fitness) {
            best = pop->individuals[i];
        }
    }
    
    if (best) {
        printf("\nTesting best individual (fitness = %.4f):\n", best->fitness);
        
        float inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        float expected[4] = {0, 1, 1, 0};
        float outputs[1];
        
        for (int i = 0; i < 4; i++) {
            hyperneat_activate(best, inputs[i], outputs);
            printf("Input: [%.0f, %.0f]  Output: %.4f (Expected: %.0f)\n", 
                   inputs[i][0], inputs[i][1], outputs[0], expected[i]);
        }
        
        /* Save the best individual */
        hyperneat_save_individual(best, "best_hyperneat_xor.bin");
        printf("\nSaved best individual to 'best_hyperneat_xor.bin'\n");
    }
    
    /* Cleanup */
    hyperneat_free_population(pop);
    
    return 0;
}
