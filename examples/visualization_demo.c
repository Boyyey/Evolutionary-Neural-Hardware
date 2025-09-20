#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../include/neat.h"
#include "../include/visualization.h"

/* XOR problem inputs and expected outputs */
float xor_inputs[4][2] = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}
};

float xor_outputs[4] = {0.0f, 1.0f, 1.0f, 0.0f};

/* Fitness function for the XOR problem */
float evaluate_xor(neat_genome_t* genome) {
    float error = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        /* Set inputs */
        for (int j = 0; j < 2; j++) {
            neat_set_input(genome, j, xor_inputs[i][j]);
        }
        
        /* Activate network */
        neat_activate(genome);
        
        /* Get output */
        float output = neat_get_output(genome, 0);
        
        /* Calculate error */
        float diff = output - xor_outputs[i];
        error += diff * diff;
    }
    
    /* Return fitness (higher is better) */
    float fitness = 4.0f - error;  /* Max fitness is 4.0 (perfect solution) */
    return fitness > 0.0f ? fitness : 0.0f;
}

int main(int argc, char* argv[]) {
    /* Initialize random seed */
    srand((unsigned int)time(NULL));
    
    /* Create a NEAT population */
    neat_config_t config = {
        .population_size = 150,
        .num_inputs = 2,
        .num_outputs = 1,
        .compatibility_threshold = 3.0f,
        .compatibility_change = 0.3f,
        .species_elitism = 2,
        .elitism = 1,
        .survival_threshold = 0.2f,
        .weight_mutate_power = 2.5f,
        .weight_mutate_rate = 0.8f,
        .weight_replace_rate = 0.1f,
        .add_node_prob = 0.03f,
        .add_conn_prob = 0.05f,
        .crossover_rate = 0.75f,
        .mutation_rate = 0.8f,
        .compat_disjoint_coeff = 1.0f,
        .compat_weight_coeff = 0.5f,
        .stagnation_threshold = 15,
        .max_stagnation = 15,
        .max_fitness = 3.9f  /* Slightly below perfect to allow for floating point errors */
    };
    
    neat_population_t* pop = neat_create_population(&config);
    if (!pop) {
        fprintf(stderr, "Failed to create population\n");
        return 1;
    }
    
    /* Create visualizer */
    neat_visualizer_t* vis = neat_visualizer_create("NEAT XOR Demo", 1200, 800);
    if (!vis) {
        fprintf(stderr, "Failed to create visualizer\n");
        neat_free_population(pop);
        return 1;
    }
    
    /* Create plots for visualization */
    neat_plot_t* fitness_plot = neat_plot_create(100, neat_rgba(255, 0, 0, 255), "Best Fitness");
    neat_plot_t* species_plot = neat_plot_create(100, neat_rgba(0, 0, 255, 255), "Species Count");
    neat_plot_t* nodes_plot = neat_plot_create(100, neat_rgba(0, 150, 0, 255), "Avg. Nodes");
    neat_plot_t* conns_plot = neat_plot_create(100, neat_rgba(150, 0, 150, 255), "Avg. Connections");
    
    /* Main loop */
    int generation = 0;
    int max_generations = 1000;
    int solution_found = 0;
    
    while (neat_visualizer_is_running(vis) && generation < max_generations && !solution_found) {
        /* Evaluate population */
        float total_fitness = 0.0f;
        float best_fitness = 0.0f;
        neat_genome_t* best_genome = NULL;
        
        /* Evaluate each genome */
        for (size_t i = 0; i < pop->genome_count; i++) {
            neat_genome_t* genome = pop->genomes[i];
            float fitness = evaluate_xor(genome);
            genome->fitness = fitness;
            
            total_fitness += fitness;
            
            if (fitness > best_fitness) {
                best_fitness = fitness;
                best_genome = genome;
                
                /* Check for solution */
                if (fitness >= 3.9f) {  /* Close enough to perfect */
                    solution_found = 1;
                    printf("Solution found at generation %d!\n", generation);
                }
            }
        }
        
        /* Update plots */
        neat_plot_add_value(fitness_plot, best_fitness);
        neat_plot_add_value(species_plot, (float)pop->species_count);
        
        /* Calculate average nodes and connections */
        float avg_nodes = 0.0f;
        float avg_conns = 0.0f;
        
        for (size_t i = 0; i < pop->genome_count; i++) {
            avg_nodes += (float)pop->genomes[i]->node_count;
            avg_conns += (float)pop->genomes[i]->connection_count;
        }
        
        avg_nodes /= pop->genome_count;
        avg_conns /= pop->genome_count;
        
        neat_plot_add_value(nodes_plot, avg_nodes);
        neat_plot_add_value(conns_plot, avg_conns);
        
        /* Print progress */
        if (generation % 10 == 0 || solution_found) {
            printf("Generation %d: Best fitness = %.4f, Species = %zu, Avg nodes = %.1f, Avg conns = %.1f\n",
                   generation, best_fitness, pop->species_count, avg_nodes, avg_conns);
        }
        
        /* Visualize */
        neat_visualizer_clear(vis, neat_rgba(240, 240, 240, 255));
        
        /* Draw title */
        char title[256];
        snprintf(title, sizeof(title), "NEAT XOR Demo - Generation %d (Best: %.4f)", 
                generation, best_fitness);
        neat_draw_text(vis, title, 10, 10, neat_rgba(0, 0, 0, 255), 20);
        
        /* Draw best genome */
        if (best_genome) {
            neat_draw_rect(vis, 10, 50, 400, 400, neat_rgba(255, 255, 255, 255));
            neat_draw_rect(vis, 10, 50, 400, 400, neat_rgba(200, 200, 200, 255));
            neat_draw_text(vis, "Best Genome", 20, 60, neat_rgba(0, 0, 0, 255), 16);
            
            /* Draw genome */
            neat_draw_network(vis, best_genome, 20, 80, 380, 360, 15, 2);
            
            /* Draw XOR truth table */
            neat_draw_rect(vis, 430, 50, 200, 150, neat_rgba(255, 255, 255, 255));
            neat_draw_rect(vis, 430, 50, 200, 150, neat_rgba(200, 200, 200, 255));
            neat_draw_text(vis, "XOR Truth Table", 440, 60, neat_rgba(0, 0, 0, 255), 16);
            
            /* Draw table header */
            neat_draw_text(vis, "In1 In2  Out  Pred", 440, 90, neat_rgba(0, 0, 0, 255), 14);
            
            /* Draw table rows */
            for (int i = 0; i < 4; i++) {
                /* Set inputs */
                for (int j = 0; j < 2; j++) {
                    neat_set_input(best_genome, j, xor_inputs[i][j]);
                }
                
                /* Activate network */
                neat_activate(best_genome);
                
                /* Get output */
                float output = neat_get_output(best_genome, 0);
                
                /* Draw row */
                char row[64];
                snprintf(row, sizeof(row), "%2.0f   %2.0f   %2.0f   %.2f",
                        xor_inputs[i][0], xor_inputs[i][1], 
                        xor_outputs[i], output);
                
                neat_draw_text(vis, row, 440, 110 + i * 20, neat_rgba(0, 0, 0, 255), 14);
            }
            
            /* Draw fitness */
            char fitness_str[64];
            snprintf(fitness_str, sizeof(fitness_str), "Fitness: %.4f", best_fitness);
            neat_draw_text(vis, fitness_str, 430, 210, 
                          best_fitness >= 3.9f ? neat_rgba(0, 180, 0, 255) : neat_rgba(180, 0, 0, 255), 
                          16);
            
            /* Draw genome info */
            char info_str[128];
            snprintf(info_str, sizeof(info_str), "Nodes: %zu, Connections: %zu",
                    best_genome->node_count, best_genome->connection_count);
            neat_draw_text(vis, info_str, 430, 240, neat_rgba(0, 0, 0, 255), 14);
        }
        
        /* Draw plots */
        neat_draw_graph(vis, fitness_plot->values, fitness_plot->count, 
                       430, 280, 360, 150, 0.0f, 4.0f, 
                       fitness_plot->color, "Best Fitness");
        
        neat_draw_graph(vis, species_plot->values, species_plot->count, 
                       810, 50, 360, 150, 0.0f, 20.0f, 
                       species_plot->color, "Species Count");
        
        neat_draw_graph(vis, nodes_plot->values, nodes_plot->count, 
                       810, 220, 360, 150, 0.0f, 50.0f, 
                       nodes_plot->color, "Average Nodes");
        
        neat_draw_graph(vis, conns_plot->values, conns_plot->count, 
                       430, 450, 360, 150, 0.0f, 100.0f, 
                       conns_plot->color, "Average Connections");
        
        /* Draw instructions */
        neat_draw_text(vis, "Press ESC to exit", 10, 760, neat_rgba(100, 100, 100, 255), 14);
        
        /* Present the frame */
        neat_visualizer_present(vis);
        
        /* Handle events */
        neat_visualizer_handle_events(vis);
        
        /* Save screenshot of solution */
        if (solution_found) {
            neat_save_screenshot(vis, "neat_xor_solution.png");
            printf("Solution saved as 'neat_xor_solution.png'\n");
        }
        
        /* Evolve to next generation if solution not found */
        if (!solution_found) {
            neat_evolve(pop);
            generation++;
        }
    }
    
    /* Cleanup */
    neat_plot_destroy(fitness_plot);
    neat_plot_destroy(species_plot);
    neat_plot_destroy(nodes_plot);
    neat_plot_destroy(conns_plot);
    
    neat_visualizer_destroy(vis);
    neat_free_population(pop);
    
    return 0;
}
