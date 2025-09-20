#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "../include/neat.h"

/* Global test data */
static const double XOR_INPUTS[4][2] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

static const double XOR_OUTPUTS[4] = {0.0, 1.0, 1.0, 0.0};

/* Test activation functions */
void test_activation_functions() {
    print_test_header("Testing Activation Functions");
    
    /* Test sigmoid */
    TEST(fabs(neat_sigmoid(0.0) - 0.5) < 0.001, "Sigmoid(0.0) should be ~0.5");
    TEST(neat_sigmoid(100.0) > 0.99, "Sigmoid(100.0) should be close to 1.0");
    TEST(neat_sigmoid(-100.0) < 0.01, "Sigmoid(-100.0) should be close to 0.0");
    
    /* Test tanh */
    TEST(fabs(neat_tanh(0.0)) < 0.001, "tanh(0.0) should be ~0.0");
    TEST(neat_tanh(100.0) > 0.99, "tanh(100.0) should be close to 1.0");
    TEST(neat_tanh(-100.0) < -0.99, "tanh(-100.0) should be close to -1.0");
    
    /* Test ReLU */
    TEST(fabs(neat_relu(0.0)) < 0.001, "ReLU(0.0) should be 0.0");
    TEST(fabs(neat_relu(1.0) - 1.0) < 0.001, "ReLU(1.0) should be 1.0");
    TEST(fabs(neat_relu(-1.0)) < 0.001, "ReLU(-1.0) should be 0.0");
    
    /* Test leaky ReLU */
    TEST(fabs(neat_leaky_relu(0.0)) < 0.001, "LeakyReLU(0.0) should be 0.0");
    TEST(fabs(neat_leaky_relu(1.0) - 1.0) < 0.001, "LeakyReLU(1.0) should be 1.0");
    TEST(fabs(neat_leaky_relu(-1.0) + 0.01) < 0.001, "LeakyReLU(-1.0) should be -0.01");
    
    /* Test linear */
    TEST(fabs(neat_linear(0.0)) < 0.001, "Linear(0.0) should be 0.0");
    TEST(fabs(neat_linear(1.0) - 1.0) < 0.001, "Linear(1.0) should be 1.0");
    TEST(fabs(neat_linear(-1.0) + 1.0) < 0.001, "Linear(-1.0) should be -1.0");
    
    /* Test get_activation_function */
    TEST(neat_get_activation_function(NEAT_ACTIVATION_SIGMOID)(0.0) == neat_sigmoid(0.0), 
         "get_activation_function should return sigmoid function");
    TEST(neat_get_activation_function(NEAT_ACTIVATION_TANH)(0.0) == neat_tanh(0.0), 
         "get_activation_function should return tanh function");
    TEST(neat_get_activation_function(NEAT_ACTIVATION_RELU)(1.0) == neat_relu(1.0), 
         "get_activation_function should return ReLU function");
}

/* Test node creation and manipulation */
void test_node_creation() {
    print_test_header("Testing Node Creation");
    
    neat_node_t* node = neat_create_node(1, NEAT_NODE_HIDDEN, NEAT_PLACEMENT_HIDDEN);
    
    TEST_NOT_EQUAL(node, NULL, "Node creation should not return NULL");
    TEST_EQUAL(node->id, 1, "Node ID should be set correctly");
    TEST_EQUAL(node->type, NEAT_NODE_HIDDEN, "Node type should be set correctly");
    TEST_EQUAL(node->placement, NEAT_PLACEMENT_HIDDEN, "Node placement should be set correctly");
    TEST(fabs(node->bias) < 1.0, "Node bias should be initialized to a small random value");
    
    neat_free_node(node);
    
    /* Test input node */
    node = neat_create_node(2, NEAT_NODE_INPUT, NEAT_PLACEMENT_INPUT);
    TEST_NOT_EQUAL(node, NULL, "Input node creation should not return NULL");
    TEST_EQUAL(node->type, NEAT_NODE_INPUT, "Input node type should be set correctly");
    neat_free_node(node);
    
    /* Test output node */
    node = neat_create_node(3, NEAT_NODE_OUTPUT, NEAT_PLACEMENT_OUTPUT);
    TEST_NOT_EQUAL(node, NULL, "Output node creation should not return NULL");
    TEST_EQUAL(node->type, NEAT_NODE_OUTPUT, "Output node type should be set correctly");
    neat_free_node(node);
    
    /* Test bias node */
    node = neat_create_node(4, NEAT_NODE_BIAS, NEAT_PLACEMENT_INPUT);
    TEST_NOT_EQUAL(node, NULL, "Bias node creation should not return NULL");
    TEST_EQUAL(node->type, NEAT_NODE_BIAS, "Bias node type should be set correctly");
    neat_free_node(node);
}

/* Test connection creation and manipulation */
void test_connection_creation() {
    print_test_header("Testing Connection Creation");
    
    neat_connection_t* conn = neat_create_connection(1, 2, 3, 0.5, true);
    
    TEST_NOT_EQUAL(conn, NULL, "Connection creation should not return NULL");
    TEST_EQUAL(conn->innovation, 1, "Innovation number should be set correctly");
    TEST_EQUAL(conn->in_node, 2, "Input node ID should be set correctly");
    TEST_EQUAL(conn->out_node, 3, "Output node ID should be set correctly");
    TEST_EQUAL(conn->weight, 0.5, "Weight should be set correctly");
    TEST_EQUAL(conn->enabled, true, "Enabled flag should be set correctly");
    
    neat_free_connection(conn);
    
    /* Test disabled connection */
    conn = neat_create_connection(2, 3, 4, -0.5, false);
    TEST_EQUAL(conn->enabled, false, "Connection should be created as disabled");
    neat_free_connection(conn);
}

/* Test genome operations */
void test_genome_operations() {
    print_test_header("Testing Genome Operations");
    
    neat_genome_t* genome = neat_create_genome(1);
    TEST_NOT_EQUAL(genome, NULL, "Genome creation should not return NULL");
    
    /* Test adding nodes */
    int node1 = neat_add_node(genome, NEAT_NODE_INPUT, NEAT_PLACEMENT_INPUT);
    int node2 = neat_add_node(genome, NEAT_NODE_OUTPUT, NEAT_PLACEMENT_OUTPUT);
    int node3 = neat_add_node(genome, NEAT_NODE_HIDDEN, NEAT_PLACEMENT_HIDDEN);
    
    TEST_EQUAL(genome->node_count, 3, "Genome should have 3 nodes after adding 3 nodes");
    TEST_EQUAL(node1, 0, "First node should have ID 0");
    TEST_EQUAL(node2, 1, "Second node should have ID 1");
    TEST_EQUAL(node3, 2, "Third node should have ID 2");
    
    /* Test adding connections */
    int conn1 = neat_add_connection(genome, node1, node2, 0.5, true);
    int conn2 = neat_add_connection(genome, node1, node3, -0.3, true);
    int conn3 = neat_add_connection(genome, node3, node2, 0.7, true);
    
    TEST_EQUAL(genome->connection_count, 3, "Genome should have 3 connections");
    TEST_EQUAL(conn1, 1, "First connection should return 1 (new connection)");
    TEST_EQUAL(conn2, 1, "Second connection should return 1 (new connection)");
    TEST_EQUAL(conn3, 1, "Third connection should return 1 (new connection)");
    
    /* Test adding duplicate connection */
    int conn_dup = neat_add_connection(genome, node1, node2, 0.8, true);
    TEST_EQUAL(conn_dup, 0, "Adding duplicate connection should return 0");
    TEST_EQUAL(genome->connection_count, 3, "Duplicate connection should not be added");
    
    /* Test cloning genome */
    neat_genome_t* clone = neat_clone_genome(genome);
    TEST_NOT_EQUAL(clone, NULL, "Clone should not be NULL");
    TEST_NOT_EQUAL(clone, genome, "Clone should be a different object");
    TEST_EQUAL(clone->node_count, genome->node_count, "Clone should have same number of nodes");
    TEST_EQUAL(clone->connection_count, genome->connection_count, "Clone should have same number of connections");
    
    /* Clean up */
    neat_free_genome(genome);
    neat_free_genome(clone);
}

/* Test mutation operations */
void test_mutations() {
    print_test_header("Testing Mutations");
    
    neat_innovation_table_t* table = neat_create_innovation_table();
    neat_genome_t* genome = neat_create_genome(1);
    
    /* Add some nodes and connections */
    int in1 = neat_add_node(genome, NEAT_NODE_INPUT, NEAT_PLACEMENT_INPUT);
    int in2 = neat_add_node(genome, NEAT_NODE_INPUT, NEAT_PLACEMENT_INPUT);
    int out = neat_add_node(genome, NEAT_NODE_OUTPUT, NEAT_PLACEMENT_OUTPUT);
    int bias = neat_add_node(genome, NEAT_NODE_BIAS, NEAT_PLACEMENT_INPUT);
    
    neat_add_connection(genome, in1, out, 0.5, true);
    neat_add_connection(genome, bias, out, 0.7, true);
    
    /* Test weight mutation */
    double original_weight = genome->connections[0]->weight;
    neat_mutate_weights(genome);
    double new_weight = genome->connections[0]->weight;
    
    /* With some probability, the weight might not change */
    if (fabs(original_weight - new_weight) > 0.001) {
        TEST_TRUE(1, "Weight mutation should change weights");
    } else {
        /* Run a few more times to be sure */
        int changed = 0;
        for (int i = 0; i < 10; i++) {
            neat_mutate_weights(genome);
            if (fabs(genome->connections[0]->weight - original_weight) > 0.001) {
                changed = 1;
                break;
            }
        }
        TEST_TRUE(changed, "Weight mutation should eventually change weights");
    }
    
    /* Test add connection mutation */
    size_t orig_conn_count = genome->connection_count;
    neat_mutate_add_connection(genome, table);
    
    /* There's a chance no connection was added (if random nodes were already connected) */
    if (genome->connection_count > orig_conn_count) {
        TEST_EQUAL(genome->connection_count, orig_conn_count + 1, 
                  "Add connection mutation should add one connection");
    } else {
        /* Try a few more times */
        int added = 0;
        for (int i = 0; i < 10; i++) {
            neat_mutate_add_connection(genome, table);
            if (genome->connection_count > orig_conn_count) {
                added = 1;
                break;
            }
        }
        TEST_TRUE(added || (genome->connection_count == 6), 
                 "Add connection should eventually add a connection");
    }
    
    /* Test add node mutation */
    orig_conn_count = genome->connection_count;
    size_t orig_node_count = genome->node_count;
    neat_mutate_add_node(genome, table);
    
    /* A node mutation should add 1 node and 2 connections (if a connection was split) */
    if (genome->node_count > orig_node_count) {
        TEST_EQUAL(genome->node_count, orig_node_count + 1, 
                  "Add node mutation should add one node");
        TEST_EQUAL(genome->connection_count, orig_conn_count + 1, 
                  "Add node mutation should add one connection (and disable one)");
    }
    
    /* Test toggle connection mutation */
    int enabled_count = 0;
    for (size_t i = 0; i < genome->connection_count; i++) {
        if (genome->connections[i]->enabled) enabled_count++;
    }
    
    neat_mutate_toggle_connection(genome);
    
    int new_enabled_count = 0;
    for (size_t i = 0; i < genome->connection_count; i++) {
        if (genome->connections[i]->enabled) new_enabled_count++;
    }
    
    /* Toggle should either enable a disabled connection or disable an enabled one */
    TEST_TRUE(abs(new_enabled_count - enabled_count) == 1, 
             "Toggle connection should change the enabled state of one connection");
    
    /* Clean up */
    neat_free_genome(genome);
    neat_free_innovation_table(table);
}

/* Test crossover operation */
void test_crossover() {
    print_test_header("Testing Crossover");
    
    /* Create two parent genomes */
    neat_genome_t* parent1 = neat_create_genome(1);
    neat_genome_t* parent2 = neat_create_genome(2);
    
    /* Add nodes to both parents */
    int p1_in1 = neat_add_node(parent1, NEAT_NODE_INPUT, NEAT_PLACEMENT_INPUT);
    int p1_out = neat_add_node(parent1, NEAT_NODE_OUTPUT, NEAT_PLACEMENT_OUTPUT);
    int p1_hidden = neat_add_node(parent1, NEAT_NODE_HIDDEN, NEAT_PLACEMENT_HIDDEN);
    
    int p2_in1 = neat_add_node(parent2, NEAT_NODE_INPUT, NEAT_PLACEMENT_INPUT);
    int p2_out = neat_add_node(parent2, NEAT_NODE_OUTPUT, NEAT_PLACEMENT_OUTPUT);
    int p2_hidden1 = neat_add_node(parent2, NEAT_NODE_HIDDEN, NEAT_PLACEMENT_HIDDEN);
    int p2_hidden2 = neat_add_node(parent2, NEAT_NODE_HIDDEN, NEAT_PLACEMENT_HIDDEN);
    
    /* Add connections with innovation numbers */
    neat_add_connection(parent1, p1_in1, p1_out, 0.5, true);
    neat_add_connection(parent1, p1_in1, p1_hidden, 0.3, true);
    neat_add_connection(parent1, p1_hidden, p1_out, 0.7, true);
    
    neat_add_connection(parent2, p2_in1, p2_hidden1, 0.4, true);
    neat_add_connection(parent2, p2_hidden1, p2_hidden2, 0.6, true);
    neat_add_connection(parent2, p2_hidden2, p2_out, 0.8, true);
    
    /* Set innovation numbers directly for testing */
    for (size_t i = 0; i < parent1->connection_count; i++) {
        parent1->connections[i]->innovation = i + 1;
    }
    
    /* Make some innovations match between parents */
    parent2->connections[0]->innovation = 1;  /* Same as parent1's first connection */
    parent2->connections[1]->innovation = 4;  /* Disjoint */
    parent2->connections[2]->innovation = 5;  /* Disjoint */
    
    /* Set fitness values */
    parent1->fitness = 2.0;
    parent2->fitness = 1.0;  /* parent1 is fitter */
    
    /* Perform crossover */
    neat_genome_t* child = neat_crossover(parent1, parent2);
    
    /* Verify child properties */
    TEST_NOT_EQUAL(child, NULL, "Crossover should produce a non-NULL child");
    TEST_EQUAL(child->node_count, 3, "Child should have 3 nodes (from fitter parent)");
    
    /* Child should have all connections from parent1 (fitter parent) */
    int matching_conns = 0;
    for (size_t i = 0; i < parent1->connection_count; i++) {
        for (size_t j = 0; j < child->connection_count; j++) {
            if (parent1->connections[i]->innovation == child->connections[j]->innovation) {
                matching_conns++;
                break;
            }
        }
    }
    
    TEST_EQUAL(matching_conns, parent1->connection_count, 
              "Child should have all connections from fitter parent");
    
    /* Clean up */
    neat_free_genome(parent1);
    neat_free_genome(parent2);
    neat_free_genome(child);
}

/* Test speciation */
void test_speciation() {
    print_test_header("Testing Speciation");
    
    neat_population_t* pop = neat_create_population(2, 1, 10);
    TEST_NOT_EQUAL(pop, NULL, "Population creation should not return NULL");
    
    /* Verify initial population */
    TEST_EQUAL(pop->genome_count, 10, "Population should have 10 genomes");
    
    /* Initial speciation */
    neat_speciate(pop);
    TEST_GREATER(pop->species_count, 0, "Population should have at least one species");
    
    /* Test compatibility distance */
    if (pop->genome_count >= 2) {
        double distance = neat_compatibility_distance(pop->genomes[0], pop->genomes[1]);
        TEST_TRUE(distance >= 0.0, "Compatibility distance should be non-negative");
    }
    
    /* Test fitness sharing */
    if (pop->species_count > 0 && pop->species[0]->member_count > 1) {
        neat_species_t* species = pop->species[0];
        double total_fitness = 0.0;
        
        for (size_t i = 0; i < species->member_count; i++) {
            species->members[i]->fitness = 1.0;  /* Set all fitnesses to 1.0 */
            total_fitness += 1.0;
        }
        
        neat_adjust_fitness(species);
        
        /* Adjusted fitness should be 1.0 / member_count for each genome */
        double expected_adj_fitness = 1.0 / species->member_count;
        for (size_t i = 0; i < species->member_count; i++) {
            TEST_APPROX_EQUAL(species->members[i]->adjusted_fitness, expected_adj_fitness, 0.0001,
                            "Adjusted fitness should be shared among species members");
        }
    }
    
    /* Clean up */
    neat_free_population(pop);
}

/* XOR problem fitness function */
double xor_fitness(neat_genome_t* genome, void* user_data) {
    double fitness = 4.0;  /* Start with maximum possible fitness */
    double outputs[1];
    
    for (int i = 0; i < 4; i++) {
        neat_evaluate(genome, XOR_INPUTS[i], outputs);
        double error = fabs(outputs[0] - XOR_OUTPUTS[i]);
        fitness -= error * error;  /* Square error */
    }
    
    return fitness;
}

/* Test XOR problem */
void test_xor_problem() {
    print_test_header("Testing XOR Problem");
    
    /* Create a population for XOR (2 inputs, 1 output) */
    neat_population_t* pop = neat_create_population(2, 1, 50);
    pop->evaluate_genome = xor_fitness;
    pop->evaluate_user_data = NULL;
    
    /* Evolution parameters */
    const int max_generations = 100;
    const double target_fitness = 3.9;  /* Close enough to perfect */
    int generation = 0;
    double best_fitness = -1.0;
    
    /* Evolution loop */
    for (generation = 0; generation < max_generations; generation++) {
        /* Evaluate all genomes */
        for (size_t i = 0; i < pop->genome_count; i++) {
            pop->genomes[i]->fitness = xor_fitness(pop->genomes[i], NULL);
            
            /* Track best fitness */
            if (pop->genomes[i]->fitness > best_fitness) {
                best_fitness = pop->genomes[i]->fitness;
                
                /* Print progress */
                if (generation % 10 == 0) {
                    printf("Generation %d: best fitness = %.4f\n", generation, best_fitness);
                }
                
                /* Check for solution */
                if (best_fitness >= target_fitness) {
                    printf("\nSolution found in generation %d with fitness %.4f\n", 
                           generation, best_fitness);
                    break;
                }
            }
        }
        
        /* Stop if we found a solution */
        if (best_fitness >= target_fitness) {
            break;
        }
        
        /* Evolve to next generation */
        neat_evolve(pop);
    }
    
    /* Test the best genome */
    neat_genome_t* best_genome = NULL;
    for (size_t i = 0; i < pop->genome_count; i++) {
        if (best_genome == NULL || pop->genomes[i]->fitness > best_genome->fitness) {
            best_genome = pop->genomes[i];
        }
    }
    
    if (best_genome) {
        printf("\nTesting best genome (fitness = %.4f):\n", best_genome->fitness);
        printf("XOR Truth Table:\n");
        printf("0 XOR 0 = 0 (got %.3f)\n", 
               neat_evaluate(best_genome, (double[]){0.0, 0.0}, (double[1]){0})[0]);
        printf("0 XOR 1 = 1 (got %.3f)\n", 
               neat_evaluate(best_genome, (double[]){0.0, 1.0}, (double[1]){0})[0]);
        printf("1 XOR 0 = 1 (got %.3f)\n", 
               neat_evaluate(best_genome, (double[]){1.0, 0.0}, (double[1]){0})[0]);
        printf("1 XOR 1 = 0 (got %.3f)\n", 
               neat_evaluate(best_genome, (double[]){1.0, 1.0}, (double[1]){0})[0]);
    }
    
    TEST_TRUE(best_fitness >= target_fitness, 
             "NEAT should be able to solve XOR problem within 100 generations");
    
    /* Clean up */
    neat_free_population(pop);
}

/* Performance test */
void test_performance() {
    print_test_header("Performance Testing");
    
    /* Test genome evaluation speed */
    {
        printf("\nTesting genome evaluation performance...\n");
        
        /* Create a moderately complex genome */
        neat_genome_t* genome = neat_create_genome(0);
        
        /* Add input, output, and hidden nodes */
        int inputs[10];
        int hiddens[20];
        int outputs[2];
        
        for (int i = 0; i < 10; i++) {
            inputs[i] = neat_add_node(genome, NEAT_NODE_INPUT, NEAT_PLACEMENT_INPUT);
        }
        
        for (int i = 0; i < 20; i++) {
            hiddens[i] = neat_add_node(genome, NEAT_NODE_HIDDEN, NEAT_PLACEMENT_HIDDEN);
        }
        
        for (int i = 0; i < 2; i++) {
            outputs[i] = neat_add_node(genome, NEAT_NODE_OUTPUT, NEAT_PLACEMENT_OUTPUT);
        }
        
        /* Add random connections */
        for (int i = 0; i < 100; i++) {
            int from = rand() % (10 + 20);  /* Inputs + hiddens */
            int to = 10 + rand() % (20 + 2);  /* Hiddens + outputs */
            
            if (from >= 10) from += 10;  /* Skip outputs in from nodes */
            if (to >= 10) to += 10;      /* Skip inputs in to nodes */
            
            neat_add_connection(genome, from, to, (double)rand() / RAND_MAX * 4.0 - 2.0, true);
        }
        
        /* Warm up */
        double input[10] = {0};
        double output[2] = {0};
        for (int i = 0; i < 1000; i++) {
            neat_evaluate(genome, input, output);
        }
        
        /* Benchmark */
        const int num_evals = 100000;
        double start = get_time();
        
        for (int i = 0; i < num_evals; i++) {
            neat_evaluate(genome, input, output);
        }
        
        double end = get_time();
        double eval_time = (end - start) * 1e6 / num_evals;  /* microseconds per evaluation */
        
        printf("Average evaluation time: %.3f µs\n", eval_time);
        TEST_TRUE(eval_time < 1000.0, "Genome evaluation should be fast (less than 1ms per eval)");
        
        neat_free_genome(genome);
    }
    
    /* Test population evolution speed */
    {
        printf("\nTesting population evolution performance...\n");
        
        const int pop_size = 100;
        const int num_generations = 10;
        
        neat_population_t* pop = neat_create_population(10, 2, pop_size);
        pop->evaluate_genome = xor_fitness;  /* Reuse XOR fitness for testing */
        
        double start = get_time();
        
        for (int gen = 0; gen < num_generations; gen++) {
            /* Evaluate */
            for (size_t i = 0; i < pop->genome_count; i++) {
                pop->genomes[i]->fitness = xor_fitness(pop->genomes[i], NULL);
            }
            
            /* Evolve */
            if (gen < num_generations - 1) {
                neat_evolve(pop);
            }
        }
        
        double end = get_time();
        double time_per_generation = (end - start) * 1000.0 / num_generations;  /* ms per generation */
        
        printf("Average time per generation: %.2f ms\n", time_per_generation);
        TEST_TRUE(time_per_generation < 1000.0, 
                 "Population evolution should be reasonably fast (less than 1s per generation)");
        
        neat_free_population(pop);
    }
}
