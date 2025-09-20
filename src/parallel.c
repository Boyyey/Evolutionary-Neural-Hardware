#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "../include/neat.h"

/* Thread data structure */
typedef struct {
    neat_genome_t** genomes;
    size_t start_idx;
    size_t end_idx;
    void* user_data;
    neat_evaluate_func_t evaluate_func;
    pthread_mutex_t* mutex;
    int result;
} thread_data_t;

/* Thread worker function */
static void* evaluate_worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    
    for (size_t i = data->start_idx; i < data->end_idx; i++) {
        if (data->genomes[i]) {
            double fitness = data->evaluate_func(data->genomes[i], data->user_data);
            
            /* Safely update the genome's fitness */
            pthread_mutex_lock(data->mutex);
            data->genomes[i]->fitness = fitness;
            pthread_mutex_unlock(data->mutex);
        }
    }
    
    return NULL;
}

/* Evaluate a population in parallel */
void neat_evaluate_parallel(neat_population_t* pop, 
                           neat_evaluate_func_t evaluate_func, 
                           void* user_data, 
                           int num_threads) {
    if (!pop || !evaluate_func || num_threads < 1) {
        return;
    }
    
    /* Cap the number of threads at the number of genomes */
    if (num_threads > (int)pop->genome_count) {
        num_threads = (int)pop->genome_count;
    }
    
    /* If single-threaded or only one genome, just evaluate directly */
    if (num_threads <= 1 || pop->genome_count == 1) {
        for (size_t i = 0; i < pop->genome_count; i++) {
            if (pop->genomes[i]) {
                pop->genomes[i]->fitness = evaluate_func(pop->genomes[i], user_data);
            }
        }
        return;
    }
    
    /* Initialize thread data */
    pthread_t* threads = (pthread_t*)neat_malloc(num_threads * sizeof(pthread_t));
    thread_data_t* thread_data = (thread_data_t*)neat_malloc(num_threads * sizeof(thread_data_t));
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    
    /* Calculate genomes per thread */
    size_t genomes_per_thread = pop->genome_count / num_threads;
    size_t remaining = pop->genome_count % num_threads;
    size_t current_idx = 0;
    
    /* Create threads */
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].genomes = pop->genomes;
        thread_data[i].start_idx = current_idx;
        thread_data[i].end_idx = current_idx + genomes_per_thread + (i < remaining ? 1 : 0);
        thread_data[i].evaluate_func = evaluate_func;
        thread_data[i].user_data = user_data;
        thread_data[i].mutex = &mutex;
        
        pthread_create(&threads[i], NULL, evaluate_worker, &thread_data[i]);
        
        current_idx = thread_data[i].end_idx;
    }
    
    /* Wait for all threads to complete */
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    /* Clean up */
    neat_free(threads);
    neat_free(thread_data);
    pthread_mutex_destroy(&mutex);
}

/* Parallel population evolution */
void neat_evolve_parallel(neat_population_t* pop, int num_threads) {
    if (!pop || !pop->evaluate_genome) {
        return;
    }
    
    /* Evaluate all genomes in parallel */
    neat_evaluate_parallel(pop, pop->evaluate_genome, pop->evaluate_user_data, num_threads);
    
    /* The rest of the evolution process remains single-threaded for simplicity */
    /* You could parallelize more of this if needed */
    
    /* Update species fitness */
    for (size_t i = 0; i < pop->species_count; i++) {
        neat_species_t* species = pop->species[i];
        
        /* Sort species members by fitness (descending) */
        qsort(species->members, species->member_count, sizeof(neat_genome_t*), 
              neat_compare_genomes);
        
        /* Update species stats */
        if (species->member_count > 0) {
            species->max_fitness = species->members[0]->fitness;
            
            /* If this is a new best for the species, reset the staleness counter */
            if (species->max_fitness > species->max_fitness_ever) {
                species->max_fitness_ever = species->max_fitness;
                species->staleness = 0;
            } else {
                species->staleness++;
            }
        }
    }
    
    /* Sort species by best fitness (descending) */
    qsort(pop->species, pop->species_count, sizeof(neat_species_t*), 
          neat_compare_species);
    
    /* Remove stale/weak species */
    neat_remove_stale_species(pop);
    
    /* Calculate total average fitness for the entire population */
    double total_avg_fitness = 0.0;
    for (size_t i = 0; i < pop->species_count; i++) {
        neat_species_t* species = pop->species[i];
        
        /* Calculate average fitness for this species */
        double species_avg_fitness = 0.0;
        for (size_t j = 0; j < species->member_count; j++) {
            species_avg_fitness += species->members[j]->fitness;
        }
        species_avg_fitness /= species->member_count;
        
        total_avg_fitness += species_avg_fitness;
    }
    
    /* Create the next generation */
    neat_genome_t** next_gen = (neat_genome_t**)neat_calloc(pop->genome_count, sizeof(neat_genome_t*));
    size_t next_gen_count = 0;
    
    /* Elitism: keep the best genome from each species */
    for (size_t i = 0; i < pop->species_count && next_gen_count < pop->genome_count; i++) {
        neat_species_t* species = pop->species[i];
        
        if (species->member_count > 0) {
            next_gen[next_gen_count++] = neat_clone_genome(species->members[0]);
        }
    }
    
    /* Fill the rest of the next generation with offspring */
    while (next_gen_count < pop->genome_count) {
        /* Select a species using fitness-proportionate selection */
        double r = neat_rand_double() * total_avg_fitness;
        neat_species_t* selected_species = NULL;
        double sum = 0.0;
        
        for (size_t i = 0; i < pop->species_count; i++) {
            neat_species_t* species = pop->species[i];
            
            /* Calculate average fitness for this species */
            double species_avg_fitness = 0.0;
            for (size_t j = 0; j < species->member_count; j++) {
                species_avg_fitness += species->members[j]->fitness;
            }
            species_avg_fitness /= species->member_count;
            
            sum += species_avg_fitness;
            if (sum >= r) {
                selected_species = species;
                break;
            }
        }
        
        /* If no species was selected (shouldn't happen), use the first one */
        if (!selected_species && pop->species_count > 0) {
            selected_species = pop->species[0];
        }
        
        /* Create an offspring */
        if (selected_species && selected_species->member_count > 0) {
            neat_genome_t* offspring = NULL;
            
            /* 75% chance of crossover, 25% chance of mutation */
            if (selected_species->member_count >= 2 && neat_rand_double() < 0.75) {
                /* Select two parents using tournament selection */
                neat_genome_t* parent1 = neat_tournament_select(selected_species);
                neat_genome_t* parent2 = neat_tournament_select(selected_species);
                
                /* Create offspring through crossover */
                offspring = neat_crossover(parent1, parent2);
                
                /* Mutate the offspring */
                neat_mutate(offspring, pop->innovation_table);
            } else {
                /* Select a single parent and mutate it */
                neat_genome_t* parent = neat_tournament_select(selected_species);
                offspring = neat_clone_genome(parent);
                
                /* Apply multiple mutations */
                for (int i = 0; i < 3; i++) {
                    neat_mutate(offspring, pop->innovation_table);
                }
            }
            
            /* Add the offspring to the next generation */
            if (offspring && next_gen_count < pop->genome_count) {
                next_gen[next_gen_count++] = offspring;
            }
        }
    }
    
    /* Replace the old generation with the new one */
    for (size_t i = 0; i < pop->genome_count; i++) {
        if (pop->genomes[i]) {
            neat_free_genome(pop->genomes[i]);
        }
    }
    neat_free(pop->genomes);
    
    pop->genomes = next_gen;
    pop->generation++;
    
    /* Re-speciate the population */
    neat_speciate(pop);
}
