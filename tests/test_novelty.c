#include "../include/novelty.h"
#include "../include/neat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Simple point structure for testing */
typedef struct {
    float x, y;
} point_t;

/* Test evaluation function for points in 2D space */
void evaluate_point(void* individual, float* fitness, float* behavior, size_t behavior_size, void* user_data) {
    (void)user_data; /* Unused parameter */
    
    point_t* point = (point_t*)individual;
    
    /* Simple fitness: distance from origin (to be maximized) */
    *fitness = sqrtf(point->x * point->x + point->y * point->y);
    
    /* Behavior is just the x, y coordinates */
    if (behavior_size >= 2) {
        behavior[0] = point->x;
        behavior[1] = point->y;
    }
}

/* Termination condition */
int should_terminate(novelty_search_t* ns, void* user_data) {
    (void)user_data; /* Unused parameter */
    
    /* Stop if we've found points in all quadrants */
    int quadrants[4] = {0};
    
    for (size_t i = 0; i < ns->archive->size; i++) {
        behavior_t* b = ns->archive->items[i];
        if (b->data[0] >= 0 && b->data[1] >= 0) quadrants[0] = 1;
        if (b->data[0] <  0 && b->data[1] >= 0) quadrants[1] = 1;
        if (b->data[0] <  0 && b->data[1] <  0) quadrants[2] = 1;
        if (b->data[0] >= 0 && b->data[1] <  0) quadrants[3] = 1;
    }
    
    int total = quadrants[0] + quadrants[1] + quadrants[2] + quadrants[3];
    
    if (total == 4) {
        printf("Found points in all quadrants!\n");
        return 1;
    }
    
    return 0;
}

/* Print population statistics */
void print_stats(novelty_search_t* ns) {
    if (!ns || !ns->stats) return;
    
    printf("  Population stats:\n");
    printf("    Centroid: (%.2f, %.2f)\n", 
           ns->stats->centroid[0], ns->stats->centroid[1]);
    printf("    Std dev:  (%.2f, %.2f)\n", 
           ns->stats->std_dev[0], ns->stats->std_dev[1]);
    printf("    Bounds:   [%.2f, %.2f] x [%.2f, %.2f]\n",
           ns->stats->min_bounds[0], ns->stats->max_bounds[0],
           ns->stats->min_bounds[1], ns->stats->max_bounds[1]);
    printf("    Coverage:  %.2f, Diversity: %.2f\n",
           ns->stats->coverage, ns->stats->diversity);
}

/* Save archive to file for visualization */
void save_archive_visualization(novelty_archive_t* archive, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    
    /* Write header */
    fprintf(fp, "x,y,novelty,fitness\n");
    
    /* Write data points */
    for (size_t i = 0; i < archive->size; i++) {
        behavior_t* b = archive->items[i];
        fprintf(fp, "%f,%f,%f,%f\n", b->data[0], b->data[1], b->novelty, b->fitness);
    }
    
    fclose(fp);
    printf("Saved archive visualization to %s\n", filename);
}

int main() {
    /* Initialize random seed */
    srand((unsigned int)time(NULL));
    
    /* Create a population of random points */
    const size_t population_size = 100;
    point_t* population = (point_t*)malloc(population_size * sizeof(point_t));
    void** pop_ptrs = (void**)malloc(population_size * sizeof(void*));
    
    if (!population || !pop_ptrs) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    /* Initialize population with random points in [-1, 1] x [-1, 1] */
    for (size_t i = 0; i < population_size; i++) {
        population[i].x = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        population[i].y = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        pop_ptrs[i] = &population[i];
    }
    
    /* Configure novelty search */
    novelty_config_t config = novelty_get_default_config();
    config.behavior_size = 2;  /* 2D points */
    config.k = 10;             /* 10 nearest neighbors */
    config.threshold = 0.5f;   /* Initial novelty threshold */
    config.max_archive_size = 1000;
    config.verbose = 1;        /* Enable progress output */
    
    /* Create novelty search context */
    novelty_search_t* ns = novelty_search_create(&config, config.behavior_size);
    if (!ns) {
        fprintf(stderr, "Failed to create novelty search context\n");
        free(population);
        free(pop_ptrs);
        return 1;
    }
    
    printf("Starting Novelty Search with %zu points in 2D space\n", population_size);
    printf("Initial threshold: %.2f, k: %zu\n", config.threshold, config.k);
    
    /* Run novelty search */
    const size_t max_generations = 100;
    
    for (size_t gen = 0; gen < max_generations; gen++) {
        printf("\nGeneration %zu/%zu\n", gen + 1, max_generations);
        
        /* Evaluate and update behaviors */
        behavior_t* behaviors = (behavior_t*)calloc(population_size, sizeof(behavior_t));
        if (!behaviors) {
            fprintf(stderr, "Memory allocation failed\n");
            break;
        }
        
        /* Initialize behaviors */
        for (size_t i = 0; i < population_size; i++) {
            behaviors[i].data = (float*)calloc(2, sizeof(float));
            behaviors[i].size = 2;
            behaviors[i].extra_data = pop_ptrs[i];
        }
        
        /* Evaluate all points */
        for (size_t i = 0; i < population_size; i++) {
            evaluate_point(pop_ptrs[i], &behaviors[i].fitness, 
                         behaviors[i].data, behaviors[i].size, NULL);
        }
        
        /* Update novelty scores */
        update_novelty_scores(ns, behaviors, population_size);
        
        /* Update archive with novel behaviors */
        update_novelty_archive(ns, behaviors, population_size);
        
        /* Print progress */
        printf("  Archive size: %zu, Threshold: %.4f\n", 
               ns->archive->size, ns->config.threshold);
        
        /* Print stats */
        if (ns->stats) {
            print_stats(ns);
        }
        
        /* Save visualization every 10 generations */
        if ((gen % 10 == 0 || gen == max_generations - 1) && ns->archive->size > 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "novelty_gen_%03zu.csv", gen);
            save_archive_visualization(ns->archive, filename);
        }
        
        /* Check termination condition */
        if (should_terminate(ns, NULL)) {
            printf("Termination condition met!\n");
            break;
        }
        
        /* Create next generation */
        for (size_t i = 0; i < population_size; i++) {
            /* Simple mutation: add Gaussian noise */
            point_t* p = (point_t*)pop_ptrs[i];
            
            /* With probability ns->current_p, select based on novelty */
            if ((float)rand() / RAND_MAX < ns->current_p) {
                /* Select parent based on novelty */
                size_t parent_idx = rand() % population_size;
                float best_novelty = behaviors[parent_idx].novelty;
                
                /* Tournament selection for novelty */
                for (int t = 1; t < 5; t++) {
                    size_t idx = rand() % population_size;
                    if (behaviors[idx].novelty > best_novelty) {
                        best_novelty = behaviors[idx].novelty;
                        parent_idx = idx;
                    }
                }
                
                /* Copy from parent */
                point_t* parent = (point_t*)pop_ptrs[parent_idx];
                p->x = parent->x;
                p->y = parent->y;
            } else {
                /* Select parent based on fitness */
                size_t parent_idx = rand() % population_size;
                float best_fitness = behaviors[parent_idx].fitness;
                
                /* Tournament selection for fitness */
                for (int t = 1; t < 5; t++) {
                    size_t idx = rand() % population_size;
                    if (behaviors[idx].fitness > best_fitness) {
                        best_fitness = behaviors[idx].fitness;
                        parent_idx = idx;
                    }
                }
                
                /* Copy from parent */
                point_t* parent = (point_t*)pop_ptrs[parent_idx];
                p->x = parent->x;
                p->y = parent->y;
            }
            
            /* Apply mutation */
            float mutation_rate = 0.1f;
            float mutation_scale = 0.1f;
            
            if ((float)rand() / RAND_MAX < mutation_rate) {
                p->x += mutation_scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
            }
            
            if ((float)rand() / RAND_MAX < mutation_rate) {
                p->y += mutation_scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
            }
            
            /* Keep within bounds */
            p->x = fmaxf(-1.0f, fminf(1.0f, p->x));
            p->y = fmaxf(-1.0f, fminf(1.0f, p->y));
        }
        
        /* Clean up behaviors */
        for (size_t i = 0; i < population_size; i++) {
            if (behaviors[i].data) {
                free(behaviors[i].data);
            }
        }
        free(behaviors);
    }
    
    /* Print final results */
    printf("\nNovelty search completed after %zu generations\n", ns->generation);
    printf("Final archive size: %zu\n", ns->archive->size);
    
    /* Save final visualization */
    save_archive_visualization(ns->archive, "novelty_final.csv");
    
    /* Clean up */
    novelty_search_free(ns);
    free(population);
    free(pop_ptrs);
    
    printf("Done!\n");
    return 0;
}
