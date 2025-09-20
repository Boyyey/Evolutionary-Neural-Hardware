#ifndef NOVELTY_H
#define NOVELTY_H

#include "neat.h"
#include <stddef.h>

/* Distance metric types */
#define NOVELTY_DIST_EUCLIDEAN 0
#define NOVELTY_DIST_MANHATTAN 1
#define NOVELTY_DIST_HAMMING   2
#define NOVELTY_DIST_COSINE    3

/* Archive file format constants */
#define NOVELTY_ARCHIVE_MAGIC   0x4E4F5645  /* 'NOVE' */
#define NOVELTY_ARCHIVE_VERSION 1

/* 
 * Novelty Search Configuration
 * Configures the behavior of the novelty search algorithm
 */
typedef struct {
    size_t k;                   /* Number of nearest neighbors to consider */
    float threshold;            /* Novelty threshold for adding to archive */
    size_t max_archive_size;    /* Maximum size of the novelty archive */
    float p_min;               /* Minimum probability of selecting for novelty */
    float p_max;               /* Maximum probability of selecting for novelty */
    float p_adjust_rate;       /* Rate at which to adjust p between p_min and p_max */
    float distance_metric;     /* Distance metric to use (0=Euclidean, 1=Manhattan, etc.) */
    int dynamic_threshold;     /* Whether to use dynamic threshold adjustment */
    float threshold_adjust_rate; /* Rate at which to adjust the threshold */
    float threshold_min;       /* Minimum threshold value */
    float threshold_max;       /* Maximum threshold value */
    int use_fitness_novelty;   /* Whether to combine fitness and novelty */
    float fitness_weight;      /* Weight for fitness in combined objective */
    float novelty_weight;      /* Weight for novelty in combined objective */
    int normalize_behavior;    /* Whether to normalize behavior vectors */
    int behavior_size;         /* Size of the behavior characterization vector */
    int use_local_competition; /* Whether to use local competition */
    int local_competition_size; /* Size of local competition neighborhood */
    int use_curvature;         /* Whether to use curvature information */
    int use_adaptive_parameters; /* Whether to adapt parameters during search */
    int use_parallel_evaluation; /* Whether to evaluate in parallel */
    int num_threads;           /* Number of threads for parallel evaluation */
    int save_archive;          /* Whether to save the archive */
    const char* archive_file;  /* File to save/load the archive */
    int verbose;               /* Verbosity level (0=none, 1=basic, 2=detailed) */
} novelty_config_t;

/* 
 * Behavior Characterization
 * Represents the behavior of an individual in the behavior space
 */
typedef struct {
    float* data;               /* Behavior vector */
    size_t size;               /* Size of the behavior vector */
    float novelty;             /* Novelty score */
    float fitness;             /* Fitness score */
    float combined_score;      /* Combined fitness and novelty score */
    size_t id;                 /* Unique identifier */
    void* extra_data;          /* Extra data associated with the behavior */
} behavior_t;

/* 
 * Novelty Archive
 * Stores novel individuals for reference in novelty calculations
 */
typedef struct {
    behavior_t** items;         /* Array of pointers to behaviors in the archive */
    size_t size;                /* Current number of items in the archive */
    size_t capacity;            /* Maximum capacity of the archive */
    size_t next_id;             /* Next available ID for new items */
    float current_threshold;    /* Current novelty threshold */
    float* min_bounds;          /* Minimum bounds for each dimension */
    float* max_bounds;          /* Maximum bounds for each dimension */
    int dimensions;             /* Dimensionality of the behavior space */
    int normalized;             /* Whether the behavior space is normalized */
    float* mean;                /* Mean of each dimension (for normalization) */
    float* std_dev;             /* Standard deviation of each dimension */
    size_t* recent_additions;   /* Indices of recently added items */
    size_t num_recent;          /* Number of recently added items */
    size_t max_recent;          /* Maximum number of recent items to track */
    void* extra_data;           /* Extra data associated with the archive */
    size_t k;                   /* Number of nearest neighbors to consider */
} novelty_archive_t;

/* 
 * Population Statistics
 * Tracks statistics about the population's behavior
 */
typedef struct {
    float* centroid;            /* Centroid of the population in behavior space */
    float* std_dev;             /* Standard deviation in each dimension */
    float* min_bounds;          /* Minimum bounds in each dimension */
    float* max_bounds;          /* Maximum bounds in each dimension */
    float coverage;             /* Coverage of the behavior space */
    float diversity;            /* Diversity of the population */
    size_t size;                /* Size of the behavior space */
} population_stats_t;

/* 
 * Novelty Search Context
 * Main structure for managing the novelty search
 */
typedef struct {
    novelty_config_t config;    /* Configuration parameters */
    novelty_archive_t* archive; /* Archive of novel individuals */
    population_stats_t* stats;  /* Population statistics */
    float current_p;            /* Current probability of selecting for novelty */
    float* distance_cache;      /* Cache for distance calculations */
    size_t cache_size;          /* Size of the distance cache */
    void* user_data;            /* User-defined data */
    int generation;             /* Current generation */
    float* distance_matrix;     /* Matrix of distances between individuals */
    size_t matrix_size;         /* Size of the distance matrix */
    int* nearest_neighbors;     /* Array of nearest neighbor indices */
    float* neighbor_distances;  /* Distances to nearest neighbors */
    size_t* selection_pool;     /* Pool of individuals for selection */
    size_t pool_size;           /* Size of the selection pool */
    float* behavior_buffer;     /* Buffer for behavior vectors */
    size_t buffer_size;         /* Size of the behavior buffer */
    int* behavior_indices;      /* Indices for behavior vectors */
    size_t num_indices;         /* Number of indices */
    float* temp_distances;      /* Temporary storage for distances */
    size_t temp_size;           /* Size of temporary storage */
    
    /* User-defined callback functions */
    novelty_alloc_func_t user_alloc_func;      /* Memory allocation function */
    novelty_free_func_t user_free_func;        /* Memory deallocation function */
    distance_func_t user_distance_func;        /* Custom distance function */
    behavior_func_t user_behavior_func;        /* Behavior extraction function */
    fitness_func_t user_fitness_func;          /* Fitness evaluation function */
    mutation_func_t user_mutation_func;        /* Mutation function */
    crossover_func_t user_crossover_func;      /* Crossover function */
    initialization_func_t user_init_func;      /* Initialization function */
    evaluation_func_t user_eval_func;          /* Evaluation function */
    visualization_func_t user_visualize_func;  /* Visualization function */
    
    /* Internal state */
    int initialized;            /* Whether the search has been initialized */
    size_t behavior_size;       /* Size of behavior vectors */
    size_t population_size;     /* Current population size */
    float best_fitness;         /* Best fitness found so far */
    float avg_fitness;          /* Average fitness of current population */
    float best_novelty;         /* Best novelty score found so far */
    float avg_novelty;          /* Average novelty of current population */
    float* behavior_min_bounds; /* Minimum bounds for behavior space */
    float* behavior_max_bounds; /* Maximum bounds for behavior space */
    int generation;             /* Current generation number */
    int max_generations;        /* Maximum number of generations to run */
    int num_evaluations;        /* Number of evaluations performed */
    int max_evaluations;        /* Maximum number of evaluations */
    float mutation_rate;        /* Current mutation rate */
    float crossover_rate;       /* Current crossover rate */
    int elitism;                /* Number of elites to preserve */
    int verbose;                /* Verbosity level */
    int save_frequency;         /* How often to save checkpoints */
    char* checkpoint_dir;       /* Directory to save checkpoints */
    int checkpoint_count;       /* Number of checkpoints saved */
    double start_time;          /* Start time of search */
    double last_checkpoint;     /* Time of last checkpoint */
} novelty_search_t;

/* 
 * Function pointer types for user-defined functions
 */
typedef void* (*novelty_alloc_func_t)(size_t size);
typedef void (*novelty_free_func_t)(void* ptr);
typedef float (*distance_func_t)(const float* a, const float* b, size_t size, void* user_data);
typedef void (*behavior_func_t)(const void* individual, float* behavior, size_t size, void* user_data);
typedef float (*fitness_func_t)(const void* individual, void* user_data);
typedef int (*termination_func_t)(novelty_search_t* ns, void* user_data);
typedef void (*mutation_func_t)(void* individual, float rate, void* user_data);
typedef void* (*crossover_func_t)(const void* parent1, const void* parent2, void* user_data);
typedef void* (*initialization_func_t)(size_t index, void* user_data);
typedef void (*evaluation_func_t)(void* individual, float* fitness, float* behavior, size_t behavior_size, void* user_data);
typedef void (*visualization_func_t)(novelty_search_t* ns, void* user_data);

/* 
 * Novelty Search API Functions
 */

/* Initialization and cleanup */
novelty_search_t* novelty_search_create(const novelty_config_t* config, size_t behavior_size);
void novelty_search_free(novelty_search_t* ns);
void novelty_search_init(novelty_search_t* ns, const novelty_config_t* config, size_t behavior_size);
void novelty_search_cleanup(novelty_search_t* ns);

/* Behavior characterization */
behavior_t* behavior_create(size_t size);
void behavior_free(behavior_t* behavior);
void behavior_copy(behavior_t* dest, const behavior_t* src);
float behavior_distance(const behavior_t* a, const behavior_t* b, distance_func_t dist_func, void* user_data);
void behavior_normalize(behavior_t* behavior, const float* min_bounds, const float* max_bounds, size_t size);
void behavior_denormalize(behavior_t* behavior, const float* min_bounds, const float* max_bounds, size_t size);

/* Novelty archive management */
novelty_archive_t* novelty_archive_create(size_t capacity, size_t behavior_size);
void novelty_archive_free(novelty_archive_t* archive);
int novelty_archive_add(novelty_archive_t* archive, const behavior_t* behavior, float threshold);
void novelty_archive_update(novelty_archive_t* archive, const behavior_t* behaviors, size_t count, float threshold);
void novelty_archive_prune(novelty_archive_t* archive, size_t max_size);
void novelty_archive_save(const novelty_archive_t* archive, const char* filename);
int novelty_archive_load(novelty_archive_t* archive, const char* filename);

/* Novelty calculation */
float calculate_novelty(const behavior_t* behavior, const novelty_archive_t* archive, size_t k, distance_func_t dist_func, void* user_data);
float* calculate_novelty_batch(const behavior_t* behaviors, size_t count, const novelty_archive_t* archive, size_t k, distance_func_t dist_func, void* user_data);
void update_novelty_scores(novelty_search_t* ns, behavior_t* behaviors, size_t count);

/* Population management */
void update_population_stats(novelty_search_t* ns, const behavior_t* behaviors, size_t count);
void update_novelty_archive(novelty_search_t* ns, const behavior_t* behaviors, size_t count);
void adjust_novelty_threshold(novelty_search_t* ns);
void adjust_selection_probability(novelty_search_t* ns, float improvement_rate);

/* Selection and reproduction */
void novelty_based_selection(novelty_search_t* ns, const behavior_t* behaviors, size_t count, size_t* selected, size_t num_to_select);
void fitness_novelty_selection(novelty_search_t* ns, const behavior_t* behaviors, size_t count, float* fitness_scores, size_t* selected, size_t num_to_select);
void tournament_selection(const behavior_t* behaviors, size_t count, size_t tournament_size, size_t* selected, size_t num_to_select);

/* Evolution */
void novelty_search_step(novelty_search_t* ns, void** population, size_t population_size, evaluation_func_t eval_func, void* user_data);
void novelty_search_run(novelty_search_t* ns, void** population, size_t population_size, size_t max_generations, evaluation_func_t eval_func, termination_func_t term_func, void* user_data);

/* Distance metrics */
float euclidean_distance(const float* a, const float* b, size_t size, void* user_data);
float manhattan_distance(const float* a, const float* b, size_t size, void* user_data);
float hamming_distance(const float* a, const float* b, size_t size, void* user_data);
float cosine_distance(const float* a, const float* b, size_t size, void* user_data);

/* Utility functions */
float* calculate_centroid(const behavior_t* behaviors, size_t count, size_t behavior_size);
float* calculate_std_dev(const behavior_t* behaviors, size_t count, const float* centroid, size_t behavior_size);
void normalize_behavior_space(novelty_search_t* ns, behavior_t* behaviors, size_t count);
void denormalize_behavior_space(novelty_search_t* ns, behavior_t* behaviors, size_t count);

/* Visualization */
void visualize_behavior_space(novelty_search_t* ns, const behavior_t* behaviors, size_t count, const char* filename);
void visualize_archive(novelty_search_t* ns, const char* filename);
void visualize_population_stats(novelty_search_t* ns, const char* filename);

/* Advanced features */
void novelty_search_with_curvature(novelty_search_t* ns, void** population, size_t population_size, evaluation_func_t eval_func, void* user_data);
void adaptive_novelty_search(novelty_search_t* ns, void** population, size_t population_size, evaluation_func_t eval_func, void* user_data);
void multi_archive_novelty_search(novelty_search_t** ns_array, size_t num_searches, void** population, size_t population_size, evaluation_func_t eval_func, void* user_data);

/* Parallel evaluation */
void parallel_evaluate(novelty_search_t* ns, void** population, size_t population_size, evaluation_func_t eval_func, void* user_data);

/* Callbacks */
void novelty_search_set_callbacks(
    novelty_search_t* ns,
    novelty_alloc_func_t alloc_func,
    novelty_free_func_t free_func,
    distance_func_t distance_func,
    behavior_func_t behavior_func,
    fitness_func_t fitness_func,
    void* user_data
);

/* Default configuration */
novelty_config_t novelty_get_default_config(void);

/* Utility macros */
#define NOVELTY_MIN(a, b) ((a) < (b) ? (a) : (b))
#define NOVELTY_MAX(a, b) ((a) > (b) ? (a) : (b))
#define NOVELTY_CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#define NOVELTY_ALLOC(ns, type, count) ((type*)(ns)->alloc_func((count) * sizeof(type)))
#define NOVELTY_FREE(ns, ptr) do { if ((ns)->free_func) (ns)->free_func(ptr); } while (0)

/* Error codes */
#define NOVELTY_SUCCESS 0
#define NOVELTY_ERROR_INVALID_ARGUMENT -1
#define NOVELTY_ERROR_MEMORY -2
#define NOVELTY_ERROR_IO -3
#define NOVELTY_ERROR_NOT_INITIALIZED -4
#define NOVELTY_ERROR_ARCHIVE_FULL -5

/* Logging */
#ifdef NOVELTY_DEBUG
#define NOVELTY_LOG(ns, level, ...) \
    do { \
        if ((ns) && (ns)->config.verbose >= (level)) { \
            printf(__VA_ARGS__); \
            printf("\n"); \
            fflush(stdout); \
        } \
    } while (0)
#else
#define NOVELTY_LOG(ns, level, ...) ((void)0)
#endif

/* Version information */
#define NOVELTY_VERSION_MAJOR 1
#define NOVELTY_VERSION_MINOR 0
#define NOVELTY_VERSION_PATCH 0

const char* novelty_get_version_string(void);
void novelty_get_version(int* major, int* minor, int* patch);

/* Deprecated functions */
#ifndef NOVELTY_NO_DEPRECATED
/* These functions are kept for backward compatibility but may be removed in future versions */
float calculate_novelty_score(const behavior_t* behavior, const novelty_archive_t* archive, size_t k) 
    __attribute__((deprecated("Use calculate_novelty instead")));
#endif

#endif /* NOVELTY_H */
