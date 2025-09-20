#include "../include/novelty.h"
#include "../include/neat.h"
#include "../include/simd_math.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>

/* Default configuration for novelty search */
novelty_config_t novelty_get_default_config(void) {
    novelty_config_t config = {0};
    
    /* Basic parameters */
    config.k = 15;                    /* Number of nearest neighbors */
    config.threshold = 6.0f;          /* Initial novelty threshold */
    config.max_archive_size = 1000;   /* Maximum archive size */
    
    /* Probability parameters */
    config.p_min = 0.1f;              /* Minimum probability of selecting for novelty */
    config.p_max = 0.9f;              /* Maximum probability of selecting for novelty */
    config.p_adjust_rate = 0.01f;     /* Rate of probability adjustment */
    
    /* Distance metric */
    config.distance_metric = NOVELTY_DIST_EUCLIDEAN;
    
    /* Dynamic threshold adjustment */
    config.dynamic_threshold = 1;     /* Enable dynamic threshold */
    config.threshold_adjust_rate = 0.1f;
    config.threshold_min = 1.0f;
    config.threshold_max = 20.0f;
    
    /* Fitness-novelty combination */
    config.use_fitness_novelty = 1;   /* Combine fitness and novelty */
    config.fitness_weight = 0.5f;     /* Weight for fitness */
    config.novelty_weight = 0.5f;     /* Weight for novelty */
    
    /* Behavior characterization */
    config.normalize_behavior = 1;    /* Normalize behavior vectors */
    config.behavior_size = 10;        /* Default behavior vector size */
    
    /* Local competition */
    config.use_local_competition = 1; /* Use local competition */
    config.local_competition_size = 10;
    
    /* Advanced features */
    config.use_curvature = 0;         /* Use curvature information */
    config.use_adaptive_parameters = 1;/* Adapt parameters during search */
    
    /* Parallel execution */
    config.use_parallel_evaluation = 1;
    config.num_threads = 4;           /* Default number of threads */
    
    /* I/O */
    config.save_archive = 1;          /* Save archive to file */
    config.archive_file = "novelty_archive.bin";
    
    /* Verbosity */
    config.verbose = 1;               /* Default verbosity level */
    
    return config;
}

/* Calculate Euclidean distance between two behavior vectors */
float euclidean_distance(const float* a, const float* b, size_t size, void* user_data) {
    (void)user_data; /* Unused parameter */
    float sum = 0.0f;
    
    #ifdef USE_SIMD
    /* Use SIMD for faster distance calculation if available */
    sum = simd_euclidean_distance(a, b, size);
    #else
    /* Standard implementation */
    for (size_t i = 0; i < size; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    #endif
    
    return sqrtf(sum);
}

/* Calculate Manhattan distance between two behavior vectors */
float manhattan_distance(const float* a, const float* b, size_t size, void* user_data) {
    (void)user_data; /* Unused parameter */
    float sum = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        sum += fabsf(a[i] - b[i]);
    }
    
    return sum;
}

/* Calculate Hamming distance between two behavior vectors */
float hamming_distance(const float* a, const float* b, size_t size, void* user_data) {
    (void)user_data; /* Unused parameter */
    float sum = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        sum += (a[i] != b[i]) ? 1.0f : 0.0f;
    }
    
    return sum;
}

/* Calculate cosine distance between two behavior vectors */
float cosine_distance(const float* a, const float* b, size_t size, void* user_data) {
    (void)user_data; /* Unused parameter */
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);
    
    if (norm_a < 1e-10f || norm_b < 1e-10f) {
        return 1.0f; /* Maximum distance if either vector is zero */
    }
    
    return 1.0f - (dot / (norm_a * norm_b));
}

/* Get the appropriate distance function based on configuration */
static distance_func_t get_distance_function(novelty_search_t* ns) {
    if (ns->user_distance_func) {
        return ns->user_distance_func;
    }
    
    int metric = (int)ns->config.distance_metric;
    switch (metric) {
        case NOVELTY_DIST_MANHATTAN:
            return manhattan_distance;
        case NOVELTY_DIST_HAMMING:
            return hamming_distance;
        case NOVELTY_DIST_COSINE:
            return cosine_distance;
        case NOVELTY_DIST_EUCLIDEAN:
        default:
            return euclidean_distance;
    }
}

/* Create a new behavior */
behavior_t* behavior_create(size_t size) {
    behavior_t* behavior = (behavior_t*)calloc(1, sizeof(behavior_t));
    if (!behavior) return NULL;
    
    behavior->data = (float*)calloc(size, sizeof(float));
    if (!behavior->data) {
        free(behavior);
        return NULL;
    }
    
    behavior->size = size;
    behavior->novelty = 0.0f;
    behavior->fitness = 0.0f;
    behavior->combined_score = 0.0f;
    behavior->id = 0;
    behavior->extra_data = NULL;
    
    return behavior;
}

/* Free a behavior */
void behavior_free(behavior_t* behavior) {
    if (!behavior) return;
    
    if (behavior->data) {
        free(behavior->data);
    }
    
    free(behavior);
}

/* Copy behavior from src to dest */
void behavior_copy(behavior_t* dest, const behavior_t* src) {
    if (!dest || !src || dest->size != src->size) return;
    
    memcpy(dest->data, src->data, src->size * sizeof(float));
    dest->novelty = src->novelty;
    dest->fitness = src->fitness;
    dest->combined_score = src->combined_score;
    dest->id = src->id;
    /* Note: extra_data is not deep copied */
}

/* Normalize behavior vector */
void behavior_normalize(behavior_t* behavior, const float* min_bounds, const float* max_bounds, size_t size) {
    if (!behavior || !min_bounds || !max_bounds || behavior->size < size) return;
    
    for (size_t i = 0; i < size; i++) {
        float range = max_bounds[i] - min_bounds[i];
        if (range > 1e-10f) {
            behavior->data[i] = (behavior->data[i] - min_bounds[i]) / range;
        } else {
            behavior->data[i] = 0.0f;
        }
    }
}

/* Denormalize behavior vector */
void behavior_denormalize(behavior_t* behavior, const float* min_bounds, const float* max_bounds, size_t size) {
    if (!behavior || !min_bounds || !max_bounds || behavior->size < size) return;
    
    for (size_t i = 0; i < size; i++) {
        float range = max_bounds[i] - min_bounds[i];
        behavior->data[i] = min_bounds[i] + behavior->data[i] * range;
    }
}

/* Create a new novelty archive */
novelty_archive_t* novelty_archive_create(size_t capacity, size_t behavior_size) {
    if (capacity == 0 || behavior_size == 0) {
        return NULL;
    }
    
    novelty_archive_t* archive = (novelty_archive_t*)calloc(1, sizeof(novelty_archive_t));
    if (!archive) {
        return NULL;
    }
    
    archive->items = (behavior_t**)calloc(capacity, sizeof(behavior_t*));
    if (!archive->items) {
        free(archive);
        return NULL;
    }
    
    archive->recent_additions = (size_t*)calloc(capacity, sizeof(size_t));
    if (!archive->recent_additions) {
        free(archive->items);
        free(archive);
        return NULL;
    }
    
    archive->min_bounds = (float*)calloc(behavior_size, sizeof(float));
    archive->max_bounds = (float*)calloc(behavior_size, sizeof(float));
    archive->mean = (float*)calloc(behavior_size, sizeof(float));
    archive->std_dev = (float*)calloc(behavior_size, sizeof(float));
    
    if (!archive->min_bounds || !archive->max_bounds || !archive->mean || !archive->std_dev) {
        free(archive->min_bounds);
        free(archive->max_bounds);
        free(archive->mean);
        free(archive->std_dev);
        free(archive->recent_additions);
        free(archive->items);
        free(archive);
        return NULL;
    }
    
    archive->capacity = capacity;
    archive->size = 0;
    archive->next_id = 0;
    archive->dimensions = behavior_size;
    archive->normalized = 0;
    archive->num_recent = 0;
    archive->max_recent = 10;  /* Default max recent items */
    archive->current_threshold = 0.0f;
    archive->k = 15;  /* Default number of nearest neighbors */
    
    /* Initialize bounds to extreme values */
    for (size_t i = 0; i < behavior_size; i++) {
        archive->min_bounds[i] = FLT_MAX;
        archive->max_bounds[i] = -FLT_MAX;
        archive->mean[i] = 0.0f;
        archive->std_dev[i] = 0.0f;
    }
    
    return archive;
}

/* Free a novelty archive */
void novelty_archive_free(novelty_archive_t* archive) {
    if (!archive) return;
    
    if (archive->items) {
        for (size_t i = 0; i < archive->size; i++) {
            if (archive->items[i]) {
                behavior_free(archive->items[i]);
            }
        }
        free(archive->items);
    }
    
    if (archive->min_bounds) free(archive->min_bounds);
    if (archive->max_bounds) free(archive->max_bounds);
    if (archive->mean) free(archive->mean);
    if (archive->std_dev) free(archive->std_dev);
    if (archive->recent_additions) free(archive->recent_additions);
    
    free(archive);
}

/* Add a behavior to the archive if it's novel enough */
int novelty_archive_add(novelty_archive_t* archive, const behavior_t* behavior, float threshold) {
    if (!archive || !behavior || behavior->size != (size_t)archive->dimensions) {
        return -1;
    }
    
    /* If archive is full, remove the oldest item */
    if (archive->size >= archive->capacity) {
        if (archive->items[0]) {
            behavior_free(archive->items[0]);
        }
        memmove(&archive->items[0], &archive->items[1], (archive->size - 1) * sizeof(behavior_t*));
        archive->size--;
    }
    
    /* Allocate space for the new behavior */
    behavior_t* new_behavior = behavior_create(behavior->size);
    if (!new_behavior) {
        return -1;
    }
    
    /* Copy the behavior data */
    behavior_copy(new_behavior, behavior);
    
    /* Add to archive */
    archive->items[archive->size] = new_behavior;
    archive->size++;
    
    /* Update bounds */
    for (int i = 0; i < archive->dimensions; i++) {
        if (behavior->data[i] < archive->min_bounds[i]) {
            archive->min_bounds[i] = behavior->data[i];
        }
        if (behavior->data[i] > archive->max_bounds[i]) {
            archive->max_bounds[i] = behavior->data[i];
        }
    }
    
    return 1;
}

/* Update the archive with new behaviors */
void novelty_archive_update(novelty_archive_t* archive, const behavior_t* behaviors, 
                           size_t count, float threshold) {
    if (!archive || !behaviors) return;
    
    for (size_t i = 0; i < count; i++) {
        /* Calculate novelty score */
        float novelty = calculate_novelty(&behaviors[i], archive, archive->k, 
                                         euclidean_distance, NULL);
        
        /* Add to archive if novel enough */
        if (novelty > threshold) {
            novelty_archive_add(archive, &behaviors[i], threshold);
        }
    }
}

/* Prune the archive to the given size */
void novelty_archive_prune(novelty_archive_t* archive, size_t max_size) {
    if (!archive || archive->size <= max_size) return;
    
    /* Simple pruning: keep the most recent items */
    size_t to_remove = archive->size - max_size;
    
    /* Free the first 'to_remove' behaviors */
    for (size_t i = 0; i < to_remove; i++) {
        behavior_free(archive->items[i]);
    }
    
    /* Shift remaining items */
    memmove(&archive->items[0], &archive->items[to_remove], 
           (archive->size - to_remove) * sizeof(behavior_t*));
    
    /* Update size */
    archive->size -= to_remove;
    
    /* Update recent additions */
    size_t write_pos = 0;
    for (size_t i = 0; i < archive->num_recent; i++) {
        if (archive->recent_additions[i] >= to_remove) {
            archive->recent_additions[write_pos++] = archive->recent_additions[i] - to_remove;
        }
    }
    archive->num_recent = write_pos;
}

/* Save the archive to a file */
int novelty_archive_save(const novelty_archive_t* archive, const char* filename) {
    if (!archive || !filename) {
        return 0;
    }
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        return 0;
    }
    
    /* Write header */
    uint32_t magic = NOVELTY_ARCHIVE_MAGIC;
    uint32_t version = NOVELTY_ARCHIVE_VERSION;
    
    if (fwrite(&magic, sizeof(uint32_t), 1, fp) != 1 ||
        fwrite(&version, sizeof(uint32_t), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }
    
    /* Write archive metadata */
    if (fwrite(&archive->size, sizeof(size_t), 1, fp) != 1 ||
        fwrite(&archive->capacity, sizeof(size_t), 1, fp) != 1 ||
        fwrite(&archive->dimensions, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }
    
    /* Write bounds */
    if (fwrite(archive->min_bounds, sizeof(float), archive->dimensions, fp) != (size_t)archive->dimensions ||
        fwrite(archive->max_bounds, sizeof(float), archive->dimensions, fp) != (size_t)archive->dimensions) {
        fclose(fp);
        return 0;
    }
    
    /* Write behaviors */
    for (size_t i = 0; i < archive->size; i++) {
        const behavior_t* b = archive->items[i];
        if (fwrite(&b->size, sizeof(size_t), 1, fp) != 1 ||
            fwrite(b->data, sizeof(float), b->size, fp) != b->size ||
            fwrite(&b->novelty, sizeof(float), 1, fp) != 1 ||
            fwrite(&b->fitness, sizeof(float), 1, fp) != 1) {
            fclose(fp);
            return 0;
        }
    }
    
    fclose(fp);
    return 1;
}

/* Load an archive from a file */
novelty_archive_t* novelty_archive_load(const char* filename) {
    if (!filename) return NULL;
    
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    /* Read and verify header */
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&version, sizeof(uint32_t), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }
    
    if (magic != NOVELTY_ARCHIVE_MAGIC || version != NOVELTY_ARCHIVE_VERSION) {
        fclose(fp);
        return NULL;
    }
    
    /* Read archive metadata */
    size_t size, capacity;
    int dimensions;
    if (fread(&size, sizeof(size_t), 1, fp) != 1 ||
        fread(&capacity, sizeof(size_t), 1, fp) != 1 ||
        fread(&dimensions, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }
    
    /* Create a new archive */
    novelty_archive_t* archive = novelty_archive_create(capacity, dimensions);
    if (!archive) {
        fclose(fp);
        return NULL;
    }
    
    /* Read bounds */
    if (fread(archive->min_bounds, sizeof(float), dimensions, fp) != (size_t)dimensions ||
        fread(archive->max_bounds, sizeof(float), dimensions, fp) != (size_t)dimensions) {
        novelty_archive_free(archive);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    return archive;
}

/* Calculate the novelty of a behavior */
float calculate_novelty(const behavior_t* behavior, const novelty_archive_t* archive, 
                       size_t k, distance_func_t dist_func, void* user_data) {
    if (!behavior || !archive || archive->size == 0 || k == 0) {
        return 0.0f;
    }
    
    /* Use default distance function if none provided */
    if (!dist_func) {
        dist_func = euclidean_distance;
    }
    
    /* Find k-nearest neighbors */
    size_t num_neighbors = (k < archive->size) ? k : archive->size;
    float* distances = (float*)calloc(archive->size, sizeof(float));
    size_t* indices = (size_t*)calloc(archive->size, sizeof(size_t));
    
    if (!distances || !indices) {
        if (distances) free(distances);
        if (indices) free(indices);
        return 0.0f;
    }
    
    /* Calculate distances to all items in the archive */
    for (size_t i = 0; i < archive->size; i++) {
        distances[i] = dist_func(behavior->data, archive->items[i]->data, 
                               behavior->size, user_data);
        indices[i] = i;
    }
    
    /* Sort distances (partial sort would be more efficient) */
    for (size_t i = 0; i < num_neighbors; i++) {
        size_t min_idx = i;
        for (size_t j = i + 1; j < archive->size; j++) {
            if (distances[j] < distances[min_idx]) {
                min_idx = j;
            }
        }
        
        /* Swap */
        if (min_idx != i) {
            float tmp_dist = distances[i];
            size_t tmp_idx = indices[i];
            
            distances[i] = distances[min_idx];
            indices[i] = indices[min_idx];
            
            distances[min_idx] = tmp_dist;
            indices[min_idx] = tmp_idx;
        }
    }
    
    /* Calculate average distance to k-nearest neighbors */
    float sum = 0.0f;
    for (size_t i = 0; i < num_neighbors; i++) {
        sum += distances[i];
    }
    
    free(distances);
    free(indices);
    
    return sum / num_neighbors;
}

/* Calculate novelty scores for multiple behaviors */
float* calculate_novelty_batch(const behavior_t* behaviors, size_t count, 
                              const novelty_archive_t* archive, size_t k, 
                              distance_func_t dist_func, void* user_data) {
    if (!behaviors || count == 0 || !archive) return NULL;
    
    float* scores = (float*)calloc(count, sizeof(float));
    if (!scores) return NULL;
    
    for (size_t i = 0; i < count; i++) {
        scores[i] = calculate_novelty(&behaviors[i], archive, k, dist_func, user_data);
    }
    
    return scores;
}

/* Update novelty scores for a population */
void update_novelty_scores(novelty_search_t* ns, behavior_t* behaviors, size_t count) {
    if (!ns || !behaviors || count == 0) return;
    
    distance_func_t dist_func = get_distance_function(ns);
    
    /* Calculate novelty for each behavior */
    for (size_t i = 0; i < count; i++) {
        behaviors[i].novelty = calculate_novelty(
            &behaviors[i], ns->archive, 
            ns->config.k, dist_func, ns->user_data
        );
        
        /* Update combined score if using fitness-novelty combination */
        if (ns->config.use_fitness_novelty) {
            behaviors[i].combined_score = 
                ns->config.fitness_weight * behaviors[i].fitness +
                ns->config.novelty_weight * behaviors[i].novelty;
        } else {
            behaviors[i].combined_score = behaviors[i].novelty;
        }
    }
}

/* Update population statistics */
void update_population_stats(novelty_search_t* ns, const behavior_t* behaviors, size_t count) {
    if (!ns || !behaviors || count == 0 || behaviors[0].size == 0) return;
    
    size_t dims = behaviors[0].size;
    
    /* Initialize stats if needed */
    if (!ns->stats) {
        ns->stats = (population_stats_t*)calloc(1, sizeof(population_stats_t));
        if (!ns->stats) return;
        
        ns->stats->centroid = (float*)calloc(dims, sizeof(float));
        ns->stats->std_dev = (float*)calloc(dims, sizeof(float));
        ns->stats->min_bounds = (float*)calloc(dims, sizeof(float));
        ns->stats->max_bounds = (float*)calloc(dims, sizeof(float));
        ns->stats->size = dims;
        
        if (!ns->stats->centroid || !ns->stats->std_dev || 
            !ns->stats->min_bounds || !ns->stats->max_bounds) {
            if (ns->stats->centroid) free(ns->stats->centroid);
            if (ns->stats->std_dev) free(ns->stats->std_dev);
            if (ns->stats->min_bounds) free(ns->stats->min_bounds);
            if (ns->stats->max_bounds) free(ns->stats->max_bounds);
            free(ns->stats);
            ns->stats = NULL;
            return;
        }
        
        /* Initialize bounds */
        for (size_t i = 0; i < dims; i++) {
            ns->stats->min_bounds[i] = FLT_MAX;
            ns->stats->max_bounds[i] = -FLT_MAX;
        }
    }
    
    /* Reset stats */
    memset(ns->stats->centroid, 0, dims * sizeof(float));
    
    /* Calculate centroid and update bounds */
    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < dims; j++) {
            float val = behaviors[i].data[j];
            ns->stats->centroid[j] += val;
            
            if (val < ns->stats->min_bounds[j]) ns->stats->min_bounds[j] = val;
            if (val > ns->stats->max_bounds[j]) ns->stats->max_bounds[j] = val;
        }
    }
    
    /* Calculate mean */
    for (size_t j = 0; j < dims; j++) {
        ns->stats->centroid[j] /= count;
    }
    
    /* Calculate standard deviation */
    memset(ns->stats->std_dev, 0, dims * sizeof(float));
    
    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < dims; j++) {
            float diff = behaviors[i].data[j] - ns->stats->centroid[j];
            ns->stats->std_dev[j] += diff * diff;
        }
    }
    
    for (size_t j = 0; j < dims; j++) {
        ns->stats->std_dev[j] = sqrtf(ns->stats->std_dev[j] / count);
    }
    
    /* Calculate coverage and diversity */
    ns->stats->coverage = 0.0f;
    ns->stats->diversity = 0.0f;
    
    for (size_t j = 0; j < dims; j++) {
        float range = ns->stats->max_bounds[j] - ns->stats->min_bounds[j];
        if (range > 1e-10f) {
            ns->stats->coverage += ns->stats->std_dev[j] / range;
        }
    }
    
    ns->stats->coverage /= dims;
    
    /* Calculate average distance between individuals */
    distance_func_t dist_func = get_distance_function(ns);
    size_t num_pairs = 0;
    
    for (size_t i = 0; i < count; i++) {
        for (size_t j = i + 1; j < count; j++) {
            ns->stats->diversity += dist_func(behaviors[i].data, behaviors[j].data, dims, ns->user_data);
            num_pairs++;
        }
    }
    
    if (num_pairs > 0) {
        ns->stats->diversity /= num_pairs;
    }
}

/* Update the novelty archive with new behaviors */
void update_novelty_archive(novelty_search_t* ns, const behavior_t* behaviors, size_t count) {
    if (!ns || !behaviors || count == 0) return;
    
    distance_func_t dist_func = get_distance_function(ns);
    
    for (size_t i = 0; i < count; i++) {
        /* Calculate novelty score */
        float novelty = calculate_novelty(
            &behaviors[i], ns->archive, 
            ns->config.k, dist_func, ns->user_data
        );
        
        /* Add to archive if novel enough */
        if (novelty > ns->config.threshold) {
            novelty_archive_add(ns->archive, &behaviors[i], ns->config.threshold);
        }
    }
    
    /* Adjust threshold if dynamic thresholding is enabled */
    if (ns->config.dynamic_threshold) {
        adjust_novelty_threshold(ns);
    }
}

/* Adjust the novelty threshold based on archive growth rate */
void adjust_novelty_threshold(novelty_search_t* ns) {
    if (!ns || !ns->archive) return;
    
    /* Simple threshold adjustment based on archive growth */
    static size_t last_archive_size = 0;
    
    if (ns->archive->size > last_archive_size) {
        /* Archive is growing too fast, increase threshold */
        ns->config.threshold *= (1.0f + ns->config.threshold_adjust_rate);
    } else {
        /* Archive is not growing fast enough, decrease threshold */
        ns->config.threshold *= (1.0f - ns->config.threshold_adjust_rate);
    }
    
    /* Clamp threshold to valid range */
    if (ns->config.threshold < ns->config.threshold_min) {
        ns->config.threshold = ns->config.threshold_min;
    } else if (ns->config.threshold > ns->config.threshold_max) {
        ns->config.threshold = ns->config.threshold_max;
    }
    
    last_archive_size = ns->archive->size;
}

/* Adjust the selection probability based on improvement rate */
void adjust_selection_probability(novelty_search_t* ns, float improvement_rate) {
    if (!ns) return;
    
    /* Simple probability adjustment based on improvement */
    if (improvement_rate > 0.0f) {
        /* Increasing probability of selecting for novelty */
        ns->current_p += ns->config.p_adjust_rate * (1.0f - ns->current_p);
    } else {
        /* Decreasing probability of selecting for novelty */
        ns->current_p -= ns->config.p_adjust_rate * ns->current_p;
    }
    
    /* Clamp probability to valid range */
    if (ns->current_p < ns->config.p_min) {
        ns->current_p = ns->config.p_min;
    } else if (ns->current_p > ns->config.p_max) {
        ns->current_p = ns->config.p_max;
    }
}

/* Select individuals based on novelty */
void novelty_based_selection(novelty_search_t* ns, const behavior_t* behaviors, 
                            size_t count, size_t* selected, size_t num_to_select) {
    if (!ns || !behaviors || !selected || count == 0 || num_to_select == 0) return;
    
    /* Use combined score if fitness-novelty combination is enabled */
    const char* score_type = ns->config.use_fitness_novelty ? "combined" : "novelty";
    
    /* Simple tournament selection based on novelty */
    for (size_t i = 0; i < num_to_select; i++) {
        size_t best_idx = 0;
        float best_score = -FLT_MAX;
        
        /* Tournament of size tournament_size */
        size_t tournament_size = (ns->config.local_competition_size < count) ? 
                               ns->config.local_competition_size : count;
        
        for (size_t j = 0; j < tournament_size; j++) {
            size_t idx = rand() % count;
            float score = (score_type[0] == 'c') ? 
                         behaviors[idx].combined_score : behaviors[idx].novelty;
            
            if (score > best_score) {
                best_score = score;
                best_idx = idx;
            }
        }
        
        selected[i] = best_idx;
    }
}

/* Select individuals based on both fitness and novelty */
void fitness_novelty_selection(novelty_search_t* ns, const behavior_t* behaviors, 
                              size_t count, float* fitness_scores, 
                              size_t* selected, size_t num_to_select) {
    if (!ns || !behaviors || !fitness_scores || !selected || 
        count == 0 || num_to_select == 0) {
        return;
    }
    
    /* Normalize fitness and novelty scores */
    float max_fitness = -FLT_MAX, min_fitness = FLT_MAX;
    float max_novelty = -FLT_MAX, min_novelty = FLT_MAX;
    
    for (size_t i = 0; i < count; i++) {
        if (fitness_scores[i] > max_fitness) max_fitness = fitness_scores[i];
        if (fitness_scores[i] < min_fitness) min_fitness = fitness_scores[i];
        if (behaviors[i].novelty > max_novelty) max_novelty = behaviors[i].novelty;
        if (behaviors[i].novelty < min_novelty) min_novelty = behaviors[i].novelty;
    }
    
    float fitness_range = max_fitness - min_fitness;
    float novelty_range = max_novelty - min_novelty;
    
    if (fitness_range < 1e-10f) fitness_range = 1.0f;
    if (novelty_range < 1e-10f) novelty_range = 1.0f;
    
    /* Calculate combined scores */
    float* combined_scores = (float*)malloc(count * sizeof(float));
    if (!combined_scores) return;
    
    for (size_t i = 0; i < count; i++) {
        float norm_fitness = (fitness_scores[i] - min_fitness) / fitness_range;
        float norm_novelty = (behaviors[i].novelty - min_novelty) / novelty_range;
        
        combined_scores[i] = ns->config.fitness_weight * norm_fitness +
                           ns->config.novelty_weight * norm_novelty;
    }
    
    /* Select based on combined scores */
    for (size_t i = 0; i < num_to_select; i++) {
        size_t best_idx = 0;
        float best_score = -FLT_MAX;
        
        /* Tournament selection */
        size_t tournament_size = (ns->config.local_competition_size < count) ? 
                               ns->config.local_competition_size : count;
        
        for (size_t j = 0; j < tournament_size; j++) {
            size_t idx = rand() % count;
            if (combined_scores[idx] > best_score) {
                best_score = combined_scores[idx];
                best_idx = idx;
            }
        }
        
        selected[i] = best_idx;
    }
    
    free(combined_scores);
}

/* Tournament selection */
void tournament_selection(const behavior_t* behaviors, size_t count, 
                         size_t tournament_size, size_t* selected, 
                         size_t num_to_select) {
    if (!behaviors || !selected || count == 0 || num_to_select == 0) return;
    
    if (tournament_size < 2) tournament_size = 2;
    if (tournament_size > count) tournament_size = count;
    
    for (size_t i = 0; i < num_to_select; i++) {
        size_t best_idx = rand() % count;
        float best_score = behaviors[best_idx].combined_score;
        
        for (size_t j = 1; j < tournament_size; j++) {
            size_t idx = rand() % count;
            if (behaviors[idx].combined_score > best_score) {
                best_score = behaviors[idx].combined_score;
                best_idx = idx;
            }
        }
        
        selected[i] = best_idx;
    }
}

/* Create a new novelty search context */
novelty_search_t* novelty_search_create(const novelty_config_t* config, size_t behavior_size) {
    if (!config || behavior_size == 0) return NULL;
    
    novelty_search_t* ns = (novelty_search_t*)calloc(1, sizeof(novelty_search_t));
    if (!ns) return NULL;
    
    /* Copy configuration */
    ns->config = *config;
    
    /* Create archive */
    ns->archive = novelty_archive_create(config->max_archive_size, behavior_size);
    if (!ns->archive) {
        free(ns);
        return NULL;
    }
    
    /* Initialize other fields */
    ns->current_p = config->p_min;
    ns->user_data = NULL;
    ns->stats = NULL;
    ns->user_distance_func = NULL;
    
    /* Allocate distance cache */
    ns->distance_cache = (float*)calloc(behavior_size * behavior_size, sizeof(float));
    ns->cache_size = behavior_size * behavior_size;
    
    if (!ns->distance_cache) {
        novelty_archive_free(ns->archive);
        free(ns);
        return NULL;
    }
    
    /* Initialize random number generator */
    srand((unsigned int)time(NULL));
    
    return ns;
}

/* Free a novelty search context */
void novelty_search_free(novelty_search_t* ns) {
    if (!ns) return;
    
    if (ns->archive) {
        novelty_archive_free(ns->archive);
    }
    
    if (ns->stats) {
        if (ns->stats->centroid) free(ns->stats->centroid);
        if (ns->stats->std_dev) free(ns->stats->std_dev);
        if (ns->stats->min_bounds) free(ns->stats->min_bounds);
        if (ns->stats->max_bounds) free(ns->stats->max_bounds);
        free(ns->stats);
    }
    
    if (ns->distance_cache) free(ns->distance_cache);
    if (ns->distance_matrix) free(ns->distance_matrix);
    if (ns->nearest_neighbors) free(ns->nearest_neighbors);
    if (ns->neighbor_distances) free(ns->neighbor_distances);
    if (ns->selection_pool) free(ns->selection_pool);
    if (ns->behavior_buffer) free(ns->behavior_buffer);
    if (ns->behavior_indices) free(ns->behavior_indices);
    if (ns->temp_distances) free(ns->temp_distances);
    
    free(ns);
}

/* Run one step of novelty search */
void novelty_search_step(novelty_search_t* ns, void** population, 
                        size_t population_size, evaluation_func_t eval_func, 
                        void* user_data) {
    if (!ns || !population || population_size == 0 || !eval_func) return;
    
    /* Evaluate all individuals to get behaviors */
    behavior_t* behaviors = (behavior_t*)calloc(population_size, sizeof(behavior_t));
    if (!behaviors) return;
    
    /* Initialize behaviors */
    for (size_t i = 0; i < population_size; i++) {
        behaviors[i].data = (float*)calloc(ns->archive->dimensions, sizeof(float));
        behaviors[i].size = ns->archive->dimensions;
        behaviors[i].extra_data = population[i];
    }
    
    /* Evaluate behaviors in parallel if enabled */
    if (ns->config.use_parallel_evaluation) {
        /* Simple parallel evaluation using OpenMP */
        #pragma omp parallel for num_threads(ns->config.num_threads)
        for (size_t i = 0; i < population_size; i++) {
            eval_func(population[i], &behaviors[i].fitness, 
                     behaviors[i].data, behaviors[i].size, user_data);
        }
    } else {
        /* Sequential evaluation */
        for (size_t i = 0; i < population_size; i++) {
            eval_func(population[i], &behaviors[i].fitness, 
                     behaviors[i].data, behaviors[i].size, user_data);
        }
    }
    
    /* Update novelty scores */
    update_novelty_scores(ns, behaviors, population_size);
    
    /* Update archive with novel behaviors */
    update_novelty_archive(ns, behaviors, population_size);
    
    /* Update population statistics */
    update_population_stats(ns, behaviors, population_size);
    
    /* Cleanup */
    for (size_t i = 0; i < population_size; i++) {
        if (behaviors[i].data) free(behaviors[i].data);
    }
    free(behaviors);
    
    /* Update generation counter */
    ns->generation++;
}

/* Run novelty search for multiple generations */
void novelty_search_run(novelty_search_t* ns, void** population, 
                       size_t population_size, size_t max_generations, 
                       evaluation_func_t eval_func, 
                       termination_func_t term_func, void* user_data) {
    if (!ns || !population || population_size == 0 || !eval_func) return;
    
    for (size_t gen = 0; gen < max_generations; gen++) {
        /* Run one step */
        novelty_search_step(ns, population, population_size, eval_func, user_data);
        
        /* Check termination condition */
        if (term_func && term_func(ns, user_data)) {
            break;
        }
        
        /* Print progress */
        if (ns->config.verbose > 0 && (gen % 10 == 0 || gen == max_generations - 1)) {
            printf("Generation %zu: Archive size = %zu, Threshold = %.4f\n",
                   gen + 1, ns->archive->size, ns->config.threshold);
        }
    }
}

/* Set callbacks and user data */
void novelty_search_set_callbacks(novelty_search_t* ns,
                                novelty_alloc_func_t alloc_func,
                                novelty_free_func_t free_func,
                                distance_func_t distance_func,
                                behavior_func_t behavior_func,
                                fitness_func_t fitness_func,
                                void* user_data) {
    if (!ns) return;
    
    ns->user_alloc_func = alloc_func;
    ns->user_free_func = free_func;
    ns->user_distance_func = distance_func;
    ns->user_behavior_func = behavior_func;
    ns->user_fitness_func = fitness_func;
    ns->user_data = user_data;
}

/* Version information */
const char* novelty_get_version_string(void) {
    static char version_str[32];
    snprintf(version_str, sizeof(version_str), "%d.%d.%d",
             NOVELTY_VERSION_MAJOR,
             NOVELTY_VERSION_MINOR,
             NOVELTY_VERSION_PATCH);
    return version_str;
}

void novelty_get_version(int* major, int* minor, int* patch) {
    if (major) *major = NOVELTY_VERSION_MAJOR;
    if (minor) *minor = NOVELTY_VERSION_MINOR;
    if (patch) *patch = NOVELTY_VERSION_PATCH;
}

/* Deprecated functions (for backward compatibility) */
#ifndef NOVELTY_NO_DEPRECATED
float calculate_novelty_score(const behavior_t* behavior, const novelty_archive_t* archive, size_t k) {
    return calculate_novelty(behavior, archive, k, euclidean_distance, NULL);
}
#endif
