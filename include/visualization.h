#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <SDL2/SDL.h>
#include "../include/neat.h"

/* Visualization context */
typedef struct {
    SDL_Window* window;
    SDL_Renderer* renderer;
    int width;
    int height;
    int is_running;
} neat_visualizer_t;

/* Color structure */
typedef struct {
    Uint8 r, g, b, a;
} neat_color_t;

/* Visualization functions */
neat_visualizer_t* neat_visualizer_create(const char* title, int width, int height);
void neat_visualizer_destroy(neat_visualizer_t* vis);
int neat_visualizer_is_running(neat_visualizer_t* vis);
void neat_visualizer_handle_events(neat_visualizer_t* vis);
void neat_visualizer_clear(neat_visualizer_t* vis, neat_color_t color);
void neat_visualizer_present(neat_visualizer_t* vis);

/* Drawing functions */
void neat_draw_rect(neat_visualizer_t* vis, int x, int y, int w, int h, neat_color_t color);
void neat_draw_circle(neat_visualizer_t* vis, int x, int y, int radius, neat_color_t color);
void neat_draw_line(neat_visualizer_t* vis, int x1, int y1, int x2, int y2, neat_color_t color, int width);
void neat_draw_text(neat_visualizer_t* vis, const char* text, int x, int y, neat_color_t color, int size);

/* Visualization of NEAT components */
void neat_visualize_genome(neat_visualizer_t* vis, neat_genome_t* genome, int x, int y, int width, int height);
void neat_visualize_species(neat_visualizer_t* vis, neat_species_t* species, int x, int y, int width, int height);
void neat_visualize_population(neat_visualizer_t* vis, neat_population_t* pop);

/* Animation and interaction */
void neat_animate_genome_evolution(neat_visualizer_t* vis, neat_population_t* pop, int generations, int delay_ms);

/* Color utilities */
neat_color_t neat_rgba(Uint8 r, Uint8 g, Uint8 b, Uint8 a);
neat_color_t neat_hsla(float h, float s, float l, float a);
neat_color_t neat_color_lerp(neat_color_t a, neat_color_t b, float t);

/* Graph visualization */
void neat_draw_graph(neat_visualizer_t* vis, float* values, int count, int x, int y, int w, int h, 
                     float min_val, float max_val, neat_color_t color, const char* title);

/* Heatmap visualization */
void neat_draw_heatmap(neat_visualizer_t* vis, float* data, int rows, int cols, 
                       int x, int y, int w, int h, const char* title);

/* 3D visualization */
void neat_draw_3d_surface(neat_visualizer_t* vis, float (*func)(float, float), 
                         float x_min, float x_max, float y_min, float y_max,
                         int x, int y, int w, int h, const char* title);

/* Network visualization */
void neat_draw_network(neat_visualizer_t* vis, neat_genome_t* genome, 
                      int x, int y, int w, int h, int node_radius, int line_width);

/* Statistics visualization */
void neat_draw_statistics(neat_visualizer_t* vis, neat_population_t* pop, 
                         int x, int y, int w, int h);

/* Interactive exploration */
void neat_interactive_explorer(neat_visualizer_t* vis, neat_population_t* pop);

/* Save visualization to file */
int neat_save_screenshot(neat_visualizer_t* vis, const char* filename);

/* Animation recording */
typedef struct {
    char** frames;
    int frame_count;
    int max_frames;
    int width;
    int height;
} neat_animation_t;

neat_animation_t* neat_animation_create(int max_frames, int width, int height);
void neat_animation_add_frame(neat_animation_t* anim, neat_visualizer_t* vis);
int neat_animation_save(neat_animation_t* anim, const char* filename);
void neat_animation_destroy(neat_animation_t* anim);

/* Real-time plotting */
typedef struct {
    float* values;
    int count;
    int capacity;
    float min_val;
    float max_val;
    neat_color_t color;
    const char* title;
} neat_plot_t;

neat_plot_t* neat_plot_create(int capacity, neat_color_t color, const char* title);
void neat_plot_add_value(neat_plot_t* plot, float value);
void neat_draw_plot(neat_visualizer_t* vis, neat_plot_t* plot, int x, int y, int w, int h);
void neat_plot_destroy(neat_plot_t* plot);

/* Genome comparison visualization */
void neat_visualize_comparison(neat_visualizer_t* vis, neat_genome_t* a, neat_genome_t* b, 
                              int x, int y, int w, int h, const char* title_a, const char* title_b);

/* Fitness landscape visualization */
void neat_visualize_fitness_landscape(neat_visualizer_t* vis, neat_population_t* pop, 
                                     int x, int y, int w, int h);

/* Innovation tracking visualization */
void neat_visualize_innovations(neat_visualizer_t* vis, neat_innovation_table_t* table, 
                               int x, int y, int w, int h);

/* Speciation visualization */
void neat_visualize_speciation(neat_visualizer_t* vis, neat_population_t* pop, 
                              int x, int y, int w, int h);

/* Interactive genome editor */
void neat_interactive_genome_editor(neat_visualizer_t* vis, neat_genome_t* genome);

/* Neural network activation visualization */
void neat_visualize_activations(neat_visualizer_t* vis, neat_genome_t* genome, 
                               const float* input, int x, int y, int w, int h);

/* Evolutionary dynamics visualization */
void neat_visualize_evolution(neat_visualizer_t* vis, neat_population_t* pop, 
                             int x, int y, int w, int h);

/* Multi-panel dashboard */
typedef struct {
    neat_visualizer_t* vis;
    int panel_count;
    SDL_Rect* panels;
    char** titles;
} neat_dashboard_t;

neat_dashboard_t* neat_dashboard_create(neat_visualizer_t* vis, int rows, int cols);
void neat_dashboard_destroy(neat_dashboard_t* dashboard);
void neat_dashboard_render(neat_dashboard_t* dashboard);
SDL_Rect* neat_dashboard_get_panel(neat_dashboard_t* dashboard, int row, int col);

/* Genome similarity matrix */
void neat_visualize_similarity_matrix(neat_visualizer_t* vis, neat_population_t* pop, 
                                     int x, int y, int w, int h);

/* Time series visualization */
void neat_visualize_time_series(neat_visualizer_t* vis, float** series, int num_series, 
                               int length, int x, int y, int w, int h, 
                               neat_color_t* colors, const char** labels);

/* Parallel coordinates plot */
void neat_visualize_parallel_coordinates(neat_visualizer_t* vis, float** data, 
                                        int num_points, int num_dimensions,
                                        int x, int y, int w, int h,
                                        const char** dimension_labels);

/* Interactive selection tool */
typedef struct {
    int x, y, w, h;
    int is_selecting;
    int start_x, start_y;
    int end_x, end_y;
} neat_selection_tool_t;

neat_selection_tool_t* neat_selection_tool_create(int x, int y, int w, int h);
void neat_selection_tool_update(neat_selection_tool_t* tool, SDL_Event* event);
void neat_selection_tool_render(neat_visualizer_t* vis, neat_selection_tool_t* tool);
void neat_selection_tool_get_selection(neat_selection_tool_t* tool, SDL_Rect* rect);
void neat_selection_tool_destroy(neat_selection_tool_t* tool);

/* Genome distance visualization */
void neat_visualize_genome_distances(neat_visualizer_t* vis, neat_genome_t** genomes, 
                                    int count, int x, int y, int w, int h);

/* Population statistics over time */
typedef struct {
    float* fitness;
    float* complexity;
    float* species_sizes;
    int* num_species;
    int capacity;
    int count;
} neat_population_history_t;

neat_population_history_t* neat_population_history_create(int capacity);
void neat_population_history_add(neat_population_history_t* history, neat_population_t* pop);
void neat_visualize_population_history(neat_visualizer_t* vis, neat_population_history_t* history, 
                                      int x, int y, int w, int h);
void neat_population_history_destroy(neat_population_history_t* history);

/* Interactive genome browser */
void neat_interactive_genome_browser(neat_visualizer_t* vis, neat_population_t* pop);

/* Real-time evolution visualization */
void neat_realtime_evolution_visualization(neat_visualizer_t* vis, neat_population_t* pop, 
                                          int num_generations, int update_interval);

/* Multi-objective optimization visualization */
void neat_visualize_pareto_front(neat_visualizer_t* vis, neat_population_t* pop, 
                                int obj1, int obj2, int x, int y, int w, int h);

/* Modular network visualization */
void neat_visualize_modularity(neat_visualizer_t* vis, neat_genome_t* genome, 
                              int x, int y, int w, int h);

/* Neural network feature visualization */
void neat_visualize_features(neat_visualizer_t* vis, neat_genome_t* genome, 
                            const float* input, int input_size, 
                            int x, int y, int w, int h);

/* Interactive experiment runner */
void neat_interactive_experiment_runner(neat_visualizer_t* vis, neat_population_t* pop, 
                                       void (*evaluate_func)(neat_population_t*), 
                                       int num_generations);

#endif /* VISUALIZATION_H */
