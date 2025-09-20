#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/visualization.h"
#include "../include/neat.h"

/* Default colors */
static const neat_color_t COLOR_BLACK = {0, 0, 0, 255};
static const neat_color_t COLOR_WHITE = {255, 255, 255, 255};
static const neat_color_t COLOR_RED = {255, 0, 0, 255};
static const neat_color_t COLOR_GREEN = {0, 255, 0, 255};
static const neat_color_t COLOR_BLUE = {0, 0, 255, 255};
static const neat_color_t COLOR_YELLOW = {255, 255, 0, 255};
static const neat_color_t COLOR_CYAN = {0, 255, 255, 255};
static const neat_color_t COLOR_MAGENTA = {255, 0, 255, 255};
static const neat_color_t COLOR_GRAY = {128, 128, 128, 255};
static const neat_color_t COLOR_LIGHT_GRAY = {200, 200, 200, 255};
static const neat_color_t COLOR_DARK_GRAY = {50, 50, 50, 255};

/* Create a new visualizer */
neat_visualizer_t* neat_visualizer_create(const char* title, int width, int height) {
    /* Initialize SDL */
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return NULL;
    }
    
    /* Initialize SDL_ttf */
    if (TTF_Init() == -1) {
        fprintf(stderr, "SDL_ttf could not initialize! SDL_ttf Error: %s\n", TTF_GetError());
        SDL_Quit();
        return NULL;
    }
    
    /* Create window */
    SDL_Window* window = SDL_CreateWindow(
        title,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        width,
        height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
    );
    
    if (!window) {
        fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
        TTF_Quit();
        SDL_Quit();
        return NULL;
    }
    
    /* Create renderer */
    SDL_Renderer* renderer = SDL_CreateRenderer(
        window,
        -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    
    if (!renderer) {
        fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
        return NULL;
    }
    
    /* Set renderer draw blend mode */
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    
    /* Create visualizer */
    neat_visualizer_t* vis = (neat_visualizer_t*)malloc(sizeof(neat_visualizer_t));
    if (!vis) {
        fprintf(stderr, "Failed to allocate memory for visualizer\n");
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
        return NULL;
    }
    
    vis->window = window;
    vis->renderer = renderer;
    vis->width = width;
    vis->height = height;
    vis->is_running = 1;
    
    return vis;
}

/* Destroy visualizer */
void neat_visualizer_destroy(neat_visualizer_t* vis) {
    if (!vis) return;
    
    SDL_DestroyRenderer(vis->renderer);
    SDL_DestroyWindow(vis->window);
    TTF_Quit();
    SDL_Quit();
    free(vis);
}

/* Check if visualizer is running */
int neat_visualizer_is_running(neat_visualizer_t* vis) {
    return vis ? vis->is_running : 0;
}

/* Handle SDL events */
void neat_visualizer_handle_events(neat_visualizer_t* vis) {
    if (!vis) return;
    
    SDL_Event e;
    while (SDL_PollEvent(&e) != 0) {
        if (e.type == SDL_QUIT) {
            vis->is_running = 0;
        } else if (e.type == SDL_WINDOWEVENT) {
            if (e.window.event == SDL_WINDOWEVENT_RESIZED) {
                vis->width = e.window.data1;
                vis->height = e.window.data2;
            }
        } else if (e.type == SDL_KEYDOWN) {
            if (e.key.keysym.sym == SDLK_ESCAPE) {
                vis->is_running = 0;
            }
        }
    }
}

/* Clear the screen */
void neat_visualizer_clear(neat_visualizer_t* vis, neat_color_t color) {
    if (!vis) return;
    
    SDL_SetRenderDrawColor(vis->renderer, color.r, color.g, color.b, color.a);
    SDL_RenderClear(vis->renderer);
}

/* Present the rendered content */
void neat_visualizer_present(neat_visualizer_t* vis) {
    if (vis) {
        SDL_RenderPresent(vis->renderer);
    }
}

/* Draw a rectangle */
void neat_draw_rect(neat_visualizer_t* vis, int x, int y, int w, int h, neat_color_t color) {
    if (!vis) return;
    
    SDL_Rect rect = {x, y, w, h};
    SDL_SetRenderDrawColor(vis->renderer, color.r, color.g, color.b, color.a);
    SDL_RenderFillRect(vis->renderer, &rect);
}

/* Draw a circle */
void neat_draw_circle(neat_visualizer_t* vis, int x, int y, int radius, neat_color_t color) {
    if (!vis) return;
    
    SDL_SetRenderDrawColor(vis->renderer, color.r, color.g, color.b, color.a);
    
    int diameter = radius * 2;
    int x_pos = radius - 1;
    int y_pos = 0;
    int tx = 1;
    int ty = 1;
    int error = tx - diameter;
    
    while (x_pos >= y_pos) {
        /* Draw eight points for each octant */
        SDL_RenderDrawPoint(vis->renderer, x + x_pos, y - y_pos);
        SDL_RenderDrawPoint(vis->renderer, x + x_pos, y + y_pos);
        SDL_RenderDrawPoint(vis->renderer, x - x_pos, y - y_pos);
        SDL_RenderDrawPoint(vis->renderer, x - x_pos, y + y_pos);
        SDL_RenderDrawPoint(vis->renderer, x + y_pos, y - x_pos);
        SDL_RenderDrawPoint(vis->renderer, x + y_pos, y + x_pos);
        SDL_RenderDrawPoint(vis->renderer, x - y_pos, y - x_pos);
        SDL_RenderDrawPoint(vis->renderer, x - y_pos, y + x_pos);
        
        if (error <= 0) {
            y_pos++;
            error += ty;
            ty += 2;
        }
        
        if (error > 0) {
            x_pos--;
            tx += 2;
            error += (tx - diameter);
        }
    }
    
    /* Fill the circle */
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx*dx + dy*dy <= radius*radius) {
                SDL_RenderDrawPoint(vis->renderer, x + dx, y + dy);
            }
        }
    }
}

/* Draw a line */
void neat_draw_line(neat_visualizer_t* vis, int x1, int y1, int x2, int y2, neat_color_t color, int width) {
    if (!vis) return;
    
    SDL_SetRenderDrawColor(vis->renderer, color.r, color.g, color.b, color.a);
    
    if (width <= 1) {
        SDL_RenderDrawLine(vis->renderer, x1, y1, x2, y2);
    } else {
        /* Draw a thick line using multiple parallel lines */
        float dx = x2 - x1;
        float dy = y2 - y1;
        float length = sqrtf(dx*dx + dy*dy);
        float nx = -dy / length;
        float ny = dx / length;
        
        for (int i = -width/2; i <= width/2; i++) {
            int offset_x = (int)(nx * i);
            int offset_y = (int)(ny * i);
            SDL_RenderDrawLine(vis->renderer, 
                              x1 + offset_x, y1 + offset_y, 
                              x2 + offset_x, y2 + offset_y);
        }
    }
}

/* Draw text */
void neat_draw_text(neat_visualizer_t* vis, const char* text, int x, int y, neat_color_t color, int size) {
    if (!vis || !text) return;
    
    static TTF_Font* font = NULL;
    if (!font) {
        /* Try to load a default font */
        font = TTF_OpenFont("Arial.ttf", size);
        if (!font) {
            /* Try a common Linux font */
            font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size);
        }
        if (!font) {
            /* Try a common macOS font */
            font = TTF_OpenFont("/System/Library/Fonts/SFNS.ttf", size);
        }
        if (!font) {
            /* Last resort: use the first available font */
            font = TTF_OpenFont("*", size);
        }
        if (!font) {
            fprintf(stderr, "Failed to load font: %s\n", TTF_GetError());
            return;
        }
    }
    
    SDL_Color sdl_color = {color.r, color.g, color.b, color.a};
    SDL_Surface* surface = TTF_RenderText_Blended(font, text, sdl_color);
    if (!surface) {
        fprintf(stderr, "Failed to render text: %s\n", TTF_GetError());
        return;
    }
    
    SDL_Texture* texture = SDL_CreateTextureFromSurface(vis->renderer, surface);
    if (!texture) {
        fprintf(stderr, "Failed to create texture: %s\n", SDL_GetError());
        SDL_FreeSurface(surface);
        return;
    }
    
    SDL_Rect dst_rect = {x, y, surface->w, surface->h};
    SDL_RenderCopy(vis->renderer, texture, NULL, &dst_rect);
    
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
}

/* Visualize a genome */
void neat_visualize_genome(neat_visualizer_t* vis, neat_genome_t* genome, 
                          int x, int y, int width, int height) {
    if (!vis || !genome) return;
    
    /* Draw background */
    neat_draw_rect(vis, x, y, width, height, COLOR_WHITE);
    
    /* Draw title */
    char title[256];
    snprintf(title, sizeof(title), "Genome ID: %d (Fitness: %.2f)", 
             genome->id, genome->fitness);
    neat_draw_text(vis, title, x + 10, y + 10, COLOR_BLACK, 16);
    
    /* Draw nodes */
    for (size_t i = 0; i < genome->node_count; i++) {
        neat_node_t* node = genome->nodes[i];
        int node_x, node_y;
        int node_radius = 15;
        
        /* Calculate node position based on type */
        switch (node->placement) {
            case NEAT_PLACEMENT_INPUT:
                node_x = x + 50;
                node_y = y + 50 + (int)(i * (height - 100) / genome->node_count);
                break;
            case NEAT_PLACEMENT_HIDDEN:
                node_x = x + width / 2;
                node_y = y + 50 + (int)(i * (height - 100) / genome->node_count);
                break;
            case NEAT_PLACEMENT_OUTPUT:
                node_x = x + width - 50;
                node_y = y + 50 + (int)(i * (height - 100) / genome->node_count);
                break;
            default:
                node_x = x + 50;
                node_y = y + 50 + (int)(i * (height - 100) / genome->node_count);
        }
        
        /* Draw node */
        neat_color_t node_color;
        switch (node->type) {
            case NEAT_NODE_INPUT:
                node_color = COLOR_BLUE;
                break;
            case NEAT_NODE_HIDDEN:
                node_color = COLOR_GREEN;
                break;
            case NEAT_NODE_OUTPUT:
                node_color = COLOR_RED;
                break;
            case NEAT_NODE_BIAS:
                node_color = COLOR_YELLOW;
                break;
            default:
                node_color = COLOR_GRAY;
        }
        
        neat_draw_circle(vis, node_x, node_y, node_radius, node_color);
        
        /* Draw node ID */
        char node_id[16];
        snprintf(node_id, sizeof(node_id), "%d", node->id);
        neat_draw_text(vis, node_id, node_x - 5, node_y - 8, COLOR_BLACK, 12);
    }
    
    /* Draw connections */
    for (size_t i = 0; i < genome->connection_count; i++) {
        neat_connection_t* conn = genome->connections[i];
        if (!conn->enabled) continue;
        
        /* Find source and target nodes */
        neat_node_t* from_node = NULL;
        neat_node_t* to_node = NULL;
        
        for (size_t j = 0; j < genome->node_count; j++) {
            if (genome->nodes[j]->id == conn->in_node) {
                from_node = genome->nodes[j];
            }
            if (genome->nodes[j]->id == conn->out_node) {
                to_node = genome->nodes[j];
            }
            if (from_node && to_node) break;
        }
        
        if (!from_node || !to_node) continue;
        
        /* Calculate node positions */
        int from_x = 0, from_y = 0, to_x = 0, to_y = 0;
        
        /* This is a simplified version - in a real implementation, 
           you'd want to store node positions in the node structure */
        for (size_t j = 0; j < genome->node_count; j++) {
            if (genome->nodes[j] == from_node) {
                from_x = x + 50 + (int)(j * 100) % (width - 100);
                from_y = y + 50 + (int)(j * 50) % (height - 100);
            }
            if (genome->nodes[j] == to_node) {
                to_x = x + 50 + (int)(j * 100) % (width - 100);
                to_y = y + 50 + (int)(j * 50) % (height - 100);
            }
        }
        
        /* Draw connection */
        neat_color_t conn_color = conn->weight > 0 ? COLOR_GREEN : COLOR_RED;
        conn_color.a = (Uint8)(fabsf(conn->weight) * 255.0f);
        
        neat_draw_line(vis, from_x, from_y, to_x, to_y, conn_color, 2);
        
        /* Draw weight */
        char weight_str[16];
        snprintf(weight_str, sizeof(weight_str), "%.2f", conn->weight);
        neat_draw_text(vis, weight_str, 
                      (from_x + to_x) / 2, 
                      (from_y + to_y) / 2, 
                      COLOR_BLACK, 10);
    }
}

/* Visualize a species */
void neat_visualize_species(neat_visualizer_t* vis, neat_species_t* species, 
                           int x, int y, int width, int height) {
    if (!vis || !species) return;
    
    /* Draw background */
    neat_draw_rect(vis, x, y, width, height, COLOR_LIGHT_GRAY);
    
    /* Draw border */
    neat_color_t border_color = {100, 100, 200, 255};
    neat_draw_rect(vis, x, y, width, 2, border_color);
    neat_draw_rect(vis, x, y + height - 2, width, 2, border_color);
    neat_draw_rect(vis, x, y, 2, height, border_color);
    neat_draw_rect(vis, x + width - 2, y, 2, height, border_color);
    
    /* Draw title */
    char title[256];
    snprintf(title, sizeof(title), "Species %d (Size: %zu, Staleness: %d, Best: %.2f)", 
             species->id, species->member_count, species->staleness, species->max_fitness_ever);
    neat_draw_text(vis, title, x + 10, y + 10, COLOR_BLACK, 14);
    
    /* Draw member genomes */
    int num_cols = 3;
    int num_rows = (species->member_count + num_cols - 1) / num_cols;
    int genome_width = (width - 40) / num_cols;
    int genome_height = (height - 50) / num_rows;
    
    for (size_t i = 0; i < species->member_count; i++) {
        int col = i % num_cols;
        int row = i / num_cols;
        int gx = x + 10 + col * (genome_width + 10);
        int gy = y + 40 + row * (genome_height + 10);
        
        neat_draw_rect(vis, gx, gy, genome_width, genome_height, COLOR_WHITE);
        neat_draw_rect(vis, gx, gy, genome_width, genome_height, COLOR_GRAY);
        
        char info[64];
        snprintf(info, sizeof(info), "Genome %d\nFitness: %.2f", 
                 species->members[i]->id, species->members[i]->fitness);
        neat_draw_text(vis, info, gx + 10, gy + 10, COLOR_BLACK, 10);
    }
}

/* Visualize a population */
void neat_visualize_population(neat_visualizer_t* vis, neat_population_t* pop) {
    if (!vis || !pop) return;
    
    /* Draw background */
    neat_visualizer_clear(vis, COLOR_WHITE);
    
    /* Draw title */
    char title[256];
    snprintf(title, sizeof(title), "NEAT Population (Generation: %zu, Species: %zu)", 
             pop->generation, pop->species_count);
    neat_draw_text(vis, title, 10, 10, COLOR_BLACK, 20);
    
    /* Draw species */
    int species_per_row = 2;
    int species_width = vis->width / species_per_row - 20;
    int species_height = vis->height / 2 - 20;
    
    for (size_t i = 0; i < pop->species_count; i++) {
        int row = i / species_per_row;
        int col = i % species_per_row;
        int x = 10 + col * (species_width + 10);
        int y = 50 + row * (species_height + 10);
        
        neat_visualize_species(vis, pop->species[i], x, y, species_width, species_height);
    }
    
    /* Draw stats */
    char stats[512];
    snprintf(stats, sizeof(stats), 
             "Population Size: %zu  |  "
             "Inputs: %d  |  "
             "Outputs: %d  |  "
             "Best Fitness: %.2f",
             pop->genome_count,
             pop->input_size,
             pop->output_size,
             pop->best_fitness);
    
    neat_draw_text(vis, stats, 10, vis->height - 30, COLOR_BLACK, 14);
    
    /* Draw instructions */
    neat_draw_text(vis, "Press ESC to exit", vis->width - 150, vis->height - 30, COLOR_GRAY, 12);
}

/* Create a color from RGBA values */
neat_color_t neat_rgba(Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
    neat_color_t color = {r, g, b, a};
    return color;
}

/* Create a color from HSLA values */
neat_color_t neat_hsla(float h, float s, float l, float a) {
    /* Convert HSL to RGB */
    float r, g, b;
    
    if (s == 0) {
        r = g = b = l;
    } else {
        float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
        float p = 2.0f * l - q;
        
        float h360 = h / 60.0f;
        if (h360 < 0) h360 += 6.0f;
        
        float t = h360 - floorf(h360);
        float tr = 1.0f - t;
        
        if (h360 < 1.0f) {
            r = q;
            g = p + (q - p) * t;
            b = p;
        } else if (h360 < 2.0f) {
            r = p + (q - p) * tr;
            g = q;
            b = p;
        } else if (h360 < 3.0f) {
            r = p;
            g = q;
            b = p + (q - p) * t;
        } else if (h360 < 4.0f) {
            r = p;
            g = p + (q - p) * tr;
            b = q;
        } else if (h360 < 5.0f) {
            r = p + (q - p) * t;
            g = p;
            b = q;
        } else {
            r = q;
            g = p;
            b = p + (q - p) * tr;
        }
    }
    
    neat_color_t color = {
        (Uint8)(r * 255.0f),
        (Uint8)(g * 255.0f),
        (Uint8)(b * 255.0f),
        (Uint8)(a * 255.0f)
    };
    
    return color;
}

/* Linear interpolation between two colors */
neat_color_t neat_color_lerp(neat_color_t a, neat_color_t b, float t) {
    if (t <= 0.0f) return a;
    if (t >= 1.0f) return b;
    
    neat_color_t result;
    result.r = (Uint8)(a.r + (b.r - a.r) * t);
    result.g = (Uint8)(a.g + (b.g - a.g) * t);
    result.b = (Uint8)(a.b + (b.b - a.b) * t);
    result.a = (Uint8)(a.a + (b.a - a.a) * t);
    
    return result;
}

/* Save a screenshot */
int neat_save_screenshot(neat_visualizer_t* vis, const char* filename) {
    if (!vis || !filename) return 0;
    
    SDL_Surface* surface = SDL_CreateRGBSurface(
        0, vis->width, vis->height, 32,
        0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000
    );
    
    if (!surface) {
        fprintf(stderr, "Failed to create surface: %s\n", SDL_GetError());
        return 0;
    }
    
    if (SDL_RenderReadPixels(
        vis->renderer, 
        NULL, 
        SDL_PIXELFORMAT_ARGB8888,
        surface->pixels, 
        surface->pitch
    ) != 0) {
        fprintf(stderr, "Failed to read pixels: %s\n", SDL_GetError());
        SDL_FreeSurface(surface);
        return 0;
    }
    
    if (SDL_SaveBMP(surface, filename) != 0) {
        fprintf(stderr, "Failed to save screenshot: %s\n", SDL_GetError());
        SDL_FreeSurface(surface);
        return 0;
    }
    
    SDL_FreeSurface(surface);
    return 1;
}

/* Draw a graph */
void neat_draw_graph(neat_visualizer_t* vis, float* values, int count, 
                    int x, int y, int w, int h,
                    float min_val, float max_val, 
                    neat_color_t color, const char* title) {
    if (!vis || !values || count < 2) return;
    
    /* Draw background */
    neat_draw_rect(vis, x, y, w, h, COLOR_WHITE);
    neat_draw_rect(vis, x, y, w, h, COLOR_GRAY);
    
    /* Draw title */
    if (title) {
        neat_draw_text(vis, title, x + 10, y + 10, COLOR_BLACK, 14);
    }
    
    /* Calculate scale and offset */
    float range = max_val - min_val;
    if (range < 1e-6f) range = 1.0f;
    
    float scale_x = (float)w / (count - 1);
    float scale_y = (float)h / range;
    
    /* Draw grid */
    neat_color_t grid_color = {200, 200, 200, 255};
    
    /* Horizontal grid lines */
    for (float v = min_val; v <= max_val; v += range / 5.0f) {
        int y_pos = y + h - (int)((v - min_val) * scale_y);
        neat_draw_line(vis, x, y_pos, x + w, y_pos, grid_color, 1);
        
        /* Draw value */
        char val_str[16];
        snprintf(val_str, sizeof(val_str), "%.2f", v);
        neat_draw_text(vis, val_str, x - 40, y_pos - 8, COLOR_BLACK, 10);
    }
    
    /* Vertical grid lines */
    for (int i = 0; i <= 10; i++) {
        int x_pos = x + (i * w) / 10;
        neat_draw_line(vis, x_pos, y, x_pos, y + h, grid_color, 1);
    }
    
    /* Draw graph line */
    for (int i = 0; i < count - 1; i++) {
        int x1 = x + (int)(i * scale_x);
        int y1 = y + h - (int)((values[i] - min_val) * scale_y);
        int x2 = x + (int)((i + 1) * scale_x);
        int y2 = y + h - (int)((values[i + 1] - min_val) * scale_y);
        
        neat_draw_line(vis, x1, y1, x2, y2, color, 2);
    }
    
    /* Draw axes */
    neat_draw_line(vis, x, y + h, x + w, y + h, COLOR_BLACK, 2);  /* X-axis */
    neat_draw_line(vis, x, y, x, y + h, COLOR_BLACK, 2);          /* Y-axis */
}

/* Animation functions */
neat_animation_t* neat_animation_create(int max_frames, int width, int height) {
    neat_animation_t* anim = (neat_animation_t*)malloc(sizeof(neat_animation_t));
    if (!anim) return NULL;
    
    anim->frames = (char**)calloc(max_frames, sizeof(char*));
    if (!anim->frames) {
        free(anim);
        return NULL;
    }
    
    anim->frame_count = 0;
    anim->max_frames = max_frames;
    anim->width = width;
    anim->height = height;
    
    return anim;
}

void neat_animation_add_frame(neat_animation_t* anim, neat_visualizer_t* vis) {
    if (!anim || !vis || anim->frame_count >= anim->max_frames) return;
    
    /* Allocate memory for the frame */
    size_t frame_size = anim->width * anim->height * 4;  /* 4 bytes per pixel (RGBA) */
    anim->frames[anim->frame_count] = (char*)malloc(frame_size);
    if (!anim->frames[anim->frame_count]) return;
    
    /* Read pixels from the renderer */
    SDL_RenderReadPixels(
        vis->renderer,
        NULL,
        SDL_PIXELFORMAT_RGBA32,
        anim->frames[anim->frame_count],
        anim->width * 4
    );
    
    anim->frame_count++;
}

int neat_animation_save(neat_animation_t* anim, const char* filename) {
    if (!anim || !filename || anim->frame_count == 0) return 0;
    
    /* In a real implementation, you would save the animation as a GIF or video */
    /* For now, we'll just save the first frame as a PNG */
    if (anim->frame_count > 0) {
        char frame_filename[256];
        snprintf(frame_filename, sizeof(frame_filename), "%s_frame_0000.png", filename);
        
        SDL_Surface* surface = SDL_CreateRGBSurfaceFrom(
            anim->frames[0],
            anim->width,
            anim->height,
            32,
            anim->width * 4,
            0x000000FF,
            0x0000FF00,
            0x00FF0000,
            0xFF000000
        );
        
        if (surface) {
            IMG_SavePNG(surface, frame_filename);
            SDL_FreeSurface(surface);
            return 1;
        }
    }
    
    return 0;
}

void neat_animation_destroy(neat_animation_t* anim) {
    if (!anim) return;
    
    for (int i = 0; i < anim->frame_count; i++) {
        free(anim->frames[i]);
    }
    
    free(anim->frames);
    free(anim);
}

/* Plot functions */
neat_plot_t* neat_plot_create(int capacity, neat_color_t color, const char* title) {
    neat_plot_t* plot = (neat_plot_t*)malloc(sizeof(neat_plot_t));
    if (!plot) return NULL;
    
    plot->values = (float*)malloc(capacity * sizeof(float));
    if (!plot->values) {
        free(plot);
        return NULL;
    }
    
    plot->count = 0;
    plot->capacity = capacity;
    plot->min_val = 0.0f;
    plot->max_val = 1.0f;
    plot->color = color;
    plot->title = title ? strdup(title) : NULL;
    
    return plot;
}

void neat_plot_add_value(neat_plot_t* plot, float value) {
    if (!plot || !plot->values) return;
    
    if (plot->count < plot->capacity) {
        plot->values[plot->count++] = value;
    } else {
        /* Shift values to the left */
        memmove(plot->values, plot->values + 1, (plot->capacity - 1) * sizeof(float));
        plot->values[plot->capacity - 1] = value;
    }
    
    /* Update min/max */
    if (plot->count == 1) {
        plot->min_val = plot->max_val = value;
    } else {
        if (value < plot->min_val) plot->min_val = value;
        if (value > plot->max_val) plot->max_val = value;
    }
}

void neat_plot_destroy(neat_plot_t* plot) {
    if (!plot) return;
    
    free(plot->values);
    if (plot->title) free((void*)plot->title);
    free(plot);
}
