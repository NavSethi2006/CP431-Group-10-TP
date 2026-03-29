#define GL_SILENCE_DEPRECATION
// #include "glut.h"
// mac
#include <GLUT/glut.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <sys/stat.h>
#include <dirent.h>
#include <limits.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define MAX_OPEN_TILES 64
#define SAFE_MAX_TILE_DIM 4096
#define MAX_TILE_LOADS_PER_FRAME 8

static GLint gpu_max_texture_size = 0;

typedef struct {
    char path[PATH_MAX];
    long payload_offset;
    int start_x;
    int start_y;
    int end_x;
    int end_y;
    int chunk_id;
    int tile_width;
    int tile_height;
    GLuint texture_id;
    int texture_loaded;
    unsigned long last_used_frame;
    int subtiles_per_side;
} TileRecord;

static TileRecord *tiles = NULL;
static int tile_count = 0;
static int tile_capacity = 0;
static int width = 0;
static int height = 0;
static char *filename = NULL;
static float zoom_factor = 1.0f;
static const float max_zoom_factor = 50.0f;
static float pan_x = 0.0f;
static float pan_y = 0.0f;
static int window_width = 1000;
static int window_height = 1000;
static unsigned long frame_counter = 0;
static int loaded_tile_count = 0;
static int needs_more_frames = 0;
static int initial_view_fitted = 0;

void get_color(unsigned char index, unsigned char *r, unsigned char *g, unsigned char *b);
int read_tiles_from_manifest(const char *input_path);
int load_input_data(const char *input_path);
int read_data_file(const char *fname);
void save_jpeg(const char *outfile);
void display(void);
void reshape(int w, int h);
void keyboard(unsigned char key, int x, int y);
void idle(void);

static int path_is_directory(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return 0;
    }
    return S_ISDIR(st.st_mode);
}

static void free_tiles(void) {
    if (tiles == NULL) {
        return;
    }

    for (int i = 0; i < tile_count; i++) {
        if (tiles[i].texture_loaded && tiles[i].texture_id != 0) {
            glDeleteTextures(1, &tiles[i].texture_id);
        }
    }

    free(tiles);
    tiles = NULL;
    tile_count = 0;
    tile_capacity = 0;
    loaded_tile_count = 0;
}

static int ensure_tile_capacity(void) {
    if (tile_count < tile_capacity) {
        return 0;
    }

    int new_capacity = (tile_capacity == 0) ? 128 : tile_capacity * 2;
    TileRecord *new_tiles = realloc(tiles, (size_t)new_capacity * sizeof(TileRecord));
    if (!new_tiles) {
        fprintf(stderr, "Failed to grow tile index\n");
        return 1;
    }

    tiles = new_tiles;
    tile_capacity = new_capacity;
    return 0;
}

static void clamp_pan(void) {
    float visible_w = 1.0f / zoom_factor;
    float visible_h = 1.0f / zoom_factor;
    float max_pan_x = (1.0f - visible_w) * 0.5f;
    float max_pan_y = (1.0f - visible_h) * 0.5f;

    if (pan_x > max_pan_x) pan_x = max_pan_x;
    if (pan_x < -max_pan_x) pan_x = -max_pan_x;
    if (pan_y > max_pan_y) pan_y = max_pan_y;
    if (pan_y < -max_pan_y) pan_y = -max_pan_y;
}

static void get_view_bounds_pixels(double *left_px, double *right_px, double *top_px, double *bottom_px) {
    float visible_w = 1.0f / zoom_factor;
    float visible_h = 1.0f / zoom_factor;
    float u0 = 0.5f - visible_w * 0.5f + pan_x;
    float v0 = 0.5f - visible_h * 0.5f + pan_y;
    float u1 = u0 + visible_w;
    float v1 = v0 + visible_h;

    if (u0 < 0.0f) { u1 -= u0; u0 = 0.0f; }
    if (v0 < 0.0f) { v1 -= v0; v0 = 0.0f; }
    if (u1 > 1.0f) { u0 -= (u1 - 1.0f); u1 = 1.0f; }
    if (v1 > 1.0f) { v0 -= (v1 - 1.0f); v1 = 1.0f; }

    *left_px = (double)u0 * (double)width;
    *right_px = (double)u1 * (double)width;
    *top_px = (double)v0 * (double)height;
    *bottom_px = (double)v1 * (double)height;
}

static int tile_intersects_view(const TileRecord *tile, double left_px, double right_px, double top_px, double bottom_px) {
    double tile_left = (double)tile->start_x;
    double tile_right = (double)(tile->end_x + 1);
    double tile_top = (double)tile->start_y;
    double tile_bottom = (double)(tile->end_y + 1);

    if (tile_right <= left_px) return 0;
    if (tile_left >= right_px) return 0;
    if (tile_bottom <= top_px) return 0;
    if (tile_top >= bottom_px) return 0;
    return 1;
}

static void evict_one_tile_if_needed(void) {
    if (loaded_tile_count < MAX_OPEN_TILES) {
        return;
    }

    int victim = -1;
    unsigned long oldest_frame = 0;

    for (int i = 0; i < tile_count; i++) {
        if (!tiles[i].texture_loaded) {
            continue;
        }
        if (victim == -1 || tiles[i].last_used_frame < oldest_frame) {
            victim = i;
            oldest_frame = tiles[i].last_used_frame;
        }
    }

    if (victim >= 0) {
        glDeleteTextures(1, &tiles[victim].texture_id);
        tiles[victim].texture_id = 0;
        tiles[victim].texture_loaded = 0;
        loaded_tile_count--;
    }
}

static int load_tile_texture(TileRecord *tile) {
    if (tile->texture_loaded) {
        tile->last_used_frame = frame_counter;
        return 0;
    }

    evict_one_tile_if_needed();

    FILE *f = fopen(tile->path, "rb");
    if (!f) {
        perror("Cannot open rank file for tile load");
        return 1;
    }

    if (fseek(f, tile->payload_offset, SEEK_SET) != 0) {
        perror("Failed to seek to tile payload");
        fclose(f);
        return 1;
    }

    size_t gray_size = (size_t)tile->tile_width * (size_t)tile->tile_height;
    uint8_t *gray = malloc(gray_size);
    if (!gray) {
        fprintf(stderr, "Failed to allocate tile grayscale buffer\n");
        fclose(f);
        return 1;
    }

    if (fread(gray, sizeof(uint8_t), gray_size, f) != gray_size) {
        fprintf(stderr, "Failed to read tile payload from %s\n", tile->path);
        free(gray);
        fclose(f);
        return 1;
    }

    fclose(f);

    int effective_limit = SAFE_MAX_TILE_DIM;
    if (gpu_max_texture_size > 0 && gpu_max_texture_size < effective_limit) {
        effective_limit = gpu_max_texture_size;
    }
    if (effective_limit < 1) {
        effective_limit = 1;
    }

    int max_dim = tile->tile_width > tile->tile_height ? tile->tile_width : tile->tile_height;
    tile->subtiles_per_side = (max_dim + effective_limit - 1) / effective_limit;
    if (tile->subtiles_per_side < 1) {
        tile->subtiles_per_side = 1;
    }

    int upload_width = tile->tile_width;
    int upload_height = tile->tile_height;
    if (upload_width > effective_limit) upload_width = effective_limit;
    if (upload_height > effective_limit) upload_height = effective_limit;

    unsigned char *rgb = malloc((size_t)upload_width * (size_t)upload_height * 3);
    if (!rgb) {
        fprintf(stderr, "Failed to allocate tile RGB buffer\n");
        free(gray);
        return 1;
    }

    for (int row = 0; row < upload_height; row++) {
        for (int col = 0; col < upload_width; col++) {
            int src_index = row * tile->tile_width + col;
            int dst_row = upload_height - 1 - row;
            int dst_index = dst_row * upload_width + col;
            unsigned char r, g, b;
            get_color(gray[src_index], &r, &g, &b);
            rgb[dst_index * 3 + 0] = r;
            rgb[dst_index * 3 + 1] = g;
            rgb[dst_index * 3 + 2] = b;
        }
    }

    free(gray);

    glGenTextures(1, &tile->texture_id);
    glBindTexture(GL_TEXTURE_2D, tile->texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        upload_width,
        upload_height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        rgb
    );

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "glTexImage2D failed for tile %d (%dx%d) with OpenGL error %u\n", tile->chunk_id, upload_width, upload_height, (unsigned int)err);
        glDeleteTextures(1, &tile->texture_id);
        tile->texture_id = 0;
        free(rgb);
        return 1;
    }

    free(rgb);

    tile->texture_loaded = 1;
    tile->last_used_frame = frame_counter;
    loaded_tile_count++;
    return 0;
}

// Color map - map escape index (0..255) to RGB
void get_color(unsigned char index, unsigned char *r, unsigned char *g, unsigned char *b) {
    if(index == 0) {
        *r = 0; *g = 0; *b = 0;
        return;
    }

    float t = (float) index / 255.0f;
    *r = (unsigned char)(sin(t * 2.0 * M_PI) * 127 + 128);
    *g = (unsigned char)(sin((t + 1.0f / 3.0f) * 2.0 * M_PI) * 127 + 128);
    *b = (unsigned char)(sin((t + 2.0f / 3.0f) * 2.0 * M_PI) * 127 + 128);
}

int read_tiles_from_manifest(const char *input_path) {
    char manifest_path[PATH_MAX];
    if (path_is_directory(input_path)) {
        snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", input_path);
    } else {
        snprintf(manifest_path, sizeof(manifest_path), "%s", input_path);
    }

    FILE *manifest = fopen(manifest_path, "r");
    if (!manifest) {
        perror("Cannot open manifest file");
        return 1;
    }

    int manifest_width = 0;
    int manifest_height = 0;
    int manifest_chunk_size = 0;
    int manifest_max_iterations = 0;
    char key[128];
    int value = 0;

    while (fscanf(manifest, "%127s %d", key, &value) == 2) {
        if (strcmp(key, "WIDTH") == 0) {
            manifest_width = value;
        } else if (strcmp(key, "HEIGHT") == 0) {
            manifest_height = value;
        } else if (strcmp(key, "CHUNK_SIZE") == 0) {
            manifest_chunk_size = value;
        } else if (strcmp(key, "MAX_ITERATIONS") == 0) {
            manifest_max_iterations = value;
        }
    }
    fclose(manifest);

    if (manifest_width <= 0 || manifest_height <= 0 || manifest_chunk_size <= 0 || manifest_max_iterations <= 0) {
        fprintf(stderr, "Invalid manifest contents in %s\n", manifest_path);
        return 1;
    }

    width = manifest_width;
    height = manifest_height;
    if (!initial_view_fitted) {
        double tiles_x = (double)width / (double)SAFE_MAX_TILE_DIM;
        double tiles_y = (double)height / (double)SAFE_MAX_TILE_DIM;
        double needed_zoom = tiles_x > tiles_y ? tiles_x : tiles_y;
        if (needed_zoom < 1.0) {
            needed_zoom = 1.0;
        }
        if (needed_zoom > max_zoom_factor) {
            needed_zoom = max_zoom_factor;
        }
        zoom_factor = (float)needed_zoom;
        pan_x = 0.0f;
        pan_y = 0.0f;
        initial_view_fitted = 1;
    }

    char tiles_dir[PATH_MAX];
    if (path_is_directory(input_path)) {
        snprintf(tiles_dir, sizeof(tiles_dir), "%s", input_path);
    } else {
        strncpy(tiles_dir, manifest_path, sizeof(tiles_dir) - 1);
        tiles_dir[sizeof(tiles_dir) - 1] = '\0';
        char *last_slash = strrchr(tiles_dir, '/');
        if (last_slash) {
            *last_slash = '\0';
        } else {
            snprintf(tiles_dir, sizeof(tiles_dir), ".");
        }
    }

    DIR *dir = opendir(tiles_dir);
    if (!dir) {
        perror("Cannot open tiles directory");
        return 1;
    }

    int rank_file_count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "rank_", 5) != 0) {
            continue;
        }
        if (strstr(entry->d_name, ".bin") == NULL) {
            continue;
        }

        char rank_path[PATH_MAX];
        snprintf(rank_path, sizeof(rank_path), "%s/%s", tiles_dir, entry->d_name);

        FILE *rank_file = fopen(rank_path, "rb");
        if (!rank_file) {
            perror("Cannot open rank file");
            closedir(dir);
            free_tiles();
            return 1;
        }

        rank_file_count++;

        while (1) {
            int chunk[5];
            long header_offset = ftell(rank_file);
            if (header_offset < 0) {
                perror("ftell failed on rank file");
                fclose(rank_file);
                closedir(dir);
                free_tiles();
                return 1;
            }

            size_t header_items = fread(chunk, sizeof(int), 5, rank_file);
            if (header_items == 0) {
                break;
            }
            if (header_items != 5) {
                fprintf(stderr, "Failed to read chunk header from %s\n", rank_path);
                fclose(rank_file);
                closedir(dir);
                free_tiles();
                return 1;
            }

            int start_x = chunk[0];
            int start_y = chunk[1];
            int end_x = chunk[2];
            int end_y = chunk[3];
            int chunk_id = chunk[4];
            int tile_width = end_x - start_x + 1;
            int tile_height = end_y - start_y + 1;

            if (start_x < 0 || start_y < 0 || end_x >= width || end_y >= height || tile_width <= 0 || tile_height <= 0) {
                fprintf(stderr, "Invalid chunk bounds in %s\n", rank_path);
                fclose(rank_file);
                closedir(dir);
                free_tiles();
                return 1;
            }

            if (ensure_tile_capacity() != 0) {
                fclose(rank_file);
                closedir(dir);
                free_tiles();
                return 1;
            }

            long payload_offset = header_offset + (long)(5 * sizeof(int));
            TileRecord *tile = &tiles[tile_count++];
            memset(tile, 0, sizeof(*tile));
            snprintf(tile->path, sizeof(tile->path), "%s", rank_path);
            tile->payload_offset = payload_offset;
            tile->start_x = start_x;
            tile->start_y = start_y;
            tile->end_x = end_x;
            tile->end_y = end_y;
            tile->chunk_id = chunk_id;
            tile->tile_width = tile_width;
            tile->tile_height = tile_height;
            tile->texture_id = 0;
            tile->texture_loaded = 0;
            tile->last_used_frame = 0;
            tile->subtiles_per_side = 1;

            size_t tile_size = (size_t)tile_width * (size_t)tile_height;
            if (fseek(rank_file, (long)tile_size, SEEK_CUR) != 0) {
                fprintf(stderr, "Failed to skip chunk payload in %s\n", rank_path);
                fclose(rank_file);
                closedir(dir);
                free_tiles();
                return 1;
            }
        }

        fclose(rank_file);
    }

    closedir(dir);

    if (rank_file_count == 0) {
        fprintf(stderr, "No rank_*.bin files found in %s\n", tiles_dir);
        free_tiles();
        return 1;
    }

    if (tile_count == 0) {
        fprintf(stderr, "No chunk records found in rank files under %s\n", tiles_dir);
        free_tiles();
        return 1;
    }

    return 0;
}

int load_input_data(const char *input_path) {
    if (path_is_directory(input_path)) {
        return read_tiles_from_manifest(input_path);
    }

    if (strstr(input_path, "manifest.txt") != NULL) {
        return read_tiles_from_manifest(input_path);
    }

    return read_data_file(input_path);
}

// Legacy single-file loader retained for compatibility.
int read_data_file(const char *fname) {
    FILE *f = fopen(fname, "rb");
    if (!f) {
        perror("Cannot open file");
        return 1;
    }

    int file_width = 0;
    int file_height = 0;

    if (fread(&file_width, sizeof(int), 1, f) != 1 ||
        fread(&file_height, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Failed to read width/height header from %s\n", fname);
        fclose(f);
        return 1;
    }

    if (file_width <= 0 || file_height <= 0) {
        fprintf(stderr, "Invalid image dimensions in %s: %d x %d\n", fname, file_width, file_height);
        fclose(f);
        return 1;
    }

    width = file_width;
    height = file_height;
    if (!initial_view_fitted) {
        double tiles_x = (double)width / (double)SAFE_MAX_TILE_DIM;
        double tiles_y = (double)height / (double)SAFE_MAX_TILE_DIM;
        double needed_zoom = tiles_x > tiles_y ? tiles_x : tiles_y;
        if (needed_zoom < 1.0) {
            needed_zoom = 1.0;
        }
        if (needed_zoom > max_zoom_factor) {
            needed_zoom = max_zoom_factor;
        }
        zoom_factor = (float)needed_zoom;
        pan_x = 0.0f;
        pan_y = 0.0f;
        initial_view_fitted = 1;
    }

    if (ensure_tile_capacity() != 0) {
        fclose(f);
        return 1;
    }

    TileRecord *tile = &tiles[tile_count++];
    memset(tile, 0, sizeof(*tile));
    snprintf(tile->path, sizeof(tile->path), "%s", fname);
    tile->payload_offset = (long)(2 * sizeof(int));
    tile->start_x = 0;
    tile->start_y = 0;
    tile->end_x = width - 1;
    tile->end_y = height - 1;
    tile->chunk_id = 0;
    tile->tile_width = width;
    tile->tile_height = height;
    tile->texture_id = 0;
    tile->texture_loaded = 0;
    tile->last_used_frame = 0;
    tile->subtiles_per_side = 1;

    fclose(f);
    return 0;
}

// Saves the current opengl window as a jpeg
void save_jpeg(const char *outfile) {
    unsigned char *pixels = malloc((size_t)window_width * (size_t)window_height * 3);
    if(!pixels) {
        fprintf(stderr, "Out of memory to take a screenshot\n");
        return;
    }
    glReadPixels(0, 0, window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    unsigned char *flipped = malloc((size_t)window_width * (size_t)window_height * 3);
    if(!flipped) {
        fprintf(stderr, "Out of memory to flip screenshot\n");
        free(pixels);
        return;
    }
    for(int i = 0; i < window_height; i++) {
        memcpy(
            flipped + (size_t)i * (size_t)window_width * 3,
            pixels + (size_t)(window_height - 1 - i) * (size_t)window_width * 3,
            (size_t)window_width * 3
        );
    }

    if(stbi_write_jpg(outfile, window_width, window_height, 3, flipped, 90) == 0) {
        fprintf(stderr, "Failed to write jpeg file %s\n", outfile);
    } else {
        printf("Saved %s\n", outfile);
    }

    free(pixels);
    free(flipped);
}

static void draw_tile(const TileRecord *tile, double left_px, double right_px, double top_px, double bottom_px) {
    double tile_left = (double)tile->start_x;
    double tile_top = (double)tile->start_y;
    double view_w = right_px - left_px;
    double view_h = bottom_px - top_px;

    int parts = tile->subtiles_per_side;
    if (parts < 1) {
        parts = 1;
    }

    for (int part_y = 0; part_y < parts; part_y++) {
        for (int part_x = 0; part_x < parts; part_x++) {
            double local_x0 = ((double)part_x / (double)parts) * (double)tile->tile_width;
            double local_x1 = ((double)(part_x + 1) / (double)parts) * (double)tile->tile_width;
            double local_y0 = ((double)part_y / (double)parts) * (double)tile->tile_height;
            double local_y1 = ((double)(part_y + 1) / (double)parts) * (double)tile->tile_height;

            double world_x0 = tile_left + local_x0;
            double world_x1 = tile_left + local_x1;
            double world_y0 = tile_top + local_y0;
            double world_y1 = tile_top + local_y1;

            double x0 = ((world_x0 - left_px) / view_w) * 2.0 - 1.0;
            double x1 = ((world_x1 - left_px) / view_w) * 2.0 - 1.0;
            double y0 = 1.0 - ((world_y0 - top_px) / view_h) * 2.0;
            double y1 = 1.0 - ((world_y1 - top_px) / view_h) * 2.0;

            double u0 = (double)part_x / (double)parts;
            double u1 = (double)(part_x + 1) / (double)parts;
            double v0 = (double)part_y / (double)parts;
            double v1 = (double)(part_y + 1) / (double)parts;

            glBindTexture(GL_TEXTURE_2D, tile->texture_id);
            glBegin(GL_QUADS);
            glTexCoord2f((GLfloat)u0, (GLfloat)v0); glVertex2f((GLfloat)x0, (GLfloat)y1);
            glTexCoord2f((GLfloat)u1, (GLfloat)v0); glVertex2f((GLfloat)x1, (GLfloat)y1);
            glTexCoord2f((GLfloat)u1, (GLfloat)v1); glVertex2f((GLfloat)x1, (GLfloat)y0);
            glTexCoord2f((GLfloat)u0, (GLfloat)v1); glVertex2f((GLfloat)x0, (GLfloat)y0);
            glEnd();
        }
    }
}

void display(void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);

    frame_counter++;
    needs_more_frames = 0;

    double left_px, right_px, top_px, bottom_px;
    get_view_bounds_pixels(&left_px, &right_px, &top_px, &bottom_px);

    int loads_this_frame = 0;

    for (int i = 0; i < tile_count; i++) {
        if (!tile_intersects_view(&tiles[i], left_px, right_px, top_px, bottom_px)) {
            continue;
        }

        if (!tiles[i].texture_loaded) {
            if (loads_this_frame >= MAX_TILE_LOADS_PER_FRAME) {
                needs_more_frames = 1;
                continue;
            }
            if (load_tile_texture(&tiles[i]) != 0) {
                continue;
            }
            loads_this_frame++;
        } else {
            tiles[i].last_used_frame = frame_counter;
        }

        draw_tile(&tiles[i], left_px, right_px, top_px, bottom_px);
    }

    glDisable(GL_TEXTURE_2D);
    glutSwapBuffers();

    if (needs_more_frames) {
        glutPostRedisplay();
    }
}

void idle(void) {
    if (needs_more_frames) {
        glutPostRedisplay();
    }
}

void reshape(int w, int h) {
    window_width = w;
    window_height = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void keyboard(unsigned char key, int x, int y) {
    (void)x;
    (void)y;

    float pan_step = 0.05f / zoom_factor;

    switch (key) {
        case 'q':
        case 27:
            exit(0);
            break;
        case '+':
        case '=':
            zoom_factor *= 1.25f;
            if (zoom_factor > max_zoom_factor) {
                zoom_factor = max_zoom_factor;
            }
            clamp_pan();
            glutPostRedisplay();
            break;
        case ']':
        case '}':
            zoom_factor *= 2.0f;
            if (zoom_factor > max_zoom_factor) {
                zoom_factor = max_zoom_factor;
            }
            clamp_pan();
            glutPostRedisplay();
            break;
        case '-':
        case '_':
            zoom_factor /= 1.25f;
            if (zoom_factor < 1.0f) {
                zoom_factor = 1.0f;
            }
            clamp_pan();
            glutPostRedisplay();
            break;
        case '[':
        case '{':
            zoom_factor /= 2.0f;
            if (zoom_factor < 1.0f) {
                zoom_factor = 1.0f;
            }
            clamp_pan();
            glutPostRedisplay();
            break;
        case 'w':
        case 'W':
            pan_y -= pan_step;
            clamp_pan();
            glutPostRedisplay();
            break;
        case 's':
        case 'S':
            pan_y += pan_step;
            clamp_pan();
            glutPostRedisplay();
            break;
        case 'a':
        case 'A':
            pan_x -= pan_step;
            clamp_pan();
            glutPostRedisplay();
            break;
        case 'd':
        case 'D':
            pan_x += pan_step;
            clamp_pan();
            glutPostRedisplay();
            break;
        case 'r':
        case 'R':
            zoom_factor = 1.0f;
            pan_x = 0.0f;
            pan_y = 0.0f;
            glutPostRedisplay();
            break;
        case 'j':
        case 'J': {
            char outfile[256];
            snprintf(outfile, sizeof(outfile), "%s_view.jpg", filename ? filename : "julia");
            save_jpeg(outfile);
            break;
        }
        default:
            break;
    }
}

int main(int argc, char **argv) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <datafile.bin | tiles_directory | manifest.txt>\n", argv[0]);
        return 1;
    }
    filename = argv[1];

    if(load_input_data(filename) != 0) {
        return 1;
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Julia Set Viewer");
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &gpu_max_texture_size);
    printf("GL_MAX_TEXTURE_SIZE: %d\n", (int)gpu_max_texture_size);
    printf("Viewer safe tile limit: %d\n", SAFE_MAX_TILE_DIM);
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(idle);
    printf("Loaded image bounds: %d x %d\n", width, height);
    printf("Initial zoom factor: %.2f\n", zoom_factor);
    printf("Indexed tiles: %d\n", tile_count);
    printf("Large tiles will be cropped to fit GPU texture limits if needed\n");
    printf("Tile streaming limit: %d new tiles per frame\n", MAX_TILE_LOADS_PER_FRAME);
    printf("Controls: +/- zoom, [ ] fast zoom, W/A/S/D pan, R reset, J save JPEG, Q or ESC quit\n");
    printf("Input loader: tiled rank_*.bin viewer enabled\n");

    glutMainLoop();

    free_tiles();
    return 0;
}