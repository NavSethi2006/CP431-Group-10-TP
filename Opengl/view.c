

#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define DEFAULT_DATA_DIR "../MPI/file"
#define MAX_TILE_LOADS_PER_FRAME 6
#define MAX_OPEN_TILES 64
#define DEFAULT_WINDOW_WIDTH 1280
#define DEFAULT_WINDOW_HEIGHT 800
#define JPEG_EXPORT_SIZE 4096

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
    unsigned long visible_frame;
} TileRecord;

typedef struct {
    int width;
    int height;
    int chunk_size;
    int max_iterations;
    double threshold;
    char function_name[64];
    double c_re;
    double c_im;
} Manifest;

typedef struct {
    double center_x;
    double center_y;
    double zoom;
} Camera;

static Manifest manifest_data;
static TileRecord *tiles = NULL;
static int tile_count = 0;
static int tile_capacity = 0;
static char data_dir[PATH_MAX] = DEFAULT_DATA_DIR;

static int window_width = DEFAULT_WINDOW_WIDTH;
static int window_height = DEFAULT_WINDOW_HEIGHT;
static int dragging = 0;
static int last_mouse_x = 0;
static int last_mouse_y = 0;
static unsigned long frame_counter = 0;
static GLint gpu_max_texture_size = 0;
static Camera camera_state = {0.0, 0.0, 1.0};

static void fail_and_exit(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static double clamp_double(double value, double minimum, double maximum) {
    if (value < minimum) {
        return minimum;
    }
    if (value > maximum) {
        return maximum;
    }
    return value;
}

static void ensure_tile_capacity(void) {
    if (tile_count < tile_capacity) {
        return;
    }

    int new_capacity = (tile_capacity == 0) ? 128 : tile_capacity * 2;
    TileRecord *new_tiles = (TileRecord *)realloc(tiles, (size_t)new_capacity * sizeof(TileRecord));
    if (new_tiles == NULL) {
        fail_and_exit("Failed to allocate tile metadata array");
    }

    tiles = new_tiles;
    tile_capacity = new_capacity;
}

static void trim_newline(char *text) {
    size_t length = strlen(text);
    while (length > 0 && (text[length - 1] == '\n' || text[length - 1] == '\r')) {
        text[length - 1] = '\0';
        length--;
    }
}

static void read_manifest(const char *directory) {
    char manifest_path[PATH_MAX];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", directory);

    FILE *file = fopen(manifest_path, "r");
    if (file == NULL) {
        perror("Failed to open manifest.txt");
        exit(EXIT_FAILURE);
    }

    memset(&manifest_data, 0, sizeof(manifest_data));
    manifest_data.threshold = 2.0;

    char line[256];
    while (fgets(line, sizeof(line), file) != NULL) {
        trim_newline(line);

        if (sscanf(line, "WIDTH %d", &manifest_data.width) == 1) {
            continue;
        }
        if (sscanf(line, "HEIGHT %d", &manifest_data.height) == 1) {
            continue;
        }
        if (sscanf(line, "CHUNK_SIZE %d", &manifest_data.chunk_size) == 1) {
            continue;
        }
        if (sscanf(line, "MAX_ITERATIONS %d", &manifest_data.max_iterations) == 1) {
            continue;
        }
        if (sscanf(line, "THRESHOLD %lf", &manifest_data.threshold) == 1) {
            continue;
        }
        if (sscanf(line, "FUNCTION %63s", manifest_data.function_name) == 1) {
            continue;
        }
        if (sscanf(line, "C_RE %lf", &manifest_data.c_re) == 1) {
            continue;
        }
        if (sscanf(line, "C_IM %lf", &manifest_data.c_im) == 1) {
            continue;
        }
    }

    fclose(file);

    if (manifest_data.width <= 0 || manifest_data.height <= 0 || manifest_data.chunk_size <= 0) {
        fail_and_exit("Manifest is missing required image dimensions or chunk size");
    }
}

static void append_tile_record(const char *path, long payload_offset, const int *header) {
    ensure_tile_capacity();

    TileRecord *tile = &tiles[tile_count++];
    memset(tile, 0, sizeof(*tile));

    snprintf(tile->path, sizeof(tile->path), "%s", path);
    tile->payload_offset = payload_offset;
    tile->start_x = header[0];
    tile->start_y = header[1];
    tile->end_x = header[2];
    tile->end_y = header[3];
    tile->chunk_id = header[4];
    tile->tile_width = tile->end_x - tile->start_x + 1;
    tile->tile_height = tile->end_y - tile->start_y + 1;
    tile->texture_id = 0;
    tile->texture_loaded = 0;
    tile->last_used_frame = 0;
    tile->visible_frame = 0;
}

static void index_rank_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror("Failed to open rank binary file");
        exit(EXIT_FAILURE);
    }

    while (1) {
        int header[5];
        size_t header_read = fread(header, sizeof(int), 5, file);

        if (header_read == 0) {
            break;
        }

        if (header_read != 5) {
            fprintf(stderr, "Incomplete tile header in %s\n", path);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        int tile_width = header[2] - header[0] + 1;
        int tile_height = header[3] - header[1] + 1;
        if (tile_width <= 0 || tile_height <= 0) {
            fprintf(stderr, "Invalid tile dimensions in %s\n", path);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        long payload_offset = ftell(file);
        append_tile_record(path, payload_offset, header);

        size_t payload_size = (size_t)tile_width * (size_t)tile_height;
        if (fseek(file, (long)payload_size, SEEK_CUR) != 0) {
            fprintf(stderr, "Failed to seek over tile payload in %s\n", path);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}

static int has_rank_bin_suffix(const char *name) {
    return strncmp(name, "rank_", 5) == 0 && strstr(name, ".bin") != NULL;
}

static int compare_tiles_by_chunk_id(const void *a, const void *b) {
    const TileRecord *left = (const TileRecord *)a;
    const TileRecord *right = (const TileRecord *)b;
    if (left->chunk_id < right->chunk_id) {
        return -1;
    }
    if (left->chunk_id > right->chunk_id) {
        return 1;
    }
    return 0;
}

static void build_tile_index(const char *directory) {
    DIR *dir = opendir(directory);
    if (dir == NULL) {
        perror("Failed to open data directory");
        exit(EXIT_FAILURE);
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!has_rank_bin_suffix(entry->d_name)) {
            continue;
        }

        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/%s", directory, entry->d_name);
        index_rank_file(path);
    }

    closedir(dir);

    if (tile_count == 0) {
        fail_and_exit("No rank_*.bin files were found in the data directory");
    }

    qsort(tiles, (size_t)tile_count, sizeof(TileRecord), compare_tiles_by_chunk_id);
}

static double image_aspect_ratio(void) {
    return (double)manifest_data.width / (double)manifest_data.height;
}

static double base_view_world_height(void) {
    return 3.0;
}

static double current_view_world_height(void) {
    return base_view_world_height() / camera_state.zoom;
}

static double current_view_world_width(void) {
    return current_view_world_height() * ((double)window_width / (double)window_height);
}

static void clamp_camera(void) {
    double half_w = current_view_world_width() * 0.5;
    double half_h = current_view_world_height() * 0.5;

    double min_center_x = -1.5 + half_w;
    double max_center_x =  1.5 - half_w;
    double min_center_y = -1.5 + half_h;
    double max_center_y =  1.5 - half_h;

    if (min_center_x > max_center_x) {
        camera_state.center_x = 0.0;
    } else {
        camera_state.center_x = clamp_double(camera_state.center_x, min_center_x, max_center_x);
    }

    if (min_center_y > max_center_y) {
        camera_state.center_y = 0.0;
    } else {
        camera_state.center_y = clamp_double(camera_state.center_y, min_center_y, max_center_y);
    }
}

static void pixel_to_world(int px, int py, double *world_x, double *world_y) {
    double view_w = current_view_world_width();
    double view_h = current_view_world_height();

    double nx = ((double)px / (double)window_width) - 0.5;
    double ny = 0.5 - ((double)py / (double)window_height);

    *world_x = camera_state.center_x + nx * view_w;
    *world_y = camera_state.center_y + ny * view_h;
}

static int image_x_to_world(int x) {
    return x;
}

static int image_y_to_world(int y) {
    return y;
}

static void image_to_world(double image_x, double image_y, double *world_x, double *world_y) {
    *world_x = -1.5 + (3.0 * image_x) / (double)(manifest_data.width - 1);
    *world_y = -1.5 + (3.0 * image_y) / (double)(manifest_data.height - 1);
}

static void world_to_image(double world_x, double world_y, double *image_x, double *image_y) {
    *image_x = (world_x + 1.5) * (double)(manifest_data.width - 1) / 3.0;
    *image_y = (world_y + 1.5) * (double)(manifest_data.height - 1) / 3.0;
}

static void color_from_iteration(uint8_t value, uint8_t *r, uint8_t *g, uint8_t *b) {
    if (value >= (uint8_t)manifest_data.max_iterations) {
        *r = 0;
        *g = 0;
        *b = 0;
        return;
    }

    if (value <= 5) {
        *r = 255; *g = 64;  *b = 0;
    } else if (value <= 10) {
        *r = 255; *g = 170; *b = 0;
    } else if (value <= 20) {
        *r = 190; *g = 255; *b = 0;
    } else if (value <= 40) {
        *r = 0;   *g = 220; *b = 220;
    } else if (value <= 80) {
        *r = 0;   *g = 110; *b = 255;
    } else if (value <= 160) {
        *r = 150; *g = 0;   *b = 255;
    } else {
        *r = 255; *g = 0;   *b = 180;
    }
}

static int load_tile_texture(TileRecord *tile) {
    if (tile->texture_loaded) {
        return 1;
    }

    if (tile->tile_width > gpu_max_texture_size || tile->tile_height > gpu_max_texture_size) {
        fprintf(stderr, "Tile %d exceeds GPU max texture size (%d x %d, max=%d)\n",
                tile->chunk_id, tile->tile_width, tile->tile_height, (int)gpu_max_texture_size);
        return 0;
    }

    FILE *file = fopen(tile->path, "rb");
    if (file == NULL) {
        perror("Failed to reopen tile source file");
        return 0;
    }

    if (fseek(file, tile->payload_offset, SEEK_SET) != 0) {
        fclose(file);
        fprintf(stderr, "Failed to seek to tile payload for chunk %d\n", tile->chunk_id);
        return 0;
    }

    size_t element_count = (size_t)tile->tile_width * (size_t)tile->tile_height;
    uint8_t *raw = (uint8_t *)malloc(element_count);
    uint8_t *rgb = (uint8_t *)malloc(element_count * 3U);
    if (raw == NULL || rgb == NULL) {
        fclose(file);
        free(raw);
        free(rgb);
        fprintf(stderr, "Out of memory while loading chunk %d\n", tile->chunk_id);
        return 0;
    }

    if (fread(raw, 1, element_count, file) != element_count) {
        fclose(file);
        free(raw);
        free(rgb);
        fprintf(stderr, "Failed to read tile payload for chunk %d\n", tile->chunk_id);
        return 0;
    }
    fclose(file);

    for (size_t i = 0; i < element_count; i++) {
        color_from_iteration(raw[i], &rgb[i * 3U + 0U], &rgb[i * 3U + 1U], &rgb[i * 3U + 2U]);
    }

    glGenTextures(1, &tile->texture_id);
    glBindTexture(GL_TEXTURE_2D, tile->texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGB,
                 tile->tile_width,
                 tile->tile_height,
                 0,
                 GL_RGB,
                 GL_UNSIGNED_BYTE,
                 rgb);

    GLenum error = glGetError();
    free(raw);
    free(rgb);

    if (error != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL failed to create texture for chunk %d (error=%u)\n", tile->chunk_id, (unsigned)error);
        glDeleteTextures(1, &tile->texture_id);
        tile->texture_id = 0;
        return 0;
    }

    tile->texture_loaded = 1;
    tile->last_used_frame = frame_counter;
    return 1;
}

static int count_loaded_tiles(void) {
    int loaded = 0;
    for (int i = 0; i < tile_count; i++) {
        if (tiles[i].texture_loaded) {
            loaded++;
        }
    }
    return loaded;
}

static void evict_one_lru_tile(void) {
    int best_index = -1;
    unsigned long best_frame = 0;

    for (int i = 0; i < tile_count; i++) {
        if (!tiles[i].texture_loaded) {
            continue;
        }
        if (tiles[i].visible_frame == frame_counter) {
            continue;
        }
        if (best_index == -1 || tiles[i].last_used_frame < best_frame) {
            best_index = i;
            best_frame = tiles[i].last_used_frame;
        }
    }

    if (best_index >= 0) {
        glDeleteTextures(1, &tiles[best_index].texture_id);
        tiles[best_index].texture_id = 0;
        tiles[best_index].texture_loaded = 0;
    }
}

static void enforce_texture_budget(void) {
    while (count_loaded_tiles() > MAX_OPEN_TILES) {
        evict_one_lru_tile();
    }
}

static int tile_intersects_view(const TileRecord *tile) {
    double min_view_world_x = camera_state.center_x - current_view_world_width() * 0.5;
    double max_view_world_x = camera_state.center_x + current_view_world_width() * 0.5;
    double min_view_world_y = camera_state.center_y - current_view_world_height() * 0.5;
    double max_view_world_y = camera_state.center_y + current_view_world_height() * 0.5;

    double tile_min_world_x;
    double tile_min_world_y;
    double tile_max_world_x;
    double tile_max_world_y;
    image_to_world((double)tile->start_x, (double)tile->start_y, &tile_min_world_x, &tile_min_world_y);
    image_to_world((double)tile->end_x, (double)tile->end_y, &tile_max_world_x, &tile_max_world_y);

    if (tile_min_world_x > tile_max_world_x) {
        double tmp = tile_min_world_x;
        tile_min_world_x = tile_max_world_x;
        tile_max_world_x = tmp;
    }
    if (tile_min_world_y > tile_max_world_y) {
        double tmp = tile_min_world_y;
        tile_min_world_y = tile_max_world_y;
        tile_max_world_y = tmp;
    }

    return !(tile_max_world_x < min_view_world_x || tile_min_world_x > max_view_world_x ||
             tile_max_world_y < min_view_world_y || tile_min_world_y > max_view_world_y);
}

static void try_load_visible_tiles(void) {
    int loads_this_frame = 0;

    for (int i = 0; i < tile_count; i++) {
        if (!tile_intersects_view(&tiles[i])) {
            continue;
        }

        tiles[i].visible_frame = frame_counter;

        if (!tiles[i].texture_loaded && loads_this_frame < MAX_TILE_LOADS_PER_FRAME) {
            if (count_loaded_tiles() >= MAX_OPEN_TILES) {
                evict_one_lru_tile();
            }
            if (load_tile_texture(&tiles[i])) {
                loads_this_frame++;
            }
        }
    }

    enforce_texture_budget();
}

static void draw_tile(const TileRecord *tile) {
    double world_x0;
    double world_y0;
    double world_x1;
    double world_y1;
    image_to_world((double)tile->start_x, (double)tile->start_y, &world_x0, &world_y0);
    image_to_world((double)tile->end_x, (double)tile->end_y, &world_x1, &world_y1);

    glBindTexture(GL_TEXTURE_2D, tile->texture_id);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2d(world_x0, world_y0);
    glTexCoord2f(1.0f, 0.0f); glVertex2d(world_x1, world_y0);
    glTexCoord2f(1.0f, 1.0f); glVertex2d(world_x1, world_y1);
    glTexCoord2f(0.0f, 1.0f); glVertex2d(world_x0, world_y1);
    glEnd();
}

static void render_scene(void) {
    frame_counter++;
    try_load_visible_tiles();

    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double half_w = current_view_world_width() * 0.5;
    double half_h = current_view_world_height() * 0.5;
    glOrtho(camera_state.center_x - half_w,
            camera_state.center_x + half_w,
            camera_state.center_y - half_h,
            camera_state.center_y + half_h,
            -1.0,
            1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    for (int i = 0; i < tile_count; i++) {
        if (!tiles[i].texture_loaded) {
            continue;
        }
        if (!tile_intersects_view(&tiles[i])) {
            continue;
        }
        tiles[i].last_used_frame = frame_counter;
        draw_tile(&tiles[i]);
    }
    glDisable(GL_TEXTURE_2D);

    glutSwapBuffers();
}

static void display_callback(void) {
    render_scene();
}

static void reshape_callback(int w, int h) {
    window_width = (w > 1) ? w : 1;
    window_height = (h > 1) ? h : 1;
    glViewport(0, 0, window_width, window_height);
    clamp_camera();
    glutPostRedisplay();
}

static void zoom_about_pixel(int mouse_x, int mouse_y, double zoom_factor) {
    double before_x;
    double before_y;
    pixel_to_world(mouse_x, mouse_y, &before_x, &before_y);

    camera_state.zoom *= zoom_factor;
    camera_state.zoom = clamp_double(camera_state.zoom, 1.0, 1000000.0);
    clamp_camera();

    double after_x;
    double after_y;
    pixel_to_world(mouse_x, mouse_y, &after_x, &after_y);

    camera_state.center_x += before_x - after_x;
    camera_state.center_y += before_y - after_y;
    clamp_camera();
    glutPostRedisplay();
}

static void mouse_callback(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            dragging = 1;
            last_mouse_x = x;
            last_mouse_y = y;
        } else {
            dragging = 0;
        }
        return;
    }

    if (state != GLUT_DOWN) {
        return;
    }

    if (button == 3) {
        zoom_about_pixel(x, y, 1.25);
    } else if (button == 4) {
        zoom_about_pixel(x, y, 1.0 / 1.25);
    }
}

static void motion_callback(int x, int y) {
    if (!dragging) {
        return;
    }

    double dx_pixels = (double)(x - last_mouse_x);
    double dy_pixels = (double)(y - last_mouse_y);

    camera_state.center_x -= dx_pixels * current_view_world_width() / (double)window_width;
    camera_state.center_y += dy_pixels * current_view_world_height() / (double)window_height;
    clamp_camera();

    last_mouse_x = x;
    last_mouse_y = y;
    glutPostRedisplay();
}

static void free_all_tile_textures(void) {
    for (int i = 0; i < tile_count; i++) {
        if (tiles[i].texture_loaded) {
            glDeleteTextures(1, &tiles[i].texture_id);
            tiles[i].texture_loaded = 0;
            tiles[i].texture_id = 0;
        }
    }
}

static void cleanup_and_exit(void) {
    free_all_tile_textures();
    free(tiles);
    exit(EXIT_SUCCESS);
}

static void render_offscreen_region(uint8_t *dst_rgb,
                                    int export_width,
                                    int export_height,
                                    int tile_px_x,
                                    int tile_px_y,
                                    int tile_px_w,
                                    int tile_px_h) {
    const GLint old_viewport[4] = {0, 0, window_width, window_height};

    glViewport(0, 0, tile_px_w, tile_px_h);
    glClear(GL_COLOR_BUFFER_BIT);

    double full_view_w = current_view_world_width();
    double full_view_h = current_view_world_height();

    double sub_left = camera_state.center_x - full_view_w * 0.5 + full_view_w * ((double)tile_px_x / (double)export_width);
    double sub_right = camera_state.center_x - full_view_w * 0.5 + full_view_w * ((double)(tile_px_x + tile_px_w) / (double)export_width);
    double sub_top = camera_state.center_y + full_view_h * 0.5 - full_view_h * ((double)tile_px_y / (double)export_height);
    double sub_bottom = camera_state.center_y + full_view_h * 0.5 - full_view_h * ((double)(tile_px_y + tile_px_h) / (double)export_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(sub_left, sub_right, sub_bottom, sub_top, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    for (int i = 0; i < tile_count; i++) {
        if (!tiles[i].texture_loaded && !load_tile_texture(&tiles[i])) {
            continue;
        }
        draw_tile(&tiles[i]);
    }
    glDisable(GL_TEXTURE_2D);
    glFinish();

    uint8_t *tile_pixels = (uint8_t *)malloc((size_t)tile_px_w * (size_t)tile_px_h * 3U);
    if (tile_pixels == NULL) {
        fail_and_exit("Failed to allocate temporary export tile buffer");
    }

    glReadPixels(0, 0, tile_px_w, tile_px_h, GL_RGB, GL_UNSIGNED_BYTE, tile_pixels);

    for (int row = 0; row < tile_px_h; row++) {
        size_t src_offset = (size_t)row * (size_t)tile_px_w * 3U;
        size_t dst_row = (size_t)(export_height - 1 - (tile_px_y + row));
        size_t dst_offset = (dst_row * (size_t)export_width + (size_t)tile_px_x) * 3U;
        memcpy(dst_rgb + dst_offset, tile_pixels + src_offset, (size_t)tile_px_w * 3U);
    }

    free(tile_pixels);
    glViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3]);
}

static void export_current_view_to_jpeg(const char *output_path, int export_size) {
    uint8_t *rgb = (uint8_t *)malloc((size_t)export_size * (size_t)export_size * 3U);
    if (rgb == NULL) {
        fprintf(stderr, "Failed to allocate export buffer for %dx%d JPEG\n", export_size, export_size);
        return;
    }

    int tile_render_size = 1024;
    if (tile_render_size > export_size) {
        tile_render_size = export_size;
    }

    for (int y = 0; y < export_size; y += tile_render_size) {
        for (int x = 0; x < export_size; x += tile_render_size) {
            int w = tile_render_size;
            int h = tile_render_size;
            if (x + w > export_size) {
                w = export_size - x;
            }
            if (y + h > export_size) {
                h = export_size - y;
            }
            render_offscreen_region(rgb, export_size, export_size, x, y, w, h);
        }
    }

    if (!stbi_write_jpg(output_path, export_size, export_size, 3, rgb, 95)) {
        fprintf(stderr, "Failed to write JPEG to %s\n", output_path);
    } else {
        printf("Saved JPEG export to %s\n", output_path);
    }

    free(rgb);
    glutPostRedisplay();
}

static void keyboard_callback(unsigned char key, int x, int y) {
    (void)x;
    (void)y;

    if (key == 27 || key == 'q') {
        cleanup_and_exit();
        return;
    }

    if (key == '+' || key == '=') {
        zoom_about_pixel(window_width / 2, window_height / 2, 1.25);
        return;
    }

    if (key == '-' || key == '_') {
        zoom_about_pixel(window_width / 2, window_height / 2, 1.0 / 1.25);
        return;
    }

    if (key == '0') {
        camera_state.center_x = 0.0;
        camera_state.center_y = 0.0;
        camera_state.zoom = 1.0;
        clamp_camera();
        glutPostRedisplay();
        return;
    }

    if (key == 'j' || key == 'J') {
        char output_path[PATH_MAX];
        snprintf(output_path, sizeof(output_path), "%s/view_%0.2fx.jpg", data_dir, camera_state.zoom);
        export_current_view_to_jpeg(output_path, JPEG_EXPORT_SIZE);
        return;
    }
}

static void print_controls(void) {
    printf("Controls:\n");
    printf("  Mouse wheel : zoom in / out\n");
    printf("  Left drag   : pan\n");
    printf("  j           : export current view as %dx%d JPEG\n", JPEG_EXPORT_SIZE, JPEG_EXPORT_SIZE);
    printf("  0           : reset camera\n");
    printf("  q or Esc    : quit\n");
}

static void init_opengl_state(void) {
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &gpu_max_texture_size);
    if (gpu_max_texture_size <= 0) {
        gpu_max_texture_size = 4096;
    }

    printf("GPU max texture size: %d\n", (int)gpu_max_texture_size);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
}

int main(int argc, char **argv) {
    if (argc >= 2) {
        snprintf(data_dir, sizeof(data_dir), "%s", argv[1]);
    }

    read_manifest(data_dir);
    build_tile_index(data_dir);

    printf("Indexed %d tiles from %s\n", tile_count, data_dir);
    printf("Manifest: WIDTH=%d HEIGHT=%d CHUNK_SIZE=%d MAX_ITERATIONS=%d\n",
           manifest_data.width,
           manifest_data.height,
           manifest_data.chunk_size,
           manifest_data.max_iterations);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Julia Tile Viewer");

    init_opengl_state();
    print_controls();

    glutDisplayFunc(display_callback);
    glutReshapeFunc(reshape_callback);
    glutMouseFunc(mouse_callback);
    glutMotionFunc(motion_callback);
    glutKeyboardFunc(keyboard_callback);
    glutIdleFunc(display_callback);

    glutMainLoop();
    return 0;
}