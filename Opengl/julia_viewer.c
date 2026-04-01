

#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <jpeglib.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define DEFAULT_TILES_DIR "../MPI/tiles"
#define DEFAULT_OUTPUT_JPEG "julia_opengl.jpg"
#define DEFAULT_JPEG_QUALITY 95
#define DEFAULT_RENDER_STRIP_HEIGHT 1024

typedef struct {
    int width;
    int height;
    int chunk_size;
    int max_iterations;
    double threshold;
    double c_re;
    double c_im;
    char function_name[64];
} Manifest;

typedef struct {
    char path[PATH_MAX];
    FILE *fp;
} RankFile;

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
    int rank_index;
} ChunkRecord;

typedef struct {
    double t;
    uint8_t r;
    uint8_t g;
    uint8_t b;
} ColorStop;

static GLuint g_fbo = 0;
static GLuint g_color_tex = 0;
static GLuint g_input_tex = 0;
static int g_window = 0;
static int g_strip_width = 0;
static int g_strip_height = 0;

static void fail(const char *message) {
    perror(message);
    exit(EXIT_FAILURE);
}

static void fail_message(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static int starts_with(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static int ends_with(const char *s, const char *suffix) {
    size_t ls = strlen(s);
    size_t lf = strlen(suffix);
    if (lf > ls) {
        return 0;
    }
    return strcmp(s + ls - lf, suffix) == 0;
}

static int compare_rank_file_names(const void *a, const void *b) {
    const char *const *sa = (const char *const *)a;
    const char *const *sb = (const char *const *)b;
    return strcmp(*sa, *sb);
}

static uint8_t interpolate_channel(uint8_t a, uint8_t b, double t) {
    return (uint8_t)lrint((1.0 - t) * (double)a + t * (double)b);
}

static void iteration_to_rgb(uint8_t value, int max_iterations, uint8_t *r, uint8_t *g, uint8_t *b) {
    static const ColorStop palette[] = {
        {0.00,   0,   0,   0},
        {0.02,  25,   7,  26},
        {0.05,  15,  30, 120},
        {0.10,  30,  80, 200},
        {0.18,  20, 160, 230},
        {0.28,  30, 220, 180},
        {0.40, 120, 235,  80},
        {0.52, 245, 240,  60},
        {0.64, 255, 180,  35},
        {0.76, 255, 100,  60},
        {0.88, 255,  70, 170},
        {1.00, 210, 120, 255}
    };
    const int palette_count = (int)(sizeof(palette) / sizeof(palette[0]));

    int effective_max = max_iterations;
    if (effective_max > 255) {
        effective_max = 255;
    }
    if (effective_max < 1) {
        effective_max = 1;
    }

    if (value >= (uint8_t)effective_max) {
        *r = 0;
        *g = 0;
        *b = 0;
        return;
    }

    double t = sqrt((double)value / (double)effective_max);

    for (int i = 0; i < palette_count - 1; i++) {
        const ColorStop *left = &palette[i];
        const ColorStop *right = &palette[i + 1];
        if (t <= right->t) {
            double local_t = (t - left->t) / (right->t - left->t);
            if (local_t < 0.0) {
                local_t = 0.0;
            }
            if (local_t > 1.0) {
                local_t = 1.0;
            }
            *r = interpolate_channel(left->r, right->r, local_t);
            *g = interpolate_channel(left->g, right->g, local_t);
            *b = interpolate_channel(left->b, right->b, local_t);
            return;
        }
    }

    *r = palette[palette_count - 1].r;
    *g = palette[palette_count - 1].g;
    *b = palette[palette_count - 1].b;
}

static void parse_manifest(const char *manifest_path, Manifest *manifest) {
    FILE *fp = fopen(manifest_path, "r");
    if (fp == NULL) {
        fail("Failed to open manifest.txt");
    }

    memset(manifest, 0, sizeof(*manifest));
    manifest->threshold = 2.0;
    strcpy(manifest->function_name, "unknown");

    char key[128];
    while (fscanf(fp, "%127s", key) == 1) {
        if (strcmp(key, "WIDTH") == 0) {
            fscanf(fp, "%d", &manifest->width);
        } else if (strcmp(key, "HEIGHT") == 0) {
            fscanf(fp, "%d", &manifest->height);
        } else if (strcmp(key, "CHUNK_SIZE") == 0) {
            fscanf(fp, "%d", &manifest->chunk_size);
        } else if (strcmp(key, "MAX_ITERATIONS") == 0) {
            fscanf(fp, "%d", &manifest->max_iterations);
        } else if (strcmp(key, "THRESHOLD") == 0) {
            fscanf(fp, "%lf", &manifest->threshold);
        } else if (strcmp(key, "FUNCTION") == 0) {
            fscanf(fp, "%63s", manifest->function_name);
        } else if (strcmp(key, "C_RE") == 0) {
            fscanf(fp, "%lf", &manifest->c_re);
        } else if (strcmp(key, "C_IM") == 0) {
            fscanf(fp, "%lf", &manifest->c_im);
        } else {
            char discard[256];
            fgets(discard, sizeof(discard), fp);
        }
    }

    fclose(fp);

    if (manifest->width <= 0 || manifest->height <= 0 || manifest->chunk_size <= 0 || manifest->max_iterations <= 0) {
        fail_message("Manifest is missing required values or contains invalid dimensions.");
    }
}

static void collect_rank_file_names(const char *tiles_dir, char ***names_out, int *count_out) {
    DIR *dir = opendir(tiles_dir);
    if (dir == NULL) {
        fail("Failed to open tiles directory");
    }

    int capacity = 16;
    int count = 0;
    char **names = (char **)malloc((size_t)capacity * sizeof(char *));
    if (names == NULL) {
        closedir(dir);
        fail("Failed to allocate rank file name list");
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (starts_with(entry->d_name, "rank_") && ends_with(entry->d_name, ".bin")) {
            if (count == capacity) {
                capacity *= 2;
                char **grown = (char **)realloc(names, (size_t)capacity * sizeof(char *));
                if (grown == NULL) {
                    closedir(dir);
                    fail("Failed to grow rank file name list");
                }
                names = grown;
            }

            names[count] = strdup(entry->d_name);
            if (names[count] == NULL) {
                closedir(dir);
                fail("Failed to duplicate rank file name");
            }
            count++;
        }
    }

    closedir(dir);

    if (count == 0) {
        free(names);
        fail_message("No rank_*.bin files were found in the tiles directory.");
    }

    qsort(names, (size_t)count, sizeof(char *), compare_rank_file_names);
    *names_out = names;
    *count_out = count;
}

static void open_rank_files(const char *tiles_dir, RankFile **rank_files_out, int *rank_file_count_out) {
    char **names = NULL;
    int count = 0;
    collect_rank_file_names(tiles_dir, &names, &count);

    RankFile *rank_files = (RankFile *)calloc((size_t)count, sizeof(RankFile));
    if (rank_files == NULL) {
        fail("Failed to allocate rank file table");
    }

    for (int i = 0; i < count; i++) {
        snprintf(rank_files[i].path, sizeof(rank_files[i].path), "%s/%s", tiles_dir, names[i]);
        rank_files[i].fp = fopen(rank_files[i].path, "rb");
        if (rank_files[i].fp == NULL) {
            fail("Failed to open rank bin file");
        }
        free(names[i]);
    }
    free(names);

    *rank_files_out = rank_files;
    *rank_file_count_out = count;
}

static void scan_chunks(
    RankFile *rank_files,
    int rank_file_count,
    ChunkRecord **chunks_out,
    int *chunk_count_out,
    int chunks_x,
    int chunks_y
) {
    const int expected_chunks = chunks_x * chunks_y;
    ChunkRecord *chunks = (ChunkRecord *)malloc((size_t)expected_chunks * sizeof(ChunkRecord));
   
    int chunk_count = 0;
    for (int file_index = 0; file_index < rank_file_count; file_index++) {
        FILE *fp = rank_files[file_index].fp;
        while (1) {
            int header[5];
            size_t read_count = fread(header, sizeof(int), 5, fp);
           
            ChunkRecord record;
            memset(&record, 0, sizeof(record));
            snprintf(record.path, sizeof(record.path), "%s", rank_files[file_index].path);
            record.start_x = header[0];
            record.start_y = header[1];
            record.end_x = header[2];
            record.end_y = header[3];
            record.chunk_id = header[4];
            record.tile_width = record.end_x - record.start_x + 1;
            record.tile_height = record.end_y - record.start_y + 1;
            record.rank_index = file_index;
            record.payload_offset = ftell(fp);

            

            chunks[chunk_count++] = record;

            size_t payload_size = (size_t)record.tile_width * (size_t)record.tile_height;
            if (fseek(fp, (long)payload_size, SEEK_CUR) != 0) {
                fprintf(stderr, "Failed to skip payload in %s for chunk %d\n", rank_files[file_index].path, record.chunk_id);
                exit(EXIT_FAILURE);
            }
        }
        rewind(fp);
    }

   
    int *seen = (int *)calloc((size_t)expected_chunks, sizeof(int));
   

   

    free(seen);
    *chunks_out = chunks;
    *chunk_count_out = chunk_count;
}

static ChunkRecord **build_chunk_grid(ChunkRecord *chunks, int chunk_count, int expected_chunks) {
    ChunkRecord **grid = (ChunkRecord **)calloc((size_t)expected_chunks, sizeof(ChunkRecord *));
    
    for (int i = 0; i < chunk_count; i++) {
        grid[chunks[i].chunk_id] = &chunks[i];
    }
    return grid;
}

static int read_tile_row(FILE *fp, const ChunkRecord *chunk, int src_y, uint8_t *row_buffer) {
    if (src_y < 0 || src_y >= chunk->tile_height) {
        return 0;
    }

    long row_offset = chunk->payload_offset + (long)src_y * (long)chunk->tile_width;
    if (fseek(fp, row_offset, SEEK_SET) != 0) {
        return 0;
    }

    size_t expected = (size_t)chunk->tile_width;
    return fread(row_buffer, 1, expected, fp) == expected;
}

static void init_hidden_gl_context(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(1, 1);
    glutInitWindowPosition(-10000, -10000);
    g_window = glutCreateWindow("julia_opengl_export");
    glutHideWindow();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
}

static void setup_strip_renderer(int strip_width, int strip_height) {
    g_strip_width = strip_width;
    g_strip_height = strip_height;

    glGenTextures(1, &g_input_tex);
    glBindTexture(GL_TEXTURE_2D, g_input_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, strip_width, strip_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glGenTextures(1, &g_color_tex);
    glBindTexture(GL_TEXTURE_2D, g_color_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, strip_width, strip_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_color_tex, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "OpenGL framebuffer incomplete: 0x%x\n", status);
        exit(EXIT_FAILURE);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void destroy_strip_renderer(void) {
    if (g_fbo != 0) {
        glDeleteFramebuffers(1, &g_fbo);
        g_fbo = 0;
    }
    if (g_input_tex != 0) {
        glDeleteTextures(1, &g_input_tex);
        g_input_tex = 0;
    }
    if (g_color_tex != 0) {
        glDeleteTextures(1, &g_color_tex);
        g_color_tex = 0;
    }
    if (g_window != 0) {
        glutDestroyWindow(g_window);
        g_window = 0;
    }
}

static void render_rgb_strip_via_opengl(const uint8_t *cpu_rgb, uint8_t *gpu_rgb, int width, int strip_height) {
    glBindTexture(GL_TEXTURE_2D, g_input_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, strip_height, GL_RGB, GL_UNSIGNED_BYTE, cpu_rgb);

    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glViewport(0, 0, width, strip_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_input_tex);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glReadPixels(0, 0, width, strip_height, GL_RGB, GL_UNSIGNED_BYTE, gpu_rgb);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void write_jpeg_from_chunks_with_opengl(
    const char *output_path,
    const Manifest *manifest,
    RankFile *rank_files,
    ChunkRecord **chunk_grid,
    int chunks_x,
    int jpeg_quality,
    int render_strip_height
) {
    FILE *out = fopen(output_path, "wb");
    if (out == NULL) {
        fail("Failed to open output JPEG");
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, out);

    cinfo.image_width = manifest->width;
    cinfo.image_height = manifest->height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, jpeg_quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    const int strip_height = (render_strip_height > 0) ? render_strip_height : DEFAULT_RENDER_STRIP_HEIGHT;
    const size_t cpu_rgb_bytes = (size_t)manifest->width * (size_t)strip_height * 3u;
    const size_t iter_row_capacity = (size_t)manifest->chunk_size;

    uint8_t *cpu_rgb = (uint8_t *)malloc(cpu_rgb_bytes);
    uint8_t *gpu_rgb = (uint8_t *)malloc(cpu_rgb_bytes);
    uint8_t *iter_row = (uint8_t *)malloc(iter_row_capacity);
    if (cpu_rgb == NULL || gpu_rgb == NULL || iter_row == NULL) {
        fclose(out);
        free(cpu_rgb);
        free(gpu_rgb);
        free(iter_row);
        fail("Failed to allocate strip buffers");
    }

    setup_strip_renderer(manifest->width, strip_height);

    for (int strip_y = 0; strip_y < manifest->height; strip_y += strip_height) {
        const int current_strip_height = ((strip_y + strip_height) <= manifest->height)
            ? strip_height
            : (manifest->height - strip_y);

        memset(cpu_rgb, 0, (size_t)manifest->width * (size_t)current_strip_height * 3u);

        for (int local_y = 0; local_y < current_strip_height; local_y++) {
            int global_y = strip_y + local_y;
            int chunk_row = global_y / manifest->chunk_size;
            int src_y = global_y - chunk_row * manifest->chunk_size;
            uint8_t *dst_row = cpu_rgb + (size_t)local_y * (size_t)manifest->width * 3u;

            for (int chunk_col = 0; chunk_col < chunks_x; chunk_col++) {
                int chunk_id = chunk_row * chunks_x + chunk_col;
                ChunkRecord *chunk = chunk_grid[chunk_id];
                if (chunk == NULL) {
                    fprintf(stderr, "Missing chunk grid entry for chunk id %d\n", chunk_id);
                    exit(EXIT_FAILURE);
                }

                FILE *fp = rank_files[chunk->rank_index].fp;
                if (!read_tile_row(fp, chunk, src_y, iter_row)) {
                    fprintf(stderr, "Failed to read row %d from chunk %d in %s\n", src_y, chunk->chunk_id, chunk->path);
                    exit(EXIT_FAILURE);
                }

                uint8_t *dst = dst_row + (size_t)chunk->start_x * 3u;
                for (int local_x = 0; local_x < chunk->tile_width; local_x++) {
                    iteration_to_rgb(iter_row[local_x], manifest->max_iterations, &dst[0], &dst[1], &dst[2]);
                    dst += 3;
                }
            }
        }

        render_rgb_strip_via_opengl(cpu_rgb, gpu_rgb, manifest->width, current_strip_height);

        for (int local_y = 0; local_y < current_strip_height; local_y++) {
            JSAMPROW row_ptr = gpu_rgb + (size_t)local_y * (size_t)manifest->width * 3u;
            jpeg_write_scanlines(&cinfo, &row_ptr, 1);
        }

        printf("Rendered and wrote rows %d to %d of %d\n",
               strip_y,
               strip_y + current_strip_height - 1,
               manifest->height - 1);
        fflush(stdout);
    }

    destroy_strip_renderer();
    free(cpu_rgb);
    free(gpu_rgb);
    free(iter_row);

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(out);
}

int main(int argc, char **argv) {
    const char *tiles_dir = (argc >= 2) ? argv[1] : DEFAULT_TILES_DIR;
    const char *output_path = (argc >= 3) ? argv[2] : DEFAULT_OUTPUT_JPEG;
    int jpeg_quality = (argc >= 4) ? atoi(argv[3]) : DEFAULT_JPEG_QUALITY;
    int render_strip_height = (argc >= 5) ? atoi(argv[4]) : DEFAULT_RENDER_STRIP_HEIGHT;

    if (jpeg_quality < 1) {
        jpeg_quality = 1;
    }
    if (jpeg_quality > 100) {
        jpeg_quality = 100;
    }
    if (render_strip_height < 1) {
        render_strip_height = DEFAULT_RENDER_STRIP_HEIGHT;
    }

    char manifest_path[PATH_MAX];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", tiles_dir);

    Manifest manifest;
    parse_manifest(manifest_path, &manifest);

    const int chunks_x = (manifest.width + manifest.chunk_size - 1) / manifest.chunk_size;
    const int chunks_y = (manifest.height + manifest.chunk_size - 1) / manifest.chunk_size;
    const int expected_chunks = chunks_x * chunks_y;

    RankFile *rank_files = NULL;
    int rank_file_count = 0;
    open_rank_files(tiles_dir, &rank_files, &rank_file_count);

    ChunkRecord *chunks = NULL;
    int chunk_count = 0;
    scan_chunks(rank_files, rank_file_count, &chunks, &chunk_count, chunks_x, chunks_y);

    ChunkRecord **chunk_grid = build_chunk_grid(chunks, chunk_count, expected_chunks);

    printf("OpenGL Julia export\n");
    printf("  tiles_dir=%s\n", tiles_dir);
    printf("  output=%s\n", output_path);
    printf("  width=%d\n", manifest.width);
    printf("  height=%d\n", manifest.height);
    printf("  chunk_size=%d\n", manifest.chunk_size);
    printf("  max_iterations=%d\n", manifest.max_iterations);
    printf("  function=%s\n", manifest.function_name);
    printf("  c=(%.12f, %.12f)\n", manifest.c_re, manifest.c_im);
    printf("  strip_height=%d\n", render_strip_height);
    if (manifest.max_iterations > 255) {
        printf("  note=stored iteration values are 8-bit, so displayed colors are normalized to 255\n");
    }

    init_hidden_gl_context(&argc, argv);
    write_jpeg_from_chunks_with_opengl(
        output_path,
        &manifest,
        rank_files,
        chunk_grid,
        chunks_x,
        jpeg_quality,
        render_strip_height
    );

    for (int i = 0; i < rank_file_count; i++) {
        if (rank_files[i].fp != NULL) {
            fclose(rank_files[i].fp);
        }
    }

    free(rank_files);
    free(chunk_grid);
    free(chunks);

    printf("Done. OpenGL-rendered JPEG saved to %s\n", output_path);
    return 0;
}