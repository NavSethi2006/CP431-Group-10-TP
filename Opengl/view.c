

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
    char path[PATH_MAX];
    FILE *fp;
} RankFile;


static int ends_with(const char *s, const char *suffix) {
    size_t ls = strlen(s);
    size_t lf = strlen(suffix);
    if (lf > ls) {
        return 0;
    }
    return strcmp(s + ls - lf, suffix) == 0;
}

typedef struct {
    double t;
    uint8_t r;
    uint8_t g;
    uint8_t b;
} ColorStop;

static int compare_rank_file_names(const void *a, const void *b) {
    const char *const *sa = (const char *const *)a;
    const char *const *sb = (const char *const *)b;
    return strcmp(*sa, *sb);
}

// smoothly blends between two color values a and b using a linear interpolation
static uint8_t interpolation(uint8_t a, uint8_t b, double t) {
    return (uint8_t)lrint((1.0 - t) * (double)a + t * (double)b);
}

// Converts the iteration value to an RGB color
static void iteration_to_rgb(uint8_t value, int max_iterations, uint8_t *r, uint8_t *g, uint8_t *b) {
    
    static const ColorStop palette[] = {
    {0.00,   30,  30,  30},  
    {0.06,  20,  30,  90}, 
    {0.12,  255,  255, 255},  
    {0.12,  30,  80, 170},  
    {0.16, 255, 220,  40},   
    {0.17, 255, 160,  40},   
    {0.20,  20, 140, 220},    
    {0.21, 255, 220,  40},   
    {0.25, 255, 160,  40},   
    {0.30,  40, 200, 180},   
    {0.40,  50, 220, 120},   
    {0.50, 120, 230,  80},   
    {0.60, 220, 240,  60},   
    {0.70, 255, 220,  40},   
    {0.80, 255, 160,  40},   
    {0.88, 255,  90,  70},   
    {0.94, 255,  70, 150},   
    {1.00, 180,  60, 255}    
};
    const int palette_count = (int)(sizeof(palette) / sizeof(palette[0]));

    if (value >= (uint8_t)max_iterations) {
        *r = 0;
        *g = 0;
        *b = 0;
        return;
    }

    double t = (double)value / (double)(max_iterations > 0 ? max_iterations : 1);

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
            *r = interpolation(left->r, right->r, local_t);
            *g = interpolation(left->g, right->g, local_t);
            *b = interpolation(left->b, right->b, local_t);
            return;
        }
    }

    *r = palette[palette_count - 1].r;
    *g = palette[palette_count - 1].g;
    *b = palette[palette_count - 1].b;
}
// Parses the manifest file
static void parse_manifest(const char *manifest_path, Manifest *manifest) {
    FILE *fp = fopen(manifest_path, "r");
   

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

   
}

// Collects the names of the rank files in the tiles directory
static void collect_rank_file_names(const char *tiles_dir, char ***names_out, int *count_out) {
    DIR *dir = opendir(tiles_dir);
   

    int capacity = 16;
    int count = 0;
    char **names = (char **)malloc((size_t)capacity * sizeof(char *));
    if (names == NULL) {
        closedir(dir);
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {

        if (strncmp(entry->d_name, "rank_", strlen("rank_")) == 0 && ends_with( entry->d_name, ".bin")) {
            if (count == capacity) {
                capacity *= 2;
                char **grown = (char **)realloc(names, (size_t)capacity * sizeof(char *));
               
                names = grown;
            }

            names[count] = strdup(entry->d_name);
            
            count++;
        }
    }

    closedir(dir);

  

    qsort(names, (size_t)count, sizeof(char *), compare_rank_file_names);

    *names_out = names;
    *count_out = count;
}

static void open_rank_files(const char *tiles_dir, RankFile **rank_files_out, int *rank_file_count_out) {
    char **names = NULL;
    int count = 0;
    collect_rank_file_names(tiles_dir, &names, &count);

    RankFile *rank_files = (RankFile *)calloc((size_t)count, sizeof(RankFile));
    

    for (int i = 0; i < count; i++) {
        snprintf(rank_files[i].path, sizeof(rank_files[i].path), "%s/%s", tiles_dir, names[i]);
        rank_files[i].fp = fopen(rank_files[i].path, "rb");
        
        free(names[i]);
    }
    free(names);

    *rank_files_out = rank_files;
    *rank_file_count_out = count;
}

// Scans the chunks in the rank files by reading the header and the payload
static void scan_chunks(
    RankFile *rank_files,
    int rank_file_count,
    ChunkRecord **chunks_out,
    int *chunk_count_out,
    const Manifest *manifest,
    int chunks_x,
    int chunks_y
) {
    int expected_chunks = chunks_x * chunks_y;
    ChunkRecord *chunks = (ChunkRecord *)malloc((size_t)expected_chunks * sizeof(ChunkRecord));
    

    int chunk_count = 0;

    for (int file_index = 0; file_index < rank_file_count; file_index++) {
        FILE *fp = rank_files[file_index].fp;
        while (1) {
            int header[5];
            long header_offset = ftell(fp);
            size_t read_count = fread(header, sizeof(int), 5, fp);
            if (read_count == 0) {
                break;
            }
          

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


            

            size_t payload_size = (size_t)record.tile_width * (size_t)record.tile_height;
            

            chunks[chunk_count++] = record;

            if (fseek(fp, (long)payload_size, SEEK_CUR) != 0) {
                fprintf(stderr, "Failed to skip payload in %s for chunk %d\n", rank_files[file_index].path, record.chunk_id);
                exit(EXIT_FAILURE);
            }
        }

        rewind(fp);
    }

  

    int *seen = (int *)calloc((size_t)expected_chunks, sizeof(int));
    

    

    

    free(seen);

    (void)manifest;
    *chunks_out = chunks;
    *chunk_count_out = chunk_count;
}

static ChunkRecord **build_chunk_grid(ChunkRecord *chunks, int chunk_count, int expected_chunks) {
    ChunkRecord **grid = (ChunkRecord **)calloc((size_t)expected_chunks, sizeof(ChunkRecord *));
   

    for (int i = 0; i < chunk_count; i++) {
        int id = chunks[i].chunk_id;
        grid[id] = &chunks[i];
    }

    return grid;
}

static int read_tile_row_sample(FILE *fp, const ChunkRecord *chunk, int src_y, uint8_t *row_buffer) {
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

static void write_jpeg_from_chunks(
    const char *output_path,
    const Manifest *manifest,
    RankFile *rank_files,
    ChunkRecord **chunk_grid,
    int chunks_x,
    int chunks_y,
    int quality
) {
    (void)chunks_y;

    FILE *out = fopen(output_path, "wb");
    

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
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    uint8_t *iteration_row = (uint8_t *)malloc((size_t)manifest->chunk_size);
    

    const size_t rgb_row_bytes = (size_t)manifest->width * 3u;
    JSAMPROW rgb_row = (JSAMPROW)malloc(rgb_row_bytes);
    

    for (int y = 0; y < manifest->height; y++) {
        int chunk_row = y / manifest->chunk_size;
        int src_y = y - chunk_row * manifest->chunk_size;

        for (int chunk_col = 0; chunk_col < chunks_x; chunk_col++) {
            int chunk_id = chunk_row * chunks_x + chunk_col;
            ChunkRecord *chunk = chunk_grid[chunk_id];
           

            FILE *fp = rank_files[chunk->rank_index].fp;
            if (!read_tile_row_sample(fp, chunk, src_y, iteration_row)) {
                fprintf(stderr, "Failed to read row %d from chunk %d in %s\n", src_y, chunk->chunk_id, chunk->path);
                exit(EXIT_FAILURE);
            }

            uint8_t *dst = rgb_row + (size_t)chunk->start_x * 3u;
            for (int local_x = 0; local_x < chunk->tile_width; local_x++) {
                iteration_to_rgb(iteration_row[local_x], manifest->max_iterations, &dst[0], &dst[1], &dst[2]);
                dst += 3;
            }
        }

        jpeg_write_scanlines(&cinfo, &rgb_row, 1);

        if ((y + 1) % 250 == 0 || y + 1 == manifest->height) {
            printf("Wrote %d / %d rows\n", y + 1, manifest->height);
            fflush(stdout);
        }
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(out);
    free(iteration_row);
    free(rgb_row);
}

int main(int argc, char **argv) {
    const char *tiles_dir = (argc >= 2) ? argv[1] : "../MPI/tiles";
    const char *output_path = (argc >= 3) ? argv[2] : "julia_output.jpg";
    int quality = (argc >= 4) ? atoi(argv[3]) : 95;

    if (quality < 1) {
        quality = 1;
    }
    if (quality > 100) {
        quality = 100;
    }

    char manifest_path[PATH_MAX];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", tiles_dir);

    Manifest manifest;
    parse_manifest(manifest_path, &manifest);

    int chunks_x = (manifest.width + manifest.chunk_size - 1) / manifest.chunk_size;
    int chunks_y = (manifest.height + manifest.chunk_size - 1) / manifest.chunk_size;
    int expected_chunks = chunks_x * chunks_y;

    RankFile *rank_files = NULL;
    int rank_file_count = 0;
    open_rank_files(tiles_dir, &rank_files, &rank_file_count);

    ChunkRecord *chunks = NULL;
    int chunk_count = 0;
    scan_chunks(rank_files, rank_file_count, &chunks, &chunk_count, &manifest, chunks_x, chunks_y);

    ChunkRecord **chunk_grid = build_chunk_grid(chunks, chunk_count, expected_chunks);

    printf("Saving JPEG to: %s\n", output_path);

    write_jpeg_from_chunks(output_path, &manifest, rank_files, chunk_grid, chunks_x, chunks_y, quality);

    for (int i = 0; i < rank_file_count; i++) {
        if (rank_files[i].fp != NULL) {
            fclose(rank_files[i].fp);
        }
    }

    free(rank_files);
    free(chunk_grid);
    free(chunks);

    printf("Done. JPEG saved to %s\n", output_path);
    return 0;
}