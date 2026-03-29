#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <sys/stat.h>
#include <errno.h>
#include <limits.h>
#include <inttypes.h>

#define MAX_ITERATIONS 100
#define THRESHOLD 2.0
#define WIDTH 200000
#define HEIGHT 200000
#define CHUNK_SIZE 4096
#define TAG_WORK 1
#define TAG_RESULT_META 2
#define TAG_STOP 4

static char output_dir[PATH_MAX] = "tiles";

typedef struct {
    long long chunks_computed;
    double compute_seconds;
    double io_seconds;
    double total_seconds;
    unsigned long long bytes_written;
} BenchmarkStats;

void parse_args(int argc, char *argv[], int rank);
void ensure_output_dir(int rank);
void write_manifest_file(void);
FILE *open_rank_output_file(int rank);
void write_benchmark_file(const BenchmarkStats *all_stats, int size, int total_chunks, double wall_seconds);

int compute_julia_value(double x, double y) {
    double complex z = x + y * I;
    double complex c = -1;
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        if (cabs(z) > THRESHOLD) {
            return i;
        }
        z = z * z + c;
    }
    return MAX_ITERATIONS;
}

void parse_args(int argc, char *argv[], int rank) {
    if (argc >= 2) {
        if (snprintf(output_dir, sizeof(output_dir), "%s", argv[1]) >= (int)sizeof(output_dir)) {
            if (rank == 0) {
                fprintf(stderr, "Output directory path is too long\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (rank == 0) {
        printf("Julia set run configuration:\n");
        printf("  WIDTH=%d\n", WIDTH);
        printf("  HEIGHT=%d\n", HEIGHT);
        printf("  CHUNK_SIZE=%d\n", CHUNK_SIZE);
        printf("  MAX_ITERATIONS=%d\n", MAX_ITERATIONS);
        printf("  output_dir=%s\n", output_dir);
    }
}

void ensure_output_dir(int rank) {
    if (rank == 0) {
        if (mkdir(output_dir, 0777) != 0 && errno != EEXIST) {
            perror("Failed to create output directory");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void write_manifest_file(void) {
    char manifest_path[PATH_MAX];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", output_dir);

    FILE *file = fopen(manifest_path, "w");
    if (file == NULL) {
        perror("Failed to open manifest file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fprintf(file, "WIDTH %d\n", WIDTH);
    fprintf(file, "HEIGHT %d\n", HEIGHT);
    fprintf(file, "CHUNK_SIZE %d\n", CHUNK_SIZE);
    fprintf(file, "MAX_ITERATIONS %d\n", MAX_ITERATIONS);
    fprintf(file, "THRESHOLD %.6f\n", THRESHOLD);
    fprintf(file, "FUNCTION z^2+c\n");
    fprintf(file, "C_RE -1.0\n");
    fprintf(file, "C_IM 0.0\n");
    fclose(file);
}

FILE *open_rank_output_file(int rank) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/rank_%d.bin", output_dir, rank);

    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        perror("Failed to open rank output file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return file;
}

int get_chunk_width(const int *chunk) {
    return chunk[2] - chunk[0] + 1;
}

int get_chunk_height(const int *chunk) {
    return chunk[3] - chunk[1] + 1;
}

size_t get_chunk_element_count(const int *chunk) {
    return (size_t)get_chunk_width(chunk) * (size_t)get_chunk_height(chunk);
}

void write_chunk_to_rank_file(FILE *file, const int *chunk, const uint8_t *buffer) {
    size_t element_count = get_chunk_element_count(chunk);

    if (fwrite(chunk, sizeof(int), 5, file) != 5) {
        perror("Failed to write chunk header to rank file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (fwrite(buffer, sizeof(uint8_t), element_count, file) != element_count) {
        perror("Failed to write chunk payload to rank file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fflush(file);
}

int *split_data_v3(int chunk_count) {
    int chunks_x = (WIDTH + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int chunks_y = (HEIGHT + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int total_chunks = chunks_x * chunks_y;

    if (chunk_count < 0 || chunk_count >= total_chunks) {
        return NULL;
    }

    int *result = (int *)malloc(5 * sizeof(int));
    if (result == NULL) {
        printf("Error: Failed to allocate memory for result\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int chunk_row = chunk_count / chunks_x;
    int chunk_col = chunk_count % chunks_x;

    int start_x = chunk_col * CHUNK_SIZE;
    int start_y = chunk_row * CHUNK_SIZE;
    int end_x = start_x + CHUNK_SIZE - 1;
    int end_y = start_y + CHUNK_SIZE - 1;

    if (end_x >= WIDTH) {
        end_x = WIDTH - 1;
    }
    if (end_y >= HEIGHT) {
        end_y = HEIGHT - 1;
    }

    result[0] = start_x;
    result[1] = start_y;
    result[2] = end_x;
    result[3] = end_y;
    result[4] = chunk_count;
    return result;
}

int get_total_chunks(void) {
    int chunks_x = (WIDTH + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int chunks_y = (HEIGHT + CHUNK_SIZE - 1) / CHUNK_SIZE;
    return chunks_x * chunks_y;
}

void compute_chunk(const int *chunk, uint8_t *buffer) {
    int start_x = chunk[0];
    int start_y = chunk[1];
    int end_x = chunk[2];
    int end_y = chunk[3];
    int chunk_width = end_x - start_x + 1;

    for (int y = start_y; y <= end_y; y++) {
        for (int x = start_x; x <= end_x; x++) {
            double real = -1.5 + (3.0 * x) / (WIDTH - 1);
            double imag = -1.5 + (3.0 * y) / (HEIGHT - 1);
            int value = compute_julia_value(real, imag);
            buffer[(y - start_y) * chunk_width + (x - start_x)] = (uint8_t)value;
        }
    }
}

void write_benchmark_file(const BenchmarkStats *all_stats, int size, int total_chunks, double wall_seconds) {
    char benchmark_path[PATH_MAX];
    snprintf(benchmark_path, sizeof(benchmark_path), "%s/benchmark.txt", output_dir);

    FILE *file = fopen(benchmark_path, "w");
    if (file == NULL) {
        perror("Failed to open benchmark file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double total_compute = 0.0;
    double total_io = 0.0;
    unsigned long long total_bytes = 0ULL;
    long long total_chunks_recorded = 0LL;
    double max_rank_total = 0.0;
    double min_rank_total = 0.0;

    for (int i = 0; i < size; i++) {
        total_compute += all_stats[i].compute_seconds;
        total_io += all_stats[i].io_seconds;
        total_bytes += all_stats[i].bytes_written;
        total_chunks_recorded += all_stats[i].chunks_computed;

        if (i == 0 || all_stats[i].total_seconds > max_rank_total) {
            max_rank_total = all_stats[i].total_seconds;
        }
        if (i == 0 || all_stats[i].total_seconds < min_rank_total) {
            min_rank_total = all_stats[i].total_seconds;
        }
    }

    fprintf(file, "Julia Set Benchmark\n");
    fprintf(file, "===================\n");
    fprintf(file, "WIDTH %d\n", WIDTH);
    fprintf(file, "HEIGHT %d\n", HEIGHT);
    fprintf(file, "CHUNK_SIZE %d\n", CHUNK_SIZE);
    fprintf(file, "MAX_ITERATIONS %d\n", MAX_ITERATIONS);
    fprintf(file, "MPI_PROCESSES %d\n", size);
    fprintf(file, "TOTAL_CHUNKS %d\n", total_chunks);
    fprintf(file, "RECORDED_CHUNKS %lld\n", total_chunks_recorded);
    fprintf(file, "WALL_SECONDS %.6f\n", wall_seconds);
    fprintf(file, "TOTAL_COMPUTE_SECONDS %.6f\n", total_compute);
    fprintf(file, "TOTAL_IO_SECONDS %.6f\n", total_io);
    fprintf(file, "TOTAL_BYTES_WRITTEN %llu\n", total_bytes);
    fprintf(file, "MAX_RANK_TOTAL_SECONDS %.6f\n", max_rank_total);
    fprintf(file, "MIN_RANK_TOTAL_SECONDS %.6f\n", min_rank_total);
    fprintf(file, "\nPer-rank statistics\n");
    fprintf(file, "rank chunks compute_seconds io_seconds total_seconds bytes_written\n");

    for (int i = 0; i < size; i++) {
        fprintf(
            file,
            "%d %lld %.6f %.6f %.6f %llu\n",
            i,
            all_stats[i].chunks_computed,
            all_stats[i].compute_seconds,
            all_stats[i].io_seconds,
            all_stats[i].total_seconds,
            all_stats[i].bytes_written
        );
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    BenchmarkStats local_stats;
    memset(&local_stats, 0, sizeof(local_stats));
    double run_start = 0.0;
    double run_end = 0.0;

    FILE *rank_output_file = NULL;
    parse_args(argc, argv, rank);

    ensure_output_dir(rank);

    rank_output_file = open_rank_output_file(rank);
    MPI_Barrier(MPI_COMM_WORLD);
    run_start = MPI_Wtime();

    if (rank == 0) {
        write_manifest_file();

        int total_chunks = get_total_chunks();
        int next_chunk = 0;
        int completed_chunks = 0;

        for (int worker = 1; worker < size; worker++) {
            if (next_chunk < total_chunks) {
                int *chunk = split_data_v3(next_chunk);
                MPI_Send(chunk, 5, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                next_chunk++;
                free(chunk);
            } else {
                MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
            }
        }

        while (completed_chunks < total_chunks) {
            if (next_chunk < total_chunks) {
                int *chunk = split_data_v3(next_chunk);
                size_t element_count = get_chunk_element_count(chunk);
                uint8_t *buffer = (uint8_t *)malloc(element_count * sizeof(uint8_t));
                if (buffer == NULL) {
                    printf("Error: Failed to allocate memory for rank 0 chunk buffer\n");
                    free(chunk);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                next_chunk++;
                double compute_start = MPI_Wtime();
                compute_chunk(chunk, buffer);
                local_stats.compute_seconds += MPI_Wtime() - compute_start;

                double io_start = MPI_Wtime();
                write_chunk_to_rank_file(rank_output_file, chunk, buffer);
                local_stats.io_seconds += MPI_Wtime() - io_start;

                local_stats.chunks_computed++;
                local_stats.bytes_written += (unsigned long long)(5 * sizeof(int) + element_count * sizeof(uint8_t));
                completed_chunks++;

                free(buffer);
                free(chunk);
            }

            int flag = 0;
            MPI_Status status;
            do {
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT_META, MPI_COMM_WORLD, &flag, &status);
                if (flag) {
                    int worker = status.MPI_SOURCE;
                    int finished_chunk[5];
                    MPI_Recv(finished_chunk, 5, MPI_INT, worker, TAG_RESULT_META, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    completed_chunks++;

                    if (next_chunk < total_chunks) {
                        int *next = split_data_v3(next_chunk);
                        MPI_Send(next, 5, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                        next_chunk++;
                        free(next);
                    } else {
                        MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
                    }
                }
            } while (flag);

            if (next_chunk >= total_chunks && completed_chunks < total_chunks) {
                int finished_chunk[5];
                MPI_Status recv_status;
                MPI_Recv(finished_chunk, 5, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT_META, MPI_COMM_WORLD, &recv_status);
                completed_chunks++;
                MPI_Send(NULL, 0, MPI_INT, recv_status.MPI_SOURCE, TAG_STOP, MPI_COMM_WORLD);
            }
        }
    } else {
        while (1) {
            MPI_Status status;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP) {
                MPI_Recv(NULL, 0, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            }

            if (status.MPI_TAG == TAG_WORK) {
                int chunk[5];
                MPI_Recv(chunk, 5, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                size_t element_count = get_chunk_element_count(chunk);
                uint8_t *buffer = (uint8_t *)malloc(element_count * sizeof(uint8_t));
                if (buffer == NULL) {
                    printf("Error: Failed to allocate memory for worker chunk buffer\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                double compute_start = MPI_Wtime();
                compute_chunk(chunk, buffer);
                local_stats.compute_seconds += MPI_Wtime() - compute_start;

                double io_start = MPI_Wtime();
                write_chunk_to_rank_file(rank_output_file, chunk, buffer);
                local_stats.io_seconds += MPI_Wtime() - io_start;

                local_stats.chunks_computed++;
                local_stats.bytes_written += (unsigned long long)(5 * sizeof(int) + element_count * sizeof(uint8_t));
                MPI_Send(chunk, 5, MPI_INT, 0, TAG_RESULT_META, MPI_COMM_WORLD);
                free(buffer);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    run_end = MPI_Wtime();
    local_stats.total_seconds = run_end - run_start;

    if (rank_output_file != NULL) {
        fclose(rank_output_file);
    }

    BenchmarkStats *all_stats = NULL;
    if (rank == 0) {
        all_stats = (BenchmarkStats *)malloc((size_t)size * sizeof(BenchmarkStats));
        if (all_stats == NULL) {
            fprintf(stderr, "Failed to allocate benchmark gather buffer\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(&local_stats, sizeof(BenchmarkStats), MPI_BYTE,
               all_stats, sizeof(BenchmarkStats), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        write_benchmark_file(all_stats, size, get_total_chunks(), local_stats.total_seconds);
        free(all_stats);
    }

    MPI_Finalize();
    return 0;
}
