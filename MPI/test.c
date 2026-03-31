//mpicc test.c -o julia_mpi -lm
// mpirun -np 4 ./julia_mpi tiles 255 2 0.3 -0.4 10000 10000 1024
// mpirun -np 6 ./julia_mpi tiles 100 2 0 0 1000 1000 1024

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

#define DEFAULT_MAX_ITERATIONS 255
#define THRESHOLD 2.0
#define DEFAULT_WIDTH 100000
#define DEFAULT_HEIGHT 100000
#define DEFAULT_CHUNK_SIZE 4096
#define TAG_WORK 1
#define TAG_RESULT_META 2
#define TAG_STOP 4

static char output_dir[PATH_MAX] = "tiles";
static int max_iterations = DEFAULT_MAX_ITERATIONS;
static double c_re = 0.3;
static double c_im = -0.4;
static int julia_power = 2;
static int width = DEFAULT_WIDTH;
static int height = DEFAULT_HEIGHT;
static int chunk_size = DEFAULT_CHUNK_SIZE;

typedef struct {
    long long chunks_computed;
    double compute_seconds;
    double io_seconds;
    double setup_seconds;
    double total_seconds;
    unsigned long long bytes_written;
} BenchmarkStats;

void parse_args(int argc, char *argv[], int rank);
void ensure_output_dir(int rank);
void write_manifest_file(void);
FILE *open_rank_output_file(int rank);
void write_benchmark_file(const BenchmarkStats *all_stats, int size, int total_chunks, double wall_seconds);


// Computes the Julia value for a given point in the complex plane
int compute_julia_value(double x, double y) {
    double complex z = x + y * I;
    double complex c = c_re + c_im * I;

    for (int i = 0; i < max_iterations; i++) {
        if (cabs(z) > THRESHOLD) {
            return i;
        }

        if (julia_power == 2) {
            z = z * z + c;
        } else if (julia_power == 3) {
            z = z * z * z + c;
        }
    }

    return max_iterations;
}

// Parses the arguments

void parse_args(int argc, char *argv[], int rank) {

    if (argc >= 2) {
        if (snprintf(output_dir, sizeof(output_dir), "%s", argv[1]) >= (int)sizeof(output_dir)) {
            
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (argc >= 3) {
        max_iterations = atoi(argv[2]);
       
    }

    if (argc >= 4) {
        julia_power = atoi(argv[3]);
        if (julia_power != 2 && julia_power != 3) {
            if (rank == 0) {
                fprintf(stderr, "julia_power must be 2 or 3\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }


    if (argc >= 5) {
        c_re = strtod(argv[4], NULL);
    }

    if (argc >= 6) {
        c_im = strtod(argv[5], NULL);
    }

    if (argc >= 7) {
        width = atoi(argv[6]);
        if (width <= 0) {
            if (rank == 0) {
                fprintf(stderr, "width must be positive\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (argc >= 8) {
        height = atoi(argv[7]);
        if (height <= 0) {
            if (rank == 0) {
                fprintf(stderr, "height must be positive\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (argc >= 9) {
        chunk_size = atoi(argv[8]);
        if (chunk_size <= 0) {
            if (rank == 0) {
                fprintf(stderr, "chunk_size must be positive\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }


    if (rank == 0) {
        printf("Julia set run configuration:\n");
        printf("  WIDTH=%d\n", width);
        printf("  HEIGHT=%d\n", height);
        printf("  CHUNK_SIZE=%d\n", chunk_size);
        printf("  max_iterations=%d\n", max_iterations);
        printf("  julia_power=%d\n", julia_power);
        printf("  c_re=%.6f\n", c_re);
        printf("  c_im=%.6f\n", c_im);
        printf("  output_dir=%s\n", output_dir);
    }
}


// Ensures the output directory exists and is writable
// Uses MPI_Barrier to ensure all processes have reached this point 

void ensure_output_dir(int rank) {
    if (rank == 0) {
        if (mkdir(output_dir, 0777) != 0 && errno != EEXIST) {
            perror("Failed to create output directory");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// Writes the manifest file for the Julia set
// The manifest file contains the width, height, chunk size, and maximum iterations
void write_manifest_file(void) {
    char manifest_path[PATH_MAX];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", output_dir);

    FILE *file = fopen(manifest_path, "w");
    if (file == NULL) {
        perror("Failed to open manifest file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fprintf(file, "WIDTH %d\n", width);
    fprintf(file, "HEIGHT %d\n", height);
    fprintf(file, "CHUNK_SIZE %d\n", chunk_size);
    fprintf(file, "MAX_ITERATIONS %d\n", max_iterations);
    fprintf(file, "THRESHOLD %.6f\n", THRESHOLD);
    if (julia_power == 2) {
        fprintf(file, "FUNCTION z^2+c\n");
    } else {
        fprintf(file, "FUNCTION z^3+c\n");
    }
    fprintf(file, "C_RE %.12f\n", c_re);
    fprintf(file, "C_IM %.12f\n", c_im);
    fclose(file);
}

// Opens the output file for the given rank
// The output file is a binary file corresponding to the rank that contains the Julia set
FILE *open_rank_output_file(int rank) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/rank_%d.bin", output_dir, rank);

    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        perror("Failed to open the rank output file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return file;
}


// Gets the number of elements in the chunk
// The number of elements is the width multiplied by the height
size_t get_chunk_element_count(const int *chunk) {
    int width = chunk[2] - chunk[0] + 1;
    int height = chunk[3] - chunk[1] + 1;
    return (size_t)width * (size_t)height;
}

// Writes the chunk to the rank file
// The chunk is a 5 element array that contains the start x, start y, end x, end y, and chunk count
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

// Splits the data into chunks
int *split_data_v3(int chunk_count) {
    int chunks_x = (width + chunk_size - 1) / chunk_size;
    int chunks_y = (height + chunk_size - 1) / chunk_size;
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

    int start_x = chunk_col * chunk_size;
    int start_y = chunk_row * chunk_size;
    int end_x = start_x + chunk_size - 1;
    int end_y = start_y + chunk_size - 1;

    if (end_x >= width) {
        end_x = width - 1;
    }
    if (end_y >= height) {
        end_y = height - 1;
    }

    result[0] = start_x;
    result[1] = start_y;
    result[2] = end_x;
    result[3] = end_y;
    result[4] = chunk_count;
    return result;
}

// Gets the total number of chunks
// The total number of chunks is the number of chunks in the x direction multiplied by the number of chunks in the y direction
int get_total_chunks(void) {
    int chunks_x = (width + chunk_size - 1) / chunk_size;
    int chunks_y = (height + chunk_size - 1) / chunk_size;
    return chunks_x * chunks_y;
}

// Computes the chunk
// For each point in the chunk, computes the Julia value and stores it in the buffer as a uint8_t
void compute_chunk(const int *chunk, uint8_t *buffer) {
    int start_x = chunk[0];
    int start_y = chunk[1];
    int end_x = chunk[2];
    int end_y = chunk[3];
    int chunk_width = end_x - start_x + 1;

    for (int y = start_y; y <= end_y; y++) {
        for (int x = start_x; x <= end_x; x++) {
            double real = -1.5 + (3.0 * x) / (width - 1);
            double imag = -1.5 + (3.0 * y) / (height - 1);
            int value = compute_julia_value(real, imag);
            buffer[(y - start_y) * chunk_width + (x - start_x)] = (uint8_t)value;
        }
    }
}

// Writes the benchmark file
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
    double total_setup = 0.0;
    long long total_chunks_recorded = 0LL;
    double max_rank_total = 0.0;
    double min_rank_total = 0.0;


    for (int i = 0; i < size; i++) {
        total_compute += all_stats[i].compute_seconds;
        total_io += all_stats[i].io_seconds;
        total_setup += all_stats[i].setup_seconds;
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
    fprintf(file, "WIDTH %d\n", width);
    fprintf(file, "HEIGHT %d\n", height);
    fprintf(file, "CHUNK_SIZE %d\n", chunk_size);
    fprintf(file, "MAX_ITERATIONS %d\n", max_iterations);
    fprintf(file, "JULIA_POWER %d\n", julia_power);
    fprintf(file, "C_RE %.12f\n", c_re);
    fprintf(file, "C_IM %.12f\n", c_im);
    fprintf(file, "MPI_PROCESSES %d\n", size);
    fprintf(file, "TOTAL_CHUNKS %d\n", total_chunks);
    fprintf(file, "RECORDED_CHUNKS %lld\n", total_chunks_recorded);
    fprintf(file, "WALL_SECONDS %.6f\n", wall_seconds);
    fprintf(file, "TOTAL_COMPUTE_SECONDS %.6f\n", total_compute);
    fprintf(file, "TOTAL_IO_SECONDS %.6f\n", total_io);
    fprintf(file, "TOTAL_SETUP_SECONDS %.6f\n", total_setup);
    fprintf(file, "MAX_RANK_TOTAL_SECONDS %.6f\n", max_rank_total);
    fprintf(file, "MIN_RANK_TOTAL_SECONDS %.6f\n", min_rank_total);
    fprintf(file, "ESTIMATED_IDLE_SECONDS %.6f\n", max_rank_total * size - total_compute - total_io - total_setup);
    fprintf(file, "\nPer-rank statistics\n");
    fprintf(file, "rank chunks compute_seconds io_seconds setup_seconds total_seconds bytes_written\n");

    for (int i = 0; i < size; i++) {
        fprintf(
            file,
            "%d %lld %.6f %.6f %.6f %.6f %llu\n",
            i,
            all_stats[i].chunks_computed,
            all_stats[i].compute_seconds,
            all_stats[i].io_seconds,
            all_stats[i].setup_seconds,
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

    // Initializes the local stats to 0
    memset(&local_stats, 0, sizeof(local_stats));

    // Initializes the run start and end times to 0
    double run_start = 0.0;
    double run_end = 0.0;

    FILE *rank_output_file = NULL;
    double setup_start = MPI_Wtime();
    parse_args(argc, argv, rank);

    // Ensures the output directory exists and is writable
    ensure_output_dir(rank);

    // Opens the output file for the given rank
    rank_output_file = open_rank_output_file(rank);

    // Waits for all processes to reach this point, at this point, all processes have parsed the arguments and opened the output file they write to
    MPI_Barrier(MPI_COMM_WORLD);
    local_stats.setup_seconds = MPI_Wtime() - setup_start;

    // Starts the timer for the run
    run_start = MPI_Wtime();

    if (rank == 0) {

        // Writes the manifest file, the manifest file contains the width, height, chunk size, maximum iterations, julia power, c_re, c_im, and output directory
        write_manifest_file();

        // Gets the total number of chunks
        int total_chunks = get_total_chunks();

        // Initializes the next chunk to 0, this stores the next chunk that will be sent to the worker
        int next_chunk = 0;
        // Initializes the completed chunks to 0
        int completed_chunks = 0;

        // Sends the work to the workers
        for (int worker = 1; worker < size; worker++) {
            // If there are more chunks to send, sends the next chunk to the worker
            if (next_chunk < total_chunks) {

                int *chunk = split_data_v3(next_chunk);

                MPI_Send(chunk, 5, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);

                next_chunk++;

                free(chunk);

            } else {
                // Else their is no more work to send, sends the stop signal to the worker
                MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
            }
        }

        // This is for running the program with only one process, if there are no workers, rank 0 must compute everything locally.
        if (size == 1) {
            while (next_chunk < total_chunks) {
                int *chunk = split_data_v3(next_chunk);
                size_t element_count = get_chunk_element_count(chunk);

                uint8_t *buffer = (uint8_t *)malloc(element_count * sizeof(uint8_t));
                if (buffer == NULL) {
                    printf("Error: Failed to allocate memory for rank 0 chunk buffer\n");
                    free(chunk);
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
                completed_chunks++;
                next_chunk++;

                free(buffer);
                free(chunk);
            }
        } else {
            
            // This is for running the program with multiple processes (This is still rank 0), rank 0 sends the work to the workers and the workers send the results back to rank 0
            while (completed_chunks < total_chunks) {

                int finished_chunk[5];

                MPI_Status recv_status;

                // Receives the result from the worker
                MPI_Recv(finished_chunk, 5, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT_META, MPI_COMM_WORLD, &recv_status);

                // Updates the completed chunk count
                completed_chunks++;

                // If there are more chunks to send, sends the next chunk to the worker
                if (next_chunk < total_chunks) {

                    int *next = split_data_v3(next_chunk);

                    MPI_Send(next, 5, MPI_INT, recv_status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);

                    next_chunk++;
                    free(next);

                } else {
                    MPI_Send(NULL, 0, MPI_INT, recv_status.MPI_SOURCE, TAG_STOP, MPI_COMM_WORLD);
                }
            }
        }

    // If the rank is not 0, processes the chunks until the stop signal is received
    } else {
        // Processes the chunks until the stop signal is received
        while (1) {
           
            MPI_Status status;
            // Probes for any results from the master
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            // If the stop signal is received, breaks the loop
            if (status.MPI_TAG == TAG_STOP) {
                // Receives the stop signal from the master
                MPI_Recv(NULL, 0, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            }
            // If the work signal is received, receives the chunk from the master
            if (status.MPI_TAG == TAG_WORK) {

                int chunk[5];

                // Receives the chunk from the master
                MPI_Recv(chunk, 5, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Gets the number of elements in the chunk
                size_t element_count = get_chunk_element_count(chunk);

                // Allocates memory for the buffer
                uint8_t *buffer = (uint8_t *)malloc(element_count * sizeof(uint8_t));
                if (buffer == NULL) {
                    printf("Error: Failed to allocate memory for worker chunk buffer\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                // Starts the timer for the compute
                double compute_start = MPI_Wtime();

                // Computes the chunk
                compute_chunk(chunk, buffer);
                local_stats.compute_seconds += MPI_Wtime() - compute_start;

                // Starts the timer for the IO
                double io_start = MPI_Wtime();

                // Writes the chunk to the rank file
                write_chunk_to_rank_file(rank_output_file, chunk, buffer);
                local_stats.io_seconds += MPI_Wtime() - io_start;

                // Updates the local stats
                local_stats.chunks_computed++;

                local_stats.bytes_written += (unsigned long long)(5 * sizeof(int) + element_count * sizeof(uint8_t));

                // Sends the chunk to the master
                MPI_Send(chunk, 5, MPI_INT, 0, TAG_RESULT_META, MPI_COMM_WORLD);
                free(buffer);
            }
        }
    }

    // Waits for all processes to reach this point
    MPI_Barrier(MPI_COMM_WORLD);

    // Stops the timer for the run
    run_end = MPI_Wtime();

    // Updates the local stats for the total time
    local_stats.total_seconds = run_end - run_start;

    // Closes the rank output file if it is open
    if (rank_output_file != NULL) {
        fclose(rank_output_file);
    }

    // Allocates memory for the benchmark stats
    BenchmarkStats *all_stats = NULL;
    // If the rank is 0, allocates memory for the benchmark stats
    if (rank == 0) {
        all_stats = (BenchmarkStats *)malloc((size_t)size * sizeof(BenchmarkStats));
        // If the allocation fails, aborts the program
        if (all_stats == NULL) {
            fprintf(stderr, "Failed to allocate benchmark gather buffer\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    printf("[rank %d] setup_seconds=%.6f io_seconds=%.6f compute_seconds=%.6f total_seconds=%.6f\n",
           rank,
           local_stats.setup_seconds,
           local_stats.io_seconds,
           local_stats.compute_seconds,
           local_stats.total_seconds);
    // Gathers the local stats from the other processes.
    // Rank 0 may act only as a scheduler for multi-process runs, so its chunk count can be 0 by design.
    MPI_Gather(&local_stats, sizeof(BenchmarkStats), MPI_BYTE,
               all_stats, sizeof(BenchmarkStats), MPI_BYTE,
               0, MPI_COMM_WORLD);

    // If the rank is 0, writes the benchmark file
    if (rank == 0) {
        write_benchmark_file(all_stats, size, get_total_chunks(), local_stats.total_seconds);
        free(all_stats);
    }

    MPI_Finalize();
    return 0;
}
