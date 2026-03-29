#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <sys/stat.h>
#include <errno.h>

#define MAX_ITERATIONS 100
#define THRESHOLD 2.0
#define WIDTH 100000
#define HEIGHT 100000
#define CHUNK_SIZE 2048
#define TAG_WORK 1
#define TAG_RESULT_META 2
#define TAG_STOP 4

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



void ensure_output_dir(void) {
    if (mkdir("tiles", 0777) != 0 && errno != EEXIST) {
        perror("Failed to create tiles directory");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void write_manifest_file(void) {
    FILE *file = fopen("tiles/manifest.txt", "w");
    if (file == NULL) {
        perror("Failed to open tiles/manifest.txt");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fprintf(file, "WIDTH %d\n", WIDTH);
    fprintf(file, "HEIGHT %d\n", HEIGHT);
    fprintf(file, "CHUNK_SIZE %d\n", CHUNK_SIZE);
    fprintf(file, "MAX_ITERATIONS %d\n", MAX_ITERATIONS);
    fclose(file);
}

FILE *open_rank_output_file(int rank) {
    char path[256];
    snprintf(path, sizeof(path), "tiles/rank_%d.bin", rank);

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

int get_chunk_element_count(const int *chunk) {
    return get_chunk_width(chunk) * get_chunk_height(chunk);
}


void write_chunk_to_rank_file(FILE *file, const int *chunk, const uint8_t *buffer) {
    if (fwrite(chunk, sizeof(int), 5, file) != 5) {
        perror("Failed to write chunk header to rank file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (fwrite(buffer, sizeof(uint8_t), get_chunk_element_count(chunk), file) != (size_t)get_chunk_element_count(chunk)) {
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

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    FILE *rank_output_file = NULL;

    ensure_output_dir();
    rank_output_file = open_rank_output_file(rank);

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
                int element_count = get_chunk_element_count(chunk);
                uint8_t *buffer = (uint8_t *)malloc((size_t)element_count * sizeof(uint8_t));
                if (buffer == NULL) {
                    printf("Error: Failed to allocate memory for rank 0 chunk buffer\n");
                    free(chunk);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                next_chunk++;
                compute_chunk(chunk, buffer);
                write_chunk_to_rank_file(rank_output_file, chunk, buffer);
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

                int element_count = get_chunk_element_count(chunk);
                uint8_t *buffer = (uint8_t *)malloc((size_t)element_count * sizeof(uint8_t));
                if (buffer == NULL) {
                    printf("Error: Failed to allocate memory for worker chunk buffer\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                compute_chunk(chunk, buffer);
                write_chunk_to_rank_file(rank_output_file, chunk, buffer);
                MPI_Send(chunk, 5, MPI_INT, 0, TAG_RESULT_META, MPI_COMM_WORLD);
                free(buffer);
            }
        }
    }

    if (rank_output_file != NULL) {
        fclose(rank_output_file);
    }
    MPI_Finalize();
    return 0;
}
