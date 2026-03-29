#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <string.h>


// julia sets 
#include <complex.h>
#define MAX_ITERATIONS 100
#define THRESHOLD 2.0
#define WIDTH 1000
#define HEIGHT 1000
#define CHUNK_SIZE 10
#define TAG_WORK 1
#define TAG_RESULT_META 2
#define TAG_RESULT_DATA 3
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

void write_julia_set_to_file(int *julia_set, int width, int height) {
    FILE *file = fopen("julia_set.bin", "wb");
    if (file == NULL) {
        perror("Failed to open julia_set.bin");
        return;
    }

    fwrite(&width, sizeof(int), 1, file);
    fwrite(&height, sizeof(int), 1, file);
    fwrite(julia_set, sizeof(int), width * height, file);
    fclose(file);
}



//split data into rows 
int *split_data_v1(int width, int height, int rank, int size) {
    int base_rows = height / size;
    int extra = height % size;

    int local_rows, start_row;

    if (rank < extra) {
        local_rows = base_rows + 1;
        start_row = rank * local_rows;
    } else {
        local_rows = base_rows;
        start_row = rank * base_rows + extra;
    }

    int end_row = start_row + local_rows;

    int *result = (int *)malloc(2 * sizeof(int));
    if (result == NULL) {
        printf("Error: Failed to allocate memory for result\n");
        MPI_Finalize();
        return NULL;
    }

    result[0] = start_row;
    result[1] = end_row;
    return result;
}



int *split_data_v2(int width, int height, int rank, int size) {
    int chunks_x = floor((width + CHUNK_SIZE - 1) / CHUNK_SIZE);  
    int chunks_y = floor((height + CHUNK_SIZE - 1) / CHUNK_SIZE); 
    int extra_x = width % CHUNK_SIZE;
    int extra_y = height % CHUNK_SIZE;
    int total_chunks = chunks_x * chunks_y;

    // If there are more processors than chunks, some ranks get nothing
    if (rank >= total_chunks || rank >= size) {
        return NULL;
    }

    int *result = (int *)malloc(4 * sizeof(int));
    if (result == NULL) {
        return NULL;
    }

    // Map rank to 2D chunk index
    int chunk_row = rank / chunks_x;
    int chunk_col = rank % chunks_x;

    int start_x = chunk_col * CHUNK_SIZE;
    int start_y = chunk_row * CHUNK_SIZE;

    // Handle edge chunks
    int chunk_width = CHUNK_SIZE;
    int chunk_height = CHUNK_SIZE;

    if (start_x + chunk_width > width) {
        chunk_width = width - start_x;
    }

    if (start_y + chunk_height > height) {
        chunk_height = height - start_y;
    }

    result[0] = start_x;
    result[1] = start_y;
    result[2] = chunk_width;
    result[3] = chunk_height;

    return result;
}

int *split_data_v3(int chunk_count) {
    int chunks_x = (WIDTH + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int chunks_y = (HEIGHT + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int total_chunks = chunks_x * chunks_y;

    if (chunk_count < 0 || chunk_count >= total_chunks) {
        return NULL;
    }

    int *result = (int *)malloc(4 * sizeof(int));
    if (result == NULL) {
        printf("Error: Failed to allocate memory for result\n");
        MPI_Finalize();
        exit(1);
    }

    /*
      Chunks are assigned in row-major order across the matrix.
      Example for CHUNK_SIZE=2:
      chunk 0 -> top-left chunk
      chunk 1 -> next chunk to the right
      chunk 2 -> next chunk to the right, etc.
      Once a process finishes, the master can increment chunk_count and hand
      out the next available chunk.
    */
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
    printf("Chunk %d: from (%d, %d) to (%d, %d)\n", chunk_count, start_x, start_y, end_x, end_y);

    return result;
}

int get_total_chunks(void) {
    int chunks_x = (WIDTH + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int chunks_y = (HEIGHT + CHUNK_SIZE - 1) / CHUNK_SIZE;
    return chunks_x * chunks_y;
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

void compute_chunk(const int *chunk, int *buffer) {
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
            buffer[(y - start_y) * chunk_width + (x - start_x)] = value;
        }
    }
}

void store_chunk_in_matrix(int *julia_set, const int *chunk, const int *buffer) {
    int start_x = chunk[0];
    int start_y = chunk[1];
    int end_x = chunk[2];
    int end_y = chunk[3];
    int chunk_width = end_x - start_x + 1;

    for (int y = start_y; y <= end_y; y++) {
        for (int x = start_x; x <= end_x; x++) {
            julia_set[y * WIDTH + x] = buffer[(y - start_y) * chunk_width + (x - start_x)];
        }
    }
}

void print_julia_set(int *julia_set, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%d ", julia_set[y * width + x]);
        }
        printf("\n");
    }
}

void print_local_julia_set(int *local_julia_set, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%d ", local_julia_set[y * width + x]);
        }
        printf("\n");
    }
}
int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        int total_chunks = get_total_chunks();
        int next_chunk = 0;
        int completed_chunks = 0;

        int *julia_set = (int *)malloc(WIDTH * HEIGHT * sizeof(int));
        if (julia_set == NULL) {
            printf("Error: Failed to allocate memory for julia_set\n");
            MPI_Finalize();
            return 1;
        }

        /* Send one initial chunk to each worker if work remains. */
        for (int worker = 1; worker < size; worker++) {
            if (next_chunk < total_chunks) {
                int *chunk = split_data_v3(next_chunk);
                MPI_Send(chunk, 4, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                next_chunk++;
                free(chunk);
            } else {
                MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
            }
        }

        while (completed_chunks < total_chunks) {
            /* Rank 0 also computes the next available chunk if there is one. */
            if (next_chunk < total_chunks) {
                int *chunk = split_data_v3(next_chunk);
                int element_count = get_chunk_element_count(chunk);
                int *buffer = (int *)malloc(element_count * sizeof(int));
                if (buffer == NULL) {
                    printf("Error: Failed to allocate memory for rank 0 chunk buffer\n");
                    free(chunk);
                    free(julia_set);
                    MPI_Finalize();
                    return 1;
                }

                next_chunk++;
                compute_chunk(chunk, buffer);
                store_chunk_in_matrix(julia_set, chunk, buffer);
                completed_chunks++;

                free(buffer);
                free(chunk);
            }

            /* Process any completed worker results that are waiting. */
            int flag = 0;
            MPI_Status status;
            do {
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT_META, MPI_COMM_WORLD, &flag, &status);
                if (flag) {
                    int worker = status.MPI_SOURCE;
                    int chunk[4];
                    MPI_Recv(chunk, 4, MPI_INT, worker, TAG_RESULT_META, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    int element_count = get_chunk_element_count(chunk);
                    int *buffer = (int *)malloc(element_count * sizeof(int));
                    if (buffer == NULL) {
                        printf("Error: Failed to allocate memory for recv buffer\n");
                        free(julia_set);
                        MPI_Finalize();
                        return 1;
                    }

                    MPI_Recv(buffer, element_count, MPI_INT, worker, TAG_RESULT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    store_chunk_in_matrix(julia_set, chunk, buffer);
                    completed_chunks++;
                    free(buffer);

                    if (next_chunk < total_chunks) {
                        int *next = split_data_v3(next_chunk);
                        MPI_Send(next, 4, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                        next_chunk++;
                        free(next);
                    } else {
                        MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
                    }
                }
            } while (flag);

            /* If rank 0 has no more local chunks to compute, block until workers finish. */
            if (next_chunk >= total_chunks && completed_chunks < total_chunks) {
                int worker_chunk[4];
                MPI_Status recv_status;
                MPI_Recv(worker_chunk, 4, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT_META, MPI_COMM_WORLD, &recv_status);
                int worker = recv_status.MPI_SOURCE;

                int element_count = get_chunk_element_count(worker_chunk);
                int *buffer = (int *)malloc(element_count * sizeof(int));
                if (buffer == NULL) {
                    printf("Error: Failed to allocate memory for blocking recv buffer\n");
                    free(julia_set);
                    MPI_Finalize();
                    return 1;
                }

                MPI_Recv(buffer, element_count, MPI_INT, worker, TAG_RESULT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                store_chunk_in_matrix(julia_set, worker_chunk, buffer);
                completed_chunks++;
                free(buffer);

                MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
            }
        }

        write_julia_set_to_file(julia_set, WIDTH, HEIGHT);
        free(julia_set);
    } else {
        while (1) {
            MPI_Status status;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP) {
                MPI_Recv(NULL, 0, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            }

            if (status.MPI_TAG == TAG_WORK) {
                int chunk[4];
                MPI_Recv(chunk, 4, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int element_count = get_chunk_element_count(chunk);
                int *buffer = (int *)malloc(element_count * sizeof(int));
                if (buffer == NULL) {
                    printf("Error: Failed to allocate memory for worker chunk buffer\n");
                    MPI_Finalize();
                    return 1;
                }

                compute_chunk(chunk, buffer);
                MPI_Send(chunk, 4, MPI_INT, 0, TAG_RESULT_META, MPI_COMM_WORLD);
                MPI_Send(buffer, element_count, MPI_INT, 0, TAG_RESULT_DATA, MPI_COMM_WORLD);
                free(buffer);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
