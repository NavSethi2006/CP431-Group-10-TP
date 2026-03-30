

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
#define WIDTH 100
#define HEIGHT 100
#define CHUNK_SIZE 2



int compute_julia_value(double x, double y) {
    double complex z = x + y * I;
    double complex c = 0.285 + 0.01 * I;
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

//test the split data into chunks
void test_split_data_v2(int size) {
   //split the julia set and send the chunks to the other processors
        int *result = split_data_v2(WIDTH, HEIGHT, 0, size);
        int start_x = result[0];
        int start_y = result[1];
        int chunk_width = result[2];
        int chunk_height = result[3];
        free(result);

        // Send the chunks to the other processors
        for (int i = 0; i < size; i++) {
            start_x = i * chunk_width;
            start_y = i * chunk_height;
            MPI_Send(&start_x, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&start_y, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&chunk_width, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&chunk_height, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
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

    int *julia_set = (int *)malloc(WIDTH * HEIGHT * sizeof(int));
    if (julia_set == NULL) {
        printf("Error: Failed to allocate memory for julia_set\n");
        MPI_Finalize();
        return 1;
    }

    int *result = split_data_v1(WIDTH, HEIGHT, rank, size);
    int start_row = result[0];
    int end_row = result[1];
    // printf("Processor %d received rows from %d to %d\n", rank, start_row, end_row);
    free(result);

    int *local_julia_set = (int *)malloc((end_row - start_row) * WIDTH * sizeof(int));
   
    if (local_julia_set == NULL) {
        printf("Error: Failed to allocate memory for local_julia_set\n");
        MPI_Finalize();
        return 1;
    }

    // compute the julia set
    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = -1.5 + (3.0 * x) / (WIDTH - 1);
            double imag = -1.5 + (3.0 * y) / (HEIGHT - 1);
            int value = compute_julia_value(real, imag);
            // printf("Processor %d computed value %d at (%d, %d)\n", rank, value, x, y);
            //send the value to the master processor
            local_julia_set[(y - start_row) * WIDTH + x] = value;
        }
    }
    // print the local julia set
    // print_local_julia_set(local_julia_set, WIDTH, end_row - start_row);

    if (rank == 0) {
        // copy rank 0's own local results directly into the full image
        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < WIDTH; x++) {
                julia_set[y * WIDTH + x] = local_julia_set[(y - start_row) * WIDTH + x];
            }
        }

        // receive the julia set chunks from the other processors
        for (int i = 1; i < size; i++) {
            int recv_start_row;
            int recv_end_row;

            MPI_Recv(&recv_start_row, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&recv_end_row, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int recv_rows = recv_end_row - recv_start_row;
            int *recv_buffer = (int *)malloc(recv_rows * WIDTH * sizeof(int));
            if (recv_buffer == NULL) {
                printf("Error: Failed to allocate memory for recv_buffer\n");
                free(julia_set);
                free(local_julia_set);
                MPI_Finalize();
                return 1;
            }

            MPI_Recv(recv_buffer, recv_rows * WIDTH, MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int y = recv_start_row; y < recv_end_row; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    julia_set[y * WIDTH + x] = recv_buffer[(y - recv_start_row) * WIDTH + x];
                }
            }

            free(recv_buffer);
        }

        print_julia_set(julia_set, WIDTH, HEIGHT);
        write_julia_set_to_file(julia_set, WIDTH, HEIGHT);
    } else {
        // send metadata first, then the local chunk data
        MPI_Send(&start_row, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&end_row, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(local_julia_set, (end_row - start_row) * WIDTH, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }

    free(julia_set);
    free(local_julia_set);
    MPI_Finalize();
    return 0;}

