#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int nproc, rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nproc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: prog2 (ring) requires at least 2 MPI processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int next = (rank + 1) % nproc;
    int prev = (rank - 1 + nproc) % nproc;
    int tag = 0;

    int value = -1;

    if (rank == 0) {
        // Create a random number
        unsigned int seed = (unsigned int)time(NULL);
        value = (int)(rand_r(&seed) % 100000);  // 0..99999
        printf("[rank %d] Generated random number: %d\n", rank, value);

        // Send to rank 1
        printf("[rank %d] Sending %d to rank %d\n", rank, value, next);
        MPI_Send(&value, 1, MPI_INT, next, tag, MPI_COMM_WORLD);

        // Receive it back from the last rank
        int returned = -1;
        MPI_Recv(&returned, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[rank %d] Received %d back from rank %d (ring complete)\n", rank, returned, prev);
    } else {
        // Receive from previous rank
        MPI_Recv(&value, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[rank %d] Received %d from rank %d\n", rank, value, prev);

        // Send to next rank
        printf("[rank %d] Sending %d to rank %d\n", rank, value, next);
        MPI_Send(&value, 1, MPI_INT, next, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

