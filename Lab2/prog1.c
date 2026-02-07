#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int nproc, rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nproc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: prog1 requires at least 2 MPI processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int tag = 0;

    if (rank == 0) {
        int my_id = rank;
        printf("[rank %d] Sending my id (%d) to rank 1...\n", rank, my_id);
        MPI_Send(&my_id, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
        printf("[rank %d] Send complete.\n", rank);
    } else if (rank == 1) {
        int received_id = -1;
        MPI_Recv(&received_id, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[rank %d] Received id %d from rank 0.\n", rank, received_id);
    } else {
        // Other ranks do nothing (still participate in MPI_Init/Finalize)
        printf("[rank %d] Idle (not participating in send/recv).\n", rank);
    }

    MPI_Finalize();
    return 0;
}

