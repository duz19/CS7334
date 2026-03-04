#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"

#define MAXPROC 100 /* Max number of processes */

int main(int argc, char* argv[]) {
    int i, nproc, rank, index;
    const int tag = 42;   /* Tag value for communication */
    const int root = 0;   /* Root process in broadcast */

    MPI_Status status;                 /* Status object for non-blocking receive */
    MPI_Request recv_req[MAXPROC];     /* Request objects for non-blocking receive */

    char hostname[MAXPROC][MPI_MAX_PROCESSOR_NAME]; /* buffers to store received hostnames */
    char myname[MPI_MAX_PROCESSOR_NAME];            /* local host name string */
    int namelen = 0;                                  /* Length of the name */

    /* Init MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (nproc > MAXPROC) {
        if (rank == 0) {
            fprintf(stderr, "Error: nproc (%d) exceeds MAXPROC (%d)\n", nproc, MAXPROC);
        }
        MPI_Finalize();
        return 1;
    }

    /* Get local hostname */
    MPI_Get_processor_name(myname, &namelen);

    /* MPI_Get_processor_name may not null-terminate */
    if (namelen >= MPI_MAX_PROCESSOR_NAME) namelen = MPI_MAX_PROCESSOR_NAME - 1;
    myname[namelen] = '\0';

    /* Also pre-clear receive buffers (optional, helps clean output) */
    for (i = 0; i < MAXPROC; i++) {
        hostname[i][0] = '\0';
    }

    if (rank == 0) {
        /* Broadcast a message containing the process id (rank 0's id) */
        int pid = rank; /* 0 */
        MPI_Bcast(&pid, 1, MPI_INT, root, MPI_COMM_WORLD);

        /* Start non-blocking calls to receive messages from all other processes.
           Note: recv_req[1..nproc-1] correspond to ranks 1..nproc-1. */
        for (i = 1; i < nproc; i++) {
            MPI_Irecv(hostname[i], MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                      MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_req[i]);
        }

        /* Do other computation while messages are in flight */
        printf("I am a very busy professor.\n");

        /* Receive messages in the order they arrive */
        for (i = 1; i < nproc; i++) {
            /* Wait for ANY of the outstanding requests (exclude index 0) */
            MPI_Waitany(nproc - 1, &recv_req[1], &index, &status);

            /* index is relative to &recv_req[1], so actual request slot is index+1 */
            int slot = index + 1;

            /* Make sure received hostname is null-terminated */
            hostname[slot][MPI_MAX_PROCESSOR_NAME - 1] = '\0';

            printf("Received a message from process %d on %s\n",
                   status.MPI_SOURCE, hostname[slot]);
        }
    } else {
        /* Receive the broadcasted message from process 0 */
        int pid = -1;
        MPI_Bcast(&pid, 1, MPI_INT, root, MPI_COMM_WORLD);

        /* Send local hostname to process 0 */
        MPI_Send(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
