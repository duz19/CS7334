#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {
    TAG_WORK = 11,
    TAG_ACK = 12
};

static int expensive_compute(int seed) {
    double value = (double)(seed % 1000 + 1);
    int i;

    for (i = 0; i < 1000; i++) {
        value = value * 1.000001 + (double)((seed + i) % 89);
        value = value / 1.0000003;
    }

    return (int)value;
}

static int choose_destination(unsigned int *rng, int rank, int world_size) {
    int destination;

    if (world_size == 1) {
        return 0;
    }

    destination = (int)(rand_r(rng) % (unsigned int)(world_size - 1));
    if (destination >= rank) {
        destination++;
    }

    return destination;
}

static void progress_messages(long long *consumed_count, int *ack_received) {
    int has_message = 0;
    MPI_Status status;

    do {
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &has_message, &status);
        if (has_message) {
            if (status.MPI_TAG == TAG_WORK) {
                int received_value = 0;
                int ack_value = 1;
                MPI_Request ack_request;

                MPI_Recv(&received_value, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                (void)expensive_compute(received_value);
                *consumed_count += 1;
                MPI_Isend(&ack_value, 1, MPI_INT, status.MPI_SOURCE, TAG_ACK, MPI_COMM_WORLD, &ack_request);
                MPI_Wait(&ack_request, MPI_STATUS_IGNORE);
            } else if (status.MPI_TAG == TAG_ACK) {
                int ack_value = 0;

                MPI_Recv(&ack_value, 1, MPI_INT, status.MPI_SOURCE, TAG_ACK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                *ack_received = 1;
            }
        }
    } while (has_message);
}

int main(int argc, char **argv) {
    int rank;
    int world_size;
    int seconds;
    long long local_consumed = 0;
    long long total_consumed = 0;
    double start_time;
    unsigned int rng;
    int send_active = 0;
    int ack_received = 0;
    MPI_Request send_request = MPI_REQUEST_NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <seconds>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    seconds = atoi(argv[1]);
    if (seconds <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Usage requires positive seconds.\n");
        }
        MPI_Finalize();
        return 1;
    }

    rng = (unsigned int)(time(NULL) ^ (rank * 747796405u));
    start_time = MPI_Wtime();

    for (;;) {
        int local_has_message = 0;
        int global_done = 0;
        int local_done;

        if (!send_active && MPI_Wtime() - start_time < (double)seconds) {
            int work_value = (int)(rand_r(&rng) % 1000000);
            int destination = choose_destination(&rng, rank, world_size);

            MPI_Isend(&work_value, 1, MPI_INT, destination, TAG_WORK, MPI_COMM_WORLD, &send_request);
            send_active = 1;
            ack_received = 0;
        }

        progress_messages(&local_consumed, &ack_received);

        if (send_active && ack_received) {
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            send_request = MPI_REQUEST_NULL;
            send_active = 0;
            ack_received = 0;
        }

        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &local_has_message, MPI_STATUS_IGNORE);

        local_done = (MPI_Wtime() - start_time >= (double)seconds) && !send_active && !local_has_message;
        MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (global_done) {
            break;
        }
    }

    MPI_Reduce(&local_consumed, &total_consumed, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total number of messages consumed: %lld\n", total_consumed);
    }

    MPI_Finalize();
    return 0;
}

