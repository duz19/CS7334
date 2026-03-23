#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {
    TAG_WORK = 1,
    TAG_ACK = 2,
    TAG_REQUEST = 3,
    TAG_ABORT = 4,
    TAG_NO_WORK = 5
};

typedef struct {
    int producer_rank;
    int value;
} WorkItem;

typedef struct {
    WorkItem *data;
    int capacity;
    int head;
    int size;
} WorkQueue;

static void queue_init(WorkQueue *queue, int capacity) {
    queue->data = capacity > 0 ? (WorkItem *)malloc((size_t)capacity * sizeof(WorkItem)) : NULL;
    queue->capacity = capacity;
    queue->head = 0;
    queue->size = 0;
}

static void queue_free(WorkQueue *queue) {
    free(queue->data);
    queue->data = NULL;
    queue->capacity = 0;
    queue->head = 0;
    queue->size = 0;
}

static int queue_empty(const WorkQueue *queue) {
    return queue->size == 0;
}

static int queue_full(const WorkQueue *queue) {
    return queue->size == queue->capacity;
}

static int queue_push(WorkQueue *queue, WorkItem item) {
    int tail;

    if (queue_full(queue)) {
        return 0;
    }

    tail = (queue->head + queue->size) % queue->capacity;
    queue->data[tail] = item;
    queue->size++;
    return 1;
}

static WorkItem queue_pop(WorkQueue *queue) {
    WorkItem item = queue->data[queue->head];

    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    return item;
}

static int is_producer_rank(int rank, int producer_count) {
    return rank >= 1 && rank <= producer_count;
}

static int expensive_compute(int seed) {
    double value = (double)(seed % 1000 + 1);
    int i;

    for (i = 0; i < 1000; i++) {
        value = value * 1.000001 + (double)((seed + i) % 97);
        value = value / 1.0000003;
    }

    return (int)value;
}

static void run_producer(int rank, int seconds) {
    unsigned int rng = (unsigned int)(time(NULL) ^ (rank * 1103515245u));
    int keep_running = 1;

    (void)seconds;

    while (keep_running) {
        int work_value = (int)(rand_r(&rng) % 1000000);
        int response = TAG_ACK;
        MPI_Request send_request;

        MPI_Isend(&work_value, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, &send_request);
        (void)expensive_compute(work_value);
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        MPI_Recv(&response, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (response == TAG_ABORT) {
            keep_running = 0;
        }
    }
}

static void run_consumer(int rank, int seconds, long long *consumed_count) {
    unsigned int rng = (unsigned int)(time(NULL) ^ (rank * 2654435761u));
    int keep_running = 1;
    int previous_value = (int)(rand_r(&rng) % 1000000);

    (void)seconds;

    while (keep_running) {
        int request_token = rank;
        int received_value = 0;
        MPI_Status status;
        MPI_Request send_request;

        MPI_Isend(&request_token, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD, &send_request);
        (void)expensive_compute(previous_value);
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        MPI_Recv(&received_value, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_WORK) {
            *consumed_count += 1;
            previous_value = received_value;
        } else if (status.MPI_TAG == TAG_ABORT) {
            keep_running = 0;
        }
    }
}

static void move_outstanding_to_buffer(WorkQueue *outstanding, WorkQueue *buffer, int *aborted) {
    while (!queue_empty(outstanding) && !queue_full(buffer)) {
        WorkItem item = queue_pop(outstanding);
        int ack_value = TAG_ACK;

        queue_push(buffer, item);
        MPI_Send(&ack_value, 1, MPI_INT, item.producer_rank, TAG_ACK, MPI_COMM_WORLD);
        aborted[item.producer_rank] = 0;
    }
}

static void abort_outstanding_producers(WorkQueue *outstanding, int *aborted) {
    while (!queue_empty(outstanding)) {
        WorkItem item = queue_pop(outstanding);
        int abort_value = TAG_ABORT;

        MPI_Send(&abort_value, 1, MPI_INT, item.producer_rank, TAG_ABORT, MPI_COMM_WORLD);
        aborted[item.producer_rank] = 1;
    }
}

static int all_nonzero_ranks_aborted(const int *aborted, int world_size) {
    int rank;

    for (rank = 1; rank < world_size; rank++) {
        if (!aborted[rank]) {
            return 0;
        }
    }

    return 1;
}

static void run_broker(int world_size, int seconds) {
    int producer_count = (world_size - 1) / 2;
    int buffer_capacity = producer_count > 0 ? producer_count * 2 : 1;
    WorkQueue buffer;
    WorkQueue outstanding;
    double start_time = MPI_Wtime();
    int *aborted = (int *)calloc((size_t)world_size, sizeof(int));
    int abort_value = TAG_ABORT;
    int accept_only_producer_work = 0;
    int timer_expired = 0;

    queue_init(&buffer, buffer_capacity);
    queue_init(&outstanding, producer_count > 0 ? producer_count : 1);

    while (!timer_expired) {
        MPI_Status status;
        int payload = 0;
        int response_type = -1;

        if (accept_only_producer_work) {
            MPI_Recv(&payload, 1, MPI_INT, MPI_ANY_SOURCE, TAG_WORK, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(&payload, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }

        if (MPI_Wtime() - start_time >= (double)seconds) {
            response_type = TAG_ABORT;
            timer_expired = 1;
        }

        if (status.MPI_TAG == TAG_WORK && is_producer_rank(status.MPI_SOURCE, producer_count)) {
            WorkItem item;

            item.producer_rank = status.MPI_SOURCE;
            item.value = payload;
            accept_only_producer_work = 0;

            if (!queue_full(&buffer)) {
                if (response_type == TAG_ABORT) {
                    MPI_Send(&abort_value, 1, MPI_INT, status.MPI_SOURCE, TAG_ABORT, MPI_COMM_WORLD);
                    aborted[status.MPI_SOURCE] = 1;
                } else {
                    int ack_value = TAG_ACK;

                    queue_push(&buffer, item);
                    MPI_Send(&ack_value, 1, MPI_INT, status.MPI_SOURCE, TAG_ACK, MPI_COMM_WORLD);
                }
            } else {
                if (response_type == TAG_ABORT) {
                    MPI_Send(&abort_value, 1, MPI_INT, status.MPI_SOURCE, TAG_ABORT, MPI_COMM_WORLD);
                    aborted[status.MPI_SOURCE] = 1;
                } else {
                    queue_push(&outstanding, item);
                }
            }
        } else if (status.MPI_TAG == TAG_REQUEST && !is_producer_rank(status.MPI_SOURCE, producer_count)) {
            if (response_type == TAG_ABORT) {
                MPI_Send(&abort_value, 1, MPI_INT, status.MPI_SOURCE, TAG_ABORT, MPI_COMM_WORLD);
                aborted[status.MPI_SOURCE] = 1;
            } else if (!queue_empty(&buffer)) {
                WorkItem item = queue_pop(&buffer);

                MPI_Send(&item.value, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                move_outstanding_to_buffer(&outstanding, &buffer, aborted);
            } else {
                int no_work_value = 0;

                MPI_Send(&no_work_value, 1, MPI_INT, status.MPI_SOURCE, TAG_NO_WORK, MPI_COMM_WORLD);
                accept_only_producer_work = 1;
            }
        }
    }

    abort_outstanding_producers(&outstanding, aborted);

    while (!all_nonzero_ranks_aborted(aborted, world_size)) {
        MPI_Status status;
        int payload = 0;

        MPI_Recv(&payload, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (!aborted[status.MPI_SOURCE]) {
            MPI_Send(&abort_value, 1, MPI_INT, status.MPI_SOURCE, TAG_ABORT, MPI_COMM_WORLD);
            aborted[status.MPI_SOURCE] = 1;
        }
    }

    queue_free(&buffer);
    queue_free(&outstanding);
    free(aborted);
}

int main(int argc, char **argv) {
    int rank;
    int world_size;
    int seconds;
    long long local_consumed = 0;
    long long total_consumed = 0;

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
    if (seconds <= 0 || world_size < 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage requires positive seconds and at least 3 MPI processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        run_broker(world_size, seconds);
    } else if (is_producer_rank(rank, (world_size - 1) / 2)) {
        run_producer(rank, seconds);
    } else {
        run_consumer(rank, seconds, &local_consumed);
    }

    MPI_Reduce(&local_consumed, &total_consumed, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total number of messages consumed: %lld\n", total_consumed);
    }

    MPI_Finalize();
    return 0;
}

