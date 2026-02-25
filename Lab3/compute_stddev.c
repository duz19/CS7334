#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
static float *rand_num_gen(int num_elements) {
    float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
    assert(rand_nums != NULL);

    for (int i = 0; i < num_elements; i++) {
        rand_nums[i] = (rand() / (float)RAND_MAX);
    }
    return rand_nums;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int nproc = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s num_elements_per_proc\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int num_elements_per_proc = atoi(argv[1]);
    if (num_elements_per_proc <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: num_elements_per_proc must be > 0\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Seed uniquely per process (template had a bug using nproc before it was set)
    srand(123456 * (rank + 1));

    // Create a random array of elements on all processes
    float *rand_nums = rand_num_gen(num_elements_per_proc);

    // 1) Compute local sum
    double local_sum = 0.0;
    for (int i = 0; i < num_elements_per_proc; i++) {
        local_sum += (double)rand_nums[i];
    }

    // 2) Compute global sum on ALL processes to get the mean everywhere
    // Need mean on every process to compute local squared diffs, so MPI_Allreduce fits.
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const long long global_count = (long long)nproc * (long long)num_elements_per_proc;
    const double mean = global_sum / (double)global_count;

    // 3) Compute local sum of squared differences from the mean
    double local_sq_diff_sum = 0.0;
    for (int i = 0; i < num_elements_per_proc; i++) {
        double diff = (double)rand_nums[i] - mean;
        local_sq_diff_sum += diff * diff;
    }

    // 4) Reduce squared-diff sums to root (rank 0)
    double global_sq_diff_sum = 0.0;
    MPI_Reduce(&local_sq_diff_sum, &global_sq_diff_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // 5) Root computes and prints final standard deviation
    if (rank == 0) {
        double variance = global_sq_diff_sum / (double)global_count; // population variance
        double stddev = sqrt(variance);
        printf("nproc=%d, n_per_proc=%d, total_n=%lld\n", nproc, num_elements_per_proc, global_count);
        printf("mean   = %.10f\n", mean);
        printf("stddev = %.10f\n", stddev);
    }

    free(rand_nums);
    MPI_Finalize();
    return 0;
}
