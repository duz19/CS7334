// compute_pi.c (OpenMP)
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    long num_steps = atol(argv[1]);
    if (num_steps <= 0) {
        fprintf(stderr, "N must be > 0\n");
        return 1;
    }

    const double step = 1.0 / (double)num_steps;
    double sum = 0.0;

    // ---- timed region (exclude parsing/printing) ----
    double t0 = omp_get_wtime();

    #pragma omp parallel for default(none) shared(num_steps, step) reduction(+:sum) schedule(runtime)
    for (long i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;

    double t1 = omp_get_wtime();
    // ---- end timed region ----

    printf("Pi = %.12f\n", pi);
    printf("Time = %.6f seconds\n", t1 - t0);
    return 0;
}
