#include <stdio.h>
#include <omp.h>

float dot_prod(float* a, float* b, int N)
{
    float sum = 0.0;

    #pragma omp parallel for shared(sum)
    for(int i = 0; i < N; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

int main() {

    // Hello from threads
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("Hello from thread %d out of %d threads\n", tid, nthreads);
    }

    // Example vectors
    float a[] = {1.0, 2.0, 3.0, 4.0};
    float b[] = {5.0, 6.0, 7.0, 8.0};
    int N = 4;

    // Compute dot product
    float result = dot_prod(a, b, N);

    printf("\nDot Product = %f\n", result);

    return 0;
}

