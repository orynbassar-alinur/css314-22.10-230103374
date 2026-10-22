#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <stdint.h>

#define STUDENT_ID 230103374
#define NUM_THREADS 8
#define INCREMENTS_PER_THREAD 1000000LL

#define TOTAL_OPERATIONS \
    (NUM_THREADS * INCREMENTS_PER_THREAD)


LONGLONG mutex_counter = 0;

volatile LONG64 atomic_counter = 0;

CRITICAL_SECTION counter_lock;

double get_time_ms() {

    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);

    return (
        (double)counter.QuadPart * 1000.0 /
        (double)frequency.QuadPart
    );
}

DWORD WINAPI mutex_worker(LPVOID arg) {

    for (int i = 0; i < INCREMENTS_PER_THREAD; i++) {

        EnterCriticalSection(&counter_lock);

        mutex_counter++;

        LeaveCriticalSection(&counter_lock);
    }

    return 0;
}


DWORD WINAPI atomic_worker(LPVOID arg) {

    for (int i = 0; i < INCREMENTS_PER_THREAD; i++) {

        LONG64 expected;
        LONG64 previous;

        do {

            expected = InterlockedCompareExchange64(
                &atomic_counter,
                0,
                0
            );

            previous = InterlockedCompareExchange64(
                &atomic_counter,
                expected + 1,
                expected
            );

        } while (previous != expected);
    }

    return 0;
}


double run_benchmark(
    LPTHREAD_START_ROUTINE worker
) {

    HANDLE threads[NUM_THREADS];

    double start = get_time_ms();

    for (int i = 0; i < NUM_THREADS; i++) {

        threads[i] = CreateThread(
            NULL,
            0,
            worker,
            NULL,
            0,
            NULL
        );

        if (threads[i] == NULL) {

            printf("ERROR: Thread creation failed!\n");

            exit(1);
        }
    }

    WaitForMultipleObjects(
        NUM_THREADS,
        threads,
        TRUE,
        INFINITE
    );

    for (int i = 0; i < NUM_THREADS; i++) {

        CloseHandle(threads[i]);
    }

    double end = get_time_ms();

    return end - start;
}


int main() {

    printf("============================================================\n");
    printf("TASK 6: ATOMIC CAS VS OS MUTEX\n");
    printf("============================================================\n");

    printf("Student ID             : %d\n", STUDENT_ID);
    printf("Threads                : %d\n", NUM_THREADS);
    printf("Increments per Thread  : %lld\n",
           INCREMENTS_PER_THREAD);

    printf("Total Operations       : %lld\n",
           TOTAL_OPERATIONS);

   
    InitializeCriticalSection(&counter_lock);

    printf("\nRunning Mutex benchmark...\n");

    mutex_counter = 0;

    double mutex_time = run_benchmark(mutex_worker);


    printf("Running Atomic CAS benchmark...\n");

    atomic_counter = 0;

    double atomic_time = run_benchmark(atomic_worker);


    if (
        mutex_counter != TOTAL_OPERATIONS ||
        atomic_counter != TOTAL_OPERATIONS
    ) {

        printf("\nERROR: Counter verification FAILED!\n");

        DeleteCriticalSection(&counter_lock);

        return 1;
    }


    double mutex_throughput =
        TOTAL_OPERATIONS / (mutex_time / 1000.0);

    double atomic_throughput =
        TOTAL_OPERATIONS / (atomic_time / 1000.0);

    double throughput_ratio =
        atomic_throughput / mutex_throughput;


    printf("\n============================================================\n");
    printf("BENCHMARK RESULTS\n");
    printf("============================================================\n");

    printf("\nPRIMITIVE 1: OS LOCK\n");

    printf("Execution Time         : %.3f ms\n",
           mutex_time);

    printf("Throughput             : %.3f ops/sec\n",
           mutex_throughput);

    printf("Final Counter          : %lld\n",
           mutex_counter);

    printf("------------------------------------------------------------\n");

    printf("PRIMITIVE 2: ATOMIC CAS\n");

    printf("Execution Time         : %.3f ms\n",
           atomic_time);

    printf("Throughput             : %.3f ops/sec\n",
           atomic_throughput);

    printf("Final Counter          : %lld\n",
           atomic_counter);

    printf("------------------------------------------------------------\n");

    printf("Throughput Ratio       : %.2fx\n",
           throughput_ratio);

    printf("Verification Status    : PASS\n");

    printf("============================================================\n");
}