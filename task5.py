import os
import platform
import time
import numpy as np

from numba import njit, prange, set_num_threads, get_num_threads


STUDENT_ID = 230103374

ARRAY_SIZES = [
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000
]

A = 2.5
B = 1.5

REPEATS = 5


@njit
def serial_vector(x, y, a, b):

    for i in range(len(x)):
        y[i] = a * x[i] + b


@njit(parallel=True)
def parallel_vector(x, y, a, b):

    for i in prange(len(x)):
        y[i] = a * x[i] + b


def benchmark(func, x, y, a, b):

    times = []

    for _ in range(REPEATS):

        start = time.perf_counter()

        func(x, y, a, b)

        end = time.perf_counter()

        times.append((end - start) * 1000)

    return min(times)


def main():

    logical_cores = os.cpu_count() or 1


    available_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else logical_cores

    from numba import config

    max_numba_threads = config.NUMBA_NUM_THREADS

    num_threads = min(available_threads, max_numba_threads)

    set_num_threads(num_threads)

    print("===== HARDWARE INFORMATION =====")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"CPU Model: {platform.processor()}")
    print(f"Logical CPU Cores: {logical_cores}")
    print(f"Parallel Threads Used: {get_num_threads()}")

    print("\n===== BENCHMARK CONFIGURATION =====")
    print(f"Operation: y[i] = {A} * x[i] + {B}")
    print(f"Array Sizes: {ARRAY_SIZES}")
    print(f"Repeats: {REPEATS}")
    print("Timing Method: Minimum wall-clock time")


    warmup_x = np.ones(100, dtype=np.float64)
    warmup_y = np.empty_like(warmup_x)

    serial_vector(warmup_x, warmup_y, A, B)
    parallel_vector(warmup_x, warmup_y, A, B)

    print("\n===== SERIAL VS PARALLEL BENCHMARK =====")

    print(
        f"{'N':<15}"
        f"{'Serial (ms)':<18}"
        f"{'Parallel (ms)':<18}"
        f"{'Speedup':<12}"
        f"{'Winner':<12}"
    )

    print("-" * 75)

    results = []

    for n in ARRAY_SIZES:

        rng = np.random.default_rng(STUDENT_ID)

        x = rng.random(n, dtype=np.float64)

        serial_y = np.empty_like(x)
        parallel_y = np.empty_like(x)

        serial_time = benchmark(
            serial_vector,
            x,
            serial_y,
            A,
            B
        )

        parallel_time = benchmark(
            parallel_vector,
            x,
            parallel_y,
            A,
            B
        )


        if not np.allclose(serial_y, parallel_y):
            raise RuntimeError("Serial and Parallel results differ!")

        speedup = serial_time / parallel_time

        winner = "Parallel" if parallel_time < serial_time else "Serial"

        print(
            f"{n:<15,}"
            f"{serial_time:<18.4f}"
            f"{parallel_time:<18.4f}"
            f"{speedup:<12.2f}"
            f"{winner:<12}"
        )

        results.append((n, serial_time, parallel_time))


    print("\n===== CROSSOVER ANALYSIS =====")

    crossover_found = False

    for i in range(1, len(results)):

        prev_n, prev_serial, prev_parallel = results[i - 1]
        curr_n, curr_serial, curr_parallel = results[i]

        prev_parallel_slower = prev_parallel > prev_serial
        curr_parallel_faster = curr_parallel < curr_serial

        if prev_parallel_slower and curr_parallel_faster:

            print(
                f"Crossover detected between "
                f"N = {prev_n:,} and N = {curr_n:,}"
            )

            print(
                f"Parallel execution becomes faster "
                f"at tested size N = {curr_n:,}"
            )

            crossover_found = True

    if not crossover_found:
        print(
            "No slower-to-faster crossover detected "
            "within the tested array sizes."
        )

    print("\n===== TASK 5 COMPLETED =====")


if __name__ == "__main__":
    main()