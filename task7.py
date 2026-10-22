import os
import platform
import time
import math

import numpy as np

from numba import njit, prange, set_num_threads, get_num_threads
from numba import config


STUDENT_ID = 230103374

ARRAY_SIZE = 30_000_000
REPEATS = 5


@njit(parallel=True)
def memory_kernel(a, b):

    for i in prange(len(a)):
        b[i] = a[i]


@njit(parallel=True)
def compute_kernel(a, b):

    for i in prange(len(a)):

        value = a[i]

        b[i] = (
            math.sin(value) * math.cos(value)
            + math.tan(value)
        )


def benchmark(kernel, a, b):

    times = []

    for _ in range(REPEATS):

        start = time.perf_counter()

        kernel(a, b)

        end = time.perf_counter()

        times.append((end - start) * 1000)

    return min(times)


def main():

    logical_cores = os.cpu_count() or 1

    max_threads = min(
        logical_cores,
        config.NUMBA_NUM_THREADS
    )

    worker_counts = list(dict.fromkeys([
        1,
        2,
        max_threads
    ]))

    print("===== HARDWARE INFORMATION =====")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"CPU Model: {platform.processor()}")
    print(f"Logical CPU Cores: {logical_cores}")
    print(f"Max Parallel Threads: {max_threads}")

    print("\n===== BENCHMARK CONFIGURATION =====")
    print(f"Array Size: {ARRAY_SIZE:,}")
    print("Data Type: float64")
    print(f"Repeats: {REPEATS}")
    print(f"Thread Counts: {worker_counts}")


    rng = np.random.default_rng(STUDENT_ID)

    a = rng.random(ARRAY_SIZE, dtype=np.float64)

    b = np.empty_like(a)

    set_num_threads(1)

    memory_kernel(a[:100], b[:100])
    compute_kernel(a[:100], b[:100])


    print("\n===== KERNEL A: MEMORY-BOUND =====")

    print(
        f"{'Cores':<10}"
        f"{'Time (ms)':<15}"
        f"{'Speedup':<15}"
        f"{'Efficiency':<15}"
    )

    print("-" * 55)

    memory_results = {}

    for cores in worker_counts:

        set_num_threads(cores)

        elapsed = benchmark(memory_kernel, a, b)

        memory_results[cores] = elapsed

        speedup = memory_results[1] / elapsed
        efficiency = speedup / cores

        print(
            f"{cores:<10}"
            f"{elapsed:<15.2f}"
            f"{speedup:<15.2f}"
            f"{efficiency:<15.2%}"
        )


    print("\n===== KERNEL B: COMPUTE-BOUND =====")

    print(
        f"{'Cores':<10}"
        f"{'Time (ms)':<15}"
        f"{'Speedup':<15}"
        f"{'Efficiency':<15}"
    )

    print("-" * 55)

    compute_results = {}

    for cores in worker_counts:

        set_num_threads(cores)

        elapsed = benchmark(compute_kernel, a, b)

        compute_results[cores] = elapsed

        speedup = compute_results[1] / elapsed
        efficiency = speedup / cores

        print(
            f"{cores:<10}"
            f"{elapsed:<15.2f}"
            f"{speedup:<15.2f}"
            f"{efficiency:<15.2%}"
        )


    memory_speedup = (
        memory_results[1] / memory_results[max_threads]
    )

    compute_speedup = (
        compute_results[1] / compute_results[max_threads]
    )

    memory_efficiency = memory_speedup / max_threads
    compute_efficiency = compute_speedup / max_threads

    print("\n===== SCALING ANALYSIS =====")

    print(f"Memory Kernel Speedup: {memory_speedup:.2f}x")
    print(f"Compute Kernel Speedup: {compute_speedup:.2f}x")

    print(f"Memory Kernel Efficiency: {memory_efficiency:.2%}")
    print(f"Compute Kernel Efficiency: {compute_efficiency:.2%}")

    print("\n===== BOTTLENECK CONCLUSION =====")

    if memory_efficiency < compute_efficiency:

        print("Kernel A (Memory-Bound) stalled in scaling.")
        print("Likely cause: memory bandwidth saturation.")

    else:

        print("Kernel B showed equal or worse scaling.")
        print("Memory saturation was not conclusively demonstrated.")

    print("\n===== TASK 7 COMPLETED =====")


if __name__ == "__main__":
    main()