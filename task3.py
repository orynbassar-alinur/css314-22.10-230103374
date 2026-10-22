import os
import platform
import time
import threading

import numpy as np
import psutil
from numba import njit


STUDENT_ID = 230103374
ITERATIONS = 50_000_000
CACHE_LINE_SIZE = 64


@njit(nogil=True)
def increment_counter(array, index, iterations):
    for i in range(iterations):
        array[index] += 1


def run_benchmark(array, indices, num_threads):
    threads = []

    start = time.perf_counter()

    for i in range(num_threads):
        t = threading.Thread(
            target=increment_counter,
            args=(array, indices[i], ITERATIONS)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end = time.perf_counter()

    return (end - start) * 1000


def main():
    physical_cores = psutil.cpu_count(logical=False)

    if physical_cores is None:
        raise RuntimeError("Cannot detect physical CPU cores")

    logical_cores = os.cpu_count()

    print("===== HARDWARE INFORMATION =====")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"CPU Model: {platform.processor()}")
    print(f"Physical CPU Cores: {physical_cores}")
    print(f"Logical CPU Cores: {logical_cores}")

    print("\n===== BENCHMARK CONFIGURATION =====")
    print(f"Threads: {physical_cores}")
    print(f"Iterations Per Thread: {ITERATIONS:,}")
    print(f"Cache Line Size: {CACHE_LINE_SIZE} bytes")


    warmup = np.zeros(physical_cores * 8, dtype=np.int64)
    increment_counter(warmup, 0, 1)


    shared_counts = np.zeros(physical_cores, dtype=np.int64)
    unpadded_indices = list(range(physical_cores))


    padded_counts = np.zeros(physical_cores * 8, dtype=np.int64)
    padded_indices = [i * 8 for i in range(physical_cores)]

    print("\n===== FALSE SHARING BENCHMARK =====")

    unpadded_time = run_benchmark(
        shared_counts,
        unpadded_indices,
        physical_cores
    )

    padded_time = run_benchmark(
        padded_counts,
        padded_indices,
        physical_cores
    )


    unpadded_correct = np.all(shared_counts == ITERATIONS)
    padded_correct = np.all(
        padded_counts[padded_indices] == ITERATIONS
    )

    ratio = unpadded_time / padded_time

    print(f"Unpadded Shared: {unpadded_time:.2f} ms")
    print(f"64-byte Padded:  {padded_time:.2f} ms")
    print(f"Execution Ratio: {ratio:.2f}x")

    print("\n===== VERIFICATION =====")
    print(f"Expected Count Per Thread: {ITERATIONS:,}")
    print(f"Unpadded Correct: {unpadded_correct}")
    print(f"Padded Correct: {padded_correct}")

    print("\n===== TASK 3 COMPLETED =====")


if __name__ == "__main__":
    main()