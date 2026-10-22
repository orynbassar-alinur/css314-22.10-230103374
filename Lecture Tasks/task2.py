import os
import platform
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

STUDENT_ID = 230103374
ARRAY_SIZE = 20_000_000
P = 0.95


def calculate_sum(chunk):
    total = 0.0

    for value in chunk:
        total += value

    return total


def parallel_sum(data, workers):

    chunks = np.array_split(data, workers)

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(calculate_sum, chunks)
        total_sum = sum(results)

    end_time = time.perf_counter()

    wall_time_ms = (end_time - start_time) * 1000

    return total_sum, wall_time_ms


def amdahl_speedup(workers):
    return 1 / ((1 - P) + (P / workers))


def main():

    max_cores = os.cpu_count() or 1

    # Hardware Information
    print("===== HARDWARE INFORMATION =====")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"CPU Model: {platform.processor()}")
    print(f"Logical CPU Cores: {max_cores}")

    
    print("\n===== DATA GENERATION =====")

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    print(f"Array Size: {ARRAY_SIZE:,}")
    print(f"Data Type: {data.dtype}")
    print(f"Random Seed: {STUDENT_ID}")

    
    worker_counts = [
        1,
        2,
        4,
        max_cores,
        max_cores * 2
    ]

    worker_counts = list(dict.fromkeys(worker_counts))


    print("\n===== MULTI-CORE SCALING BENCHMARK =====")

    print(
        f"{'Cores':<10}"
        f"{'Wall Time (ms)':<20}"
        f"{'Speedup':<15}"
        f"{'Amdahl Upper Bound':<20}"
    )

    print("-" * 65)

    baseline_time = None

    for workers in worker_counts:

        total_sum, wall_time = parallel_sum(data, workers)

        if baseline_time is None:
            baseline_time = wall_time

        speedup = baseline_time / wall_time
        theoretical_speedup = amdahl_speedup(workers)

        print(
            f"{workers:<10}"
            f"{wall_time:<20.2f}"
            f"{speedup:<15.2f}"
            f"{theoretical_speedup:<20.2f}"
        )

    print("\n===== TASK 2 COMPLETED =====")


if __name__ == "__main__":
    main()