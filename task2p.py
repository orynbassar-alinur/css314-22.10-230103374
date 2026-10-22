import numpy as np
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103374

TOTAL_SAMPLES = 40_000_000
NUM_THREADS = 4

SAMPLES_PER_THREAD = TOTAL_SAMPLES // NUM_THREADS

print("=" * 55)
print("TASK 2: PSEUDO-RANDOM GENERATOR STATE BOTTLENECK")
print("=" * 55)

print(f"Student ID          : {STUDENT_ID}")
print(f"Total Samples       : {TOTAL_SAMPLES:,}")
print(f"Threads             : {NUM_THREADS}")
print(f"Samples per Thread  : {SAMPLES_PER_THREAD:,}")


shared_rng = np.random.default_rng(STUDENT_ID)
rng_lock = threading.Lock()


def shared_worker(thread_index):

    inside_circle = 0

    for _ in range(SAMPLES_PER_THREAD):

        with rng_lock:
            x = shared_rng.random()
            y = shared_rng.random()

        if x * x + y * y <= 1.0:
            inside_circle += 1

    return inside_circle


def run_shared():

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:

        results = list(
            executor.map(shared_worker, range(NUM_THREADS))
        )

    total_inside = sum(results)

    pi_estimate = 4.0 * total_inside / TOTAL_SAMPLES

    elapsed_ms = (time.perf_counter() - start) * 1000

    return pi_estimate, elapsed_ms


def independent_worker(thread_index):

    seed = STUDENT_ID * 10 + thread_index

    local_rng = np.random.default_rng(seed)

    inside_circle = 0

    for _ in range(SAMPLES_PER_THREAD):

        x = local_rng.random()
        y = local_rng.random()

        if x * x + y * y <= 1.0:
            inside_circle += 1

    return inside_circle


def run_independent():

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:

        results = list(
            executor.map(independent_worker, range(NUM_THREADS))
        )

    total_inside = sum(results)

    pi_estimate = 4.0 * total_inside / TOTAL_SAMPLES

    elapsed_ms = (time.perf_counter() - start) * 1000

    return pi_estimate, elapsed_ms


print("\nRunning Shared PRNG benchmark...")

shared_pi, shared_time = run_shared()

print("Shared PRNG completed.")

print("\nRunning Independent PRNG benchmark...")

independent_pi, independent_time = run_independent()

print("Independent PRNG completed.")


shared_error = abs(shared_pi - math.pi)

independent_error = abs(independent_pi - math.pi)

speedup = shared_time / independent_time

print("\n" + "=" * 55)
print("BENCHMARK RESULTS")
print("=" * 55)

print(f"Student ID             : {STUDENT_ID}")
print(f"Total Samples          : {TOTAL_SAMPLES:,}")
print(f"Threads                : {NUM_THREADS}")

print("-" * 55)

print("MODE A: SHARED PRNG")

print(f"Computed PI            : {shared_pi:.10f}")
print(f"Absolute Error         : {shared_error:.10f}")
print(f"Wall Time              : {shared_time:.3f} ms")

print("-" * 55)

print("MODE B: INDEPENDENT PRNG")

print(f"Computed PI            : {independent_pi:.10f}")
print(f"Absolute Error         : {independent_error:.10f}")
print(f"Wall Time              : {independent_time:.3f} ms")

print("-" * 55)

print(f"Speedup Factor         : {speedup:.2f}x")

print("=" * 55)