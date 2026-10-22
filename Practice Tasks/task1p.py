import numpy as np
import numba
from numba import njit, prange
import psutil
import time
import os


STUDENT_ID = 230103374  

ROWS = 8000
COLS = 8000


logical_cores = psutil.cpu_count(logical=True)
physical_cores = psutil.cpu_count(logical=False)


numba.set_num_threads(physical_cores)

print("=" * 50)
print("TASK 1: ROW-MAJOR VS COLUMN-MAJOR")
print("=" * 50)

print(f"Student ID       : {STUDENT_ID}")
print(f"Logical Cores    : {logical_cores}")
print(f"Physical Cores   : {physical_cores}")
print(f"Threads Used     : {numba.get_num_threads()}")


print("\nAllocating 8000 x 8000 float64 matrix...")

rng = np.random.default_rng(STUDENT_ID)

matrix = rng.random(
    (ROWS, COLS),
    dtype=np.float64
)

print(f"Matrix Size      : {matrix.nbytes / (1024**2):.2f} MB")


@njit(parallel=True)
def row_major_sum(arr):

    total = 0.0

    for i in prange(arr.shape[0]):

        row_sum = 0.0

        for j in range(arr.shape[1]):
            row_sum += arr[i, j]

        total += row_sum

    return total


@njit(parallel=True)
def column_major_sum(arr):

    total = 0.0

    for j in prange(arr.shape[1]):

        col_sum = 0.0

        for i in range(arr.shape[0]):
            col_sum += arr[i, j]

        total += col_sum

    return total


print("\nCompiling Numba functions...")

row_major_sum(matrix[:10, :10].copy())
column_major_sum(matrix[:10, :10].copy())


def benchmark(func, arr, repetitions=5):

    times = []

    for _ in range(repetitions):

        start = time.perf_counter()

        result = func(arr)

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000

        times.append(elapsed_ms)

    return min(times), result


print("\nRunning benchmarks...")

row_time, row_result = benchmark(
    row_major_sum,
    matrix
)

col_time, col_result = benchmark(
    column_major_sum,
    matrix
)

cache_penalty = col_time / row_time

print("\n" + "=" * 50)
print("BENCHMARK RESULTS")
print("=" * 50)

print(f"Student ID              : {STUDENT_ID}")
print(f"Logical Core Count      : {logical_cores}")
print(f"Physical Core Count     : {physical_cores}")
print(f"Parallel Threads Used   : {numba.get_num_threads()}")

print("-" * 50)

print(f"Mode A (Row-Major)      : {row_time:.3f} ms")
print(f"Mode B (Column-Major)   : {col_time:.3f} ms")

print("-" * 50)

print(f"Cache Penalty Factor    : {cache_penalty:.2f}x")

print("-" * 50)

print(f"Row Sum                 : {row_result:.6f}")
print(f"Column Sum              : {col_result:.6f}")

print(
    f"Results Match           : "
    f"{np.isclose(row_result, col_result)}"
)

print("=" * 50)