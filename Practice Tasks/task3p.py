import numpy as np
import numba
from numba import njit, prange
import psutil
import time
import sys

# ==========================================
# TASK 3: PARALLEL PREFIX SCAN
# ==========================================

STUDENT_ID = 230103374

ARRAY_SIZE = 2_000_000

# CPU information
logical_cores = psutil.cpu_count(logical=True)
physical_cores = psutil.cpu_count(logical=False)

NUM_THREADS = physical_cores or 4

numba.set_num_threads(NUM_THREADS)

print("=" * 60)
print("TASK 3: PREFIX SCAN")
print("=" * 60)

print(f"Student ID       : {STUDENT_ID}")
print(f"Array Size       : {ARRAY_SIZE:,}")
print(f"Logical Cores    : {logical_cores}")
print(f"Physical Cores   : {physical_cores}")
print(f"Threads Used     : {numba.get_num_threads()}")

# ==========================================
# GENERATE INPUT ARRAY
# ==========================================

rng = np.random.default_rng(STUDENT_ID)

arr = rng.integers(
    low=1,
    high=11,
    size=ARRAY_SIZE,
    dtype=np.int64
)

print("\nInput array generated successfully.")

# ==========================================
# SERIAL INCLUSIVE PREFIX SUM
# ==========================================

@njit
def serial_prefix_sum(arr):

    n = len(arr)

    output = np.empty(n, dtype=np.int64)

    output[0] = arr[0]

    for i in range(1, n):

        output[i] = output[i - 1] + arr[i]

    return output


# ==========================================
# PARALLEL CHUNKED PREFIX SCAN
# ==========================================

@njit(parallel=True)
def parallel_prefix_sum(arr, num_threads):

    n = len(arr)

    output = np.empty(n, dtype=np.int64)

    chunk_size = (n + num_threads - 1) // num_threads

    chunk_sums = np.zeros(num_threads, dtype=np.int64)

    # PASS 1: Local prefix sums
    for t in prange(num_threads):

        start = t * chunk_size

        end = min(start + chunk_size, n)

        local_sum = 0

        for i in range(start, end):

            local_sum += arr[i]

            output[i] = local_sum

        chunk_sums[t] = local_sum

    # PASS 2: Calculate chunk offsets
    offsets = np.zeros(num_threads, dtype=np.int64)

    running_total = 0

    for t in range(num_threads):

        offsets[t] = running_total

        running_total += chunk_sums[t]

    # PASS 3: Apply offsets
    for t in prange(num_threads):

        start = t * chunk_size

        end = min(start + chunk_size, n)

        offset = offsets[t]

        for i in range(start, end):

            output[i] += offset

    return output


# ==========================================
# JIT WARM-UP
# ==========================================

print("\nCompiling Numba functions...")

test_arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)

serial_prefix_sum(test_arr)
parallel_prefix_sum(test_arr, NUM_THREADS)

# ==========================================
# BENCHMARK
# ==========================================

def benchmark(func, *args, repetitions=5):

    times = []
    result = None

    for _ in range(repetitions):

        start = time.perf_counter()

        result = func(*args)

        end = time.perf_counter()

        times.append((end - start) * 1000)

    return min(times), result


print("\nRunning Serial Prefix Sum...")

serial_time, serial_output = benchmark(
    serial_prefix_sum,
    arr
)

print("Serial execution completed.")

print("\nRunning Parallel Prefix Sum...")

parallel_time, parallel_output = benchmark(
    parallel_prefix_sum,
    arr,
    NUM_THREADS
)

print("Parallel execution completed.")

# ==========================================
# VERIFICATION
# ==========================================

verification_status = "FAIL"

try:

    assert parallel_output[-1] == serial_output[-1]

    verification_status = "PASS"

except AssertionError:

    print("\nERROR: Prefix scan verification FAILED!")

    sys.exit(1)

# Additional verification
full_match = np.array_equal(
    serial_output,
    parallel_output
)

if not full_match:

    print("\nERROR: Full prefix scan mismatch!")

    sys.exit(1)

# ==========================================
# RESULTS
# ==========================================

speedup = serial_time / parallel_time

print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print(f"Student ID          : {STUDENT_ID}")
print(f"Array Size          : {ARRAY_SIZE:,}")
print(f"Threads Used        : {NUM_THREADS}")

print("-" * 60)

print(f"Serial Time         : {serial_time:.3f} ms")
print(f"Parallel Time       : {parallel_time:.3f} ms")

print("-" * 60)

print(f"Speedup             : {speedup:.2f}x")

print("-" * 60)

print(f"Verification Status : {verification_status}")
print(f"Full Array Match    : {full_match}")

print(f"Final Prefix Sum    : {serial_output[-1]}")

print("=" * 60)