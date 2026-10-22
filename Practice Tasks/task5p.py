import hashlib
import time
import os
import psutil
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103374

TOTAL_TASKS = 10_000
HASH_ITERATIONS = 500
DATA_SIZE = 1024

physical_cores = psutil.cpu_count(logical=False) or 4
logical_cores = psutil.cpu_count(logical=True) or os.cpu_count()

POOL_SIZES = [
    1,
    max(1, physical_cores // 2),
    physical_cores,
    physical_cores * 2,
    physical_cores * 4,
    physical_cores * 16,
    256
]

POOL_SIZES = sorted(set(POOL_SIZES))


data = (
    str(STUDENT_ID).encode() * 128
)[:DATA_SIZE]

data = data.ljust(DATA_SIZE, b"0")

def cpu_bound_task(task_id):

    result = data

    for _ in range(HASH_ITERATIONS):

        result = hashlib.sha256(result).digest()

    return result


def benchmark(pool_size):

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=pool_size) as executor:

        results = list(
            executor.map(
                cpu_bound_task,
                range(TOTAL_TASKS)
            )
        )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return elapsed_ms, len(results)


def main():

    print("=" * 75)
    print("TASK 5: THREAD OVERSUBSCRIPTION THRESHOLD")
    print("=" * 75)

    print(f"Student ID       : {STUDENT_ID}")
    print(f"Physical Cores   : {physical_cores}")
    print(f"Logical Cores    : {logical_cores}")
    print(f"Total Tasks      : {TOTAL_TASKS:,}")
    print(f"Hash Iterations  : {HASH_ITERATIONS}")
    print(f"Data Size        : {DATA_SIZE} bytes")

    print(f"\nPool Sizes       : {POOL_SIZES}")

    results = []

    baseline_time = None
    previous_time = None
    degradation_started = None

    print("\nRunning benchmarks...\n")

    for pool_size in POOL_SIZES:

        print(
            f"Testing pool size {pool_size}...",
            flush=True
        )

        elapsed_ms, completed = benchmark(pool_size)

        assert completed == TOTAL_TASKS

        if baseline_time is None:
            baseline_time = elapsed_ms

        speedup = baseline_time / elapsed_ms

        if pool_size == 1:

            status = "BASELINE"

        elif (
            previous_time is not None
            and elapsed_ms > previous_time
        ):

            if degradation_started is None:

                degradation_started = pool_size
                status = "DEGRADATION START"

            else:

                status = "SLOWER"

        elif pool_size > physical_cores:

            status = "OVERSUBSCRIBED"

        else:

            status = "OK"

        results.append(
            (
                pool_size,
                elapsed_ms,
                speedup,
                status
            )
        )

        previous_time = elapsed_ms


    print("\n" + "=" * 85)
    print("BENCHMARK RESULTS")
    print("=" * 85)

    print(
        f"{'Pool Size':<15}"
        f"{'Execution Time (ms)':<25}"
        f"{'Speedup':<15}"
        f"{'Status':<25}"
    )

    print("-" * 85)

    for pool_size, elapsed_ms, speedup, status in results:

        print(
            f"{pool_size:<15}"
            f"{elapsed_ms:<25.3f}"
            f"{speedup:<15.2f}x"
            f"{status:<25}"
        )

    print("=" * 85)

    if degradation_started is not None:

        print(
            f"\nDegradation starts at pool size: "
            f"{degradation_started}"
        )

    else:

        print(
            "\nNo execution-time degradation detected."
        )


if __name__ == "__main__":
    main()