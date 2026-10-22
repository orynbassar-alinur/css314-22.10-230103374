import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import os

STUDENT_ID = 230103374

ARRAY_SIZES = [
    100_000,
    1_000_000,
    10_000_000
]

REPEATS = 5


def pipe_worker(conn):

    try:
        while True:

            data = conn.recv()

            if data is None:
                break

            checksum = int(data.sum())

            conn.send(checksum)

    finally:
        conn.close()


def benchmark_pipe(arr):

    parent_conn, child_conn = mp.Pipe()

    process = mp.Process(
        target=pipe_worker,
        args=(child_conn,)
    )

    process.start()
    child_conn.close()

    times = []

    try:
        for _ in range(REPEATS):

            start = time.perf_counter()

            parent_conn.send(arr)

            checksum = parent_conn.recv()

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            assert checksum == int(arr.sum())

            times.append(elapsed)

    finally:
        parent_conn.send(None)
        parent_conn.close()

        process.join()

    return min(times)


def shared_worker(shm_name, shape, dtype, conn):

    shm = shared_memory.SharedMemory(
        name=shm_name
    )

    try:

        arr = np.ndarray(
            shape,
            dtype=np.dtype(dtype),
            buffer=shm.buf
        )

        while True:

            command = conn.recv()

            if command == "STOP":
                break

            if command == "PROCESS":

                checksum = int(arr.sum())

                conn.send(checksum)

    finally:
        del arr
        shm.close()
        conn.close()


def benchmark_shared(arr):

    shm = shared_memory.SharedMemory(
        create=True,
        size=arr.nbytes
    )

    shared_arr = np.ndarray(
        arr.shape,
        dtype=arr.dtype,
        buffer=shm.buf
    )

    np.copyto(shared_arr, arr)

    parent_conn, child_conn = mp.Pipe()

    process = mp.Process(
        target=shared_worker,
        args=(
            shm.name,
            arr.shape,
            arr.dtype.str,
            child_conn
        )
    )

    process.start()
    child_conn.close()

    times = []

    try:
        for _ in range(REPEATS):

            start = time.perf_counter()

            parent_conn.send("PROCESS")

            checksum = parent_conn.recv()

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            assert checksum == int(arr.sum())

            times.append(elapsed)

    finally:
        parent_conn.send("STOP")
        parent_conn.close()

        process.join()

        del shared_arr
        shm.close()
        shm.unlink()

    return min(times)


def main():

    print("=" * 65)
    print("TASK 4: IPC VS SHARED MEMORY TRANSFER")
    print("=" * 65)

    print(f"Student ID       : {STUDENT_ID}")
    print(f"CPU Cores        : {os.cpu_count()}")
    print(f"Repetitions      : {REPEATS}")

    rng = np.random.default_rng(STUDENT_ID)

    results = []

    for n in ARRAY_SIZES:

        print(f"\nBenchmarking N = {n:,}...")

        arr = rng.integers(
            low=1,
            high=11,
            size=n,
            dtype=np.int64
        )

        pipe_time = benchmark_pipe(arr)

        shared_time = benchmark_shared(arr)

        speedup = pipe_time / shared_time

        results.append(
            (n, pipe_time, shared_time, speedup)
        )

        print(f"Path A (Pipe/Pickle)   : {pipe_time:.3f} ms")
        print(f"Path B (Shared Memory) : {shared_time:.3f} ms")
        print(f"Speedup                : {speedup:.2f}x")


    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)

    print(
        f"{'Array Size':<15}"
        f"{'Pipe (ms)':<15}"
        f"{'Shared (ms)':<15}"
        f"{'Speedup':<12}"
    )

    print("-" * 65)

    for n, pipe_time, shared_time, speedup in results:

        print(
            f"{n:<15,}"
            f"{pipe_time:<15.3f}"
            f"{shared_time:<15.3f}"
            f"{speedup:<12.2f}x"
        )

    print("=" * 65)


if __name__ == "__main__":
    mp.freeze_support()
    main()