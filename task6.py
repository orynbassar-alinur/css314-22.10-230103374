import os
import platform
import time
import threading


STUDENT_ID = 230103374

NUM_THREADS = 4
INCREMENTS = 2_000_000

EXPECTED = NUM_THREADS * INCREMENTS

RACE_ROUNDS = 100


def unsafe_increment(counter, read_barrier, write_barrier):

    for i in range(INCREMENTS):

        value = counter[0]

        if i < RACE_ROUNDS:

            read_barrier.wait()


        counter[0] = value + 1

        if i < RACE_ROUNDS:
            
            write_barrier.wait()


def safe_increment(counter, lock):

    for i in range(INCREMENTS):

        with lock:
            value = counter[0]
            counter[0] = value + 1


# =====================================
# Benchmark Function
# =====================================

def run_benchmark(safe=False):

    counter = [0]

    lock = threading.Lock()

    read_barrier = threading.Barrier(NUM_THREADS)
    write_barrier = threading.Barrier(NUM_THREADS)

    threads = []

    start_time = time.perf_counter()

    for i in range(NUM_THREADS):

        if safe:

            thread = threading.Thread(
                target=safe_increment,
                args=(counter, lock)
            )

        else:

            thread = threading.Thread(
                target=unsafe_increment,
                args=(
                    counter,
                    read_barrier,
                    write_barrier
                )
            )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    wall_time_ms = (end_time - start_time) * 1000

    return counter[0], wall_time_ms



def main():

    print("===== HARDWARE INFORMATION =====")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"CPU Model: {platform.processor()}")
    print(f"Logical CPU Cores: {os.cpu_count()}")

    print("\n===== BENCHMARK CONFIGURATION =====")
    print(f"Threads: {NUM_THREADS}")
    print(f"Increments Per Thread: {INCREMENTS:,}")
    print(f"Expected Final Value: {EXPECTED:,}")
    print(f"Forced Race Rounds: {RACE_ROUNDS}")

  
    print("\n===== RACE CONDITION BENCHMARK =====")

    unsafe_actual, unsafe_time = run_benchmark(
        safe=False
    )

    lost_updates = EXPECTED - unsafe_actual

    lost_percentage = (
        lost_updates / EXPECTED
    ) * 100

    print(f"Expected Final Value: {EXPECTED:,}")
    print(f"Unsafe Actual Value: {unsafe_actual:,}")
    print(f"Lost Updates: {lost_updates:,}")
    print(f"Lost Percentage: {lost_percentage:.4f}%")


    safe_actual, safe_time = run_benchmark(
        safe=True
    )


    penalty_ms = safe_time - unsafe_time

    slowdown = (
        safe_time / unsafe_time
        if unsafe_time > 0
        else 0
    )

    print("\n===== SYNCHRONIZATION PERFORMANCE =====")

    print(f"Unsafe Time: {unsafe_time:.2f} ms")
    print(f"Safe Time (Lock): {safe_time:.2f} ms")
    print(f"Lock Penalty: {penalty_ms:.2f} ms")
    print(f"Slowdown: {slowdown:.2f}x")


    print("\n===== VERIFICATION =====")

    print(f"Safe Final Value: {safe_actual:,}")
    print(f"Safe Result Correct: {safe_actual == EXPECTED}")

    print("\n===== TASK 6 COMPLETED =====")


if __name__ == "__main__":
    main()