import os
import platform
import time
import threading
from queue import Queue


STUDENT_ID = 230103374
TOTAL_ITEMS = 1000
NUM_WORKERS = 4
DYNAMIC_CHUNK_SIZE = 5


def simulate_work(item):
    delay = 0.0001 * (item % 20)
    time.sleep(delay)


def static_worker(worker_id, start, end, completion_times):

    for i in range(start, end):
        simulate_work(i)

    completion_times[worker_id] = time.perf_counter()


def run_static():

    threads = []
    completion_times = [0.0] * NUM_WORKERS

    chunk_size = TOTAL_ITEMS // NUM_WORKERS

    start_time = time.perf_counter()

    for worker_id in range(NUM_WORKERS):

        start = worker_id * chunk_size

        if worker_id == NUM_WORKERS - 1:
            end = TOTAL_ITEMS
        else:
            end = start + chunk_size

        thread = threading.Thread(
            target=static_worker,
            args=(worker_id, start, end, completion_times)
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000

    completion_delta_ms = (
        max(completion_times) - min(completion_times)
    ) * 1000

    return total_time_ms, completion_delta_ms


def dynamic_worker(worker_id, task_queue, completion_times):

    while True:

        try:
            chunk = task_queue.get_nowait()
        except:
            break

        for item in chunk:
            simulate_work(item)

        task_queue.task_done()

    completion_times[worker_id] = time.perf_counter()


def run_dynamic():

    threads = []
    completion_times = [0.0] * NUM_WORKERS

    task_queue = Queue()


    for start in range(0, TOTAL_ITEMS, DYNAMIC_CHUNK_SIZE):

        end = min(start + DYNAMIC_CHUNK_SIZE, TOTAL_ITEMS)

        task_queue.put(range(start, end))

    start_time = time.perf_counter()

    for worker_id in range(NUM_WORKERS):

        thread = threading.Thread(
            target=dynamic_worker,
            args=(worker_id, task_queue, completion_times)
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000

    completion_delta_ms = (
        max(completion_times) - min(completion_times)
    ) * 1000

    return total_time_ms, completion_delta_ms


def main():

    print("===== HARDWARE INFORMATION =====")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"CPU Model: {platform.processor()}")
    print(f"Logical CPU Cores: {os.cpu_count()}")

    print("\n===== BENCHMARK CONFIGURATION =====")
    print(f"Total Items: {TOTAL_ITEMS}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Dynamic Chunk Size: {DYNAMIC_CHUNK_SIZE}")
    print("Workload: sleep(0.0001 * (i % 20))")


    static_time, static_delta = run_static()


    dynamic_time, dynamic_delta = run_dynamic()

    print("\n===== SCHEDULING BENCHMARK =====")

    print(
        f"{'Strategy':<20}"
        f"{'Wall Time (ms)':<20}"
        f"{'Completion Delta (ms)':<25}"
    )

    print("-" * 65)

    print(
        f"{'Static Scheduling':<20}"
        f"{static_time:<20.2f}"
        f"{static_delta:<25.2f}"
    )

    print(
        f"{'Dynamic Scheduling':<20}"
        f"{dynamic_time:<20.2f}"
        f"{dynamic_delta:<25.2f}"
    )

    speedup = static_time / dynamic_time

    print(f"\nDynamic Speedup: {speedup:.2f}x")

    print("\n===== TASK 4 COMPLETED =====")


if __name__ == "__main__":
    main()