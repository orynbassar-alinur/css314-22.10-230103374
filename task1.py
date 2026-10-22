import os
import platform
import time
import numpy as np

STUDENT_ID = 230103374

print("===== HARDWARE INFORMATION =====")
print(f"Student ID: {STUDENT_ID}")
print(f"Operating System: {platform.system()} {platform.release()}")
print(f"CPU Model: {platform.processor()}")
print(f"Platform: {platform.platform()}")
print(f"Logical CPU Cores: {os.cpu_count()}")

ARRAY_SIZE = 20_000_000

print("\n===== DATA GENERATION =====")
print(f"Array Size: {ARRAY_SIZE:,}")
print(f"Data Type: float64")
print(f"Random Seed: {STUDENT_ID}")

rng = np.random.default_rng(STUDENT_ID)

data = rng.random(ARRAY_SIZE, dtype=np.float64)

print("\n===== SERIAL COMPUTATION =====")

start_time = time.perf_counter()

total_sum = 0.0

for value in data:
    total_sum += value

mean = total_sum / ARRAY_SIZE

squared_diff_sum = 0.0

for value in data:
    squared_diff_sum += (value - mean) ** 2

std_dev = (squared_diff_sum / ARRAY_SIZE) ** 0.5

end_time = time.perf_counter()

execution_time_ms = (end_time - start_time) * 1000

print(f"Sum: {total_sum}")
print(f"Standard Deviation: {std_dev}")
print(f"Execution Time: {execution_time_ms:.2f} ms")

print("\n===== TASK 1 COMPLETED =====")