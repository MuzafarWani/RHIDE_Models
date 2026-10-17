#!/usr/bin/env python3
import tracemalloc
import gc
import numpy as np

# -----------------------------
# Four independent code parts
# -----------------------------

def part_1():
    # Light workload: Python objects + small arrays
    data = []
    for i in range(50000):
        data.append({"i": i, "s": str(i)})
    arr = np.random.rand(1000, 1000)
    return data, arr


def part_2():
    # Medium workload: larger arrays + transforms
    a = np.random.rand(2000, 2000)
    b = np.fft.fft2(a)
    c = np.abs(b)
    return c


def part_3():
    # Heavier workload: multiple intermediate tensors
    x = np.random.rand(3000, 1500)
    y = np.random.rand(1500, 2000)
    z = x @ y
    w = np.tanh(z)
    return w


def part_4():
    # Mixed workload: text + numeric + temporary buffers
    blobs = []
    for i in range(2000):
        blobs.append("x" * (1000 + i))
    mats = [np.random.rand(800, 800) for _ in range(5)]
    out = sum(mats)
    return blobs, out


# -----------------------------
# Measurement harness
# -----------------------------

def measure_part(fn, name):
    tracemalloc.reset_peak()

    result = fn()

    current, peak = tracemalloc.get_traced_memory()

    print(f"{name}")
    print(f"  Current memory after part : {current / (1024**2):.2f} MB")
    print(f"  Peak memory during part   : {peak / (1024**2):.2f} MB")
    print("-" * 50)

    # Explicit cleanup
    del result
    gc.collect()


if __name__ == "__main__":
    print("Starting runtime memory tracking...\n")
    tracemalloc.start()

    measure_part(part_1, "Part 1")
    measure_part(part_2, "Part 2")
    measure_part(part_3, "Part 3")
    measure_part(part_4, "Part 4")

    tracemalloc.stop()

