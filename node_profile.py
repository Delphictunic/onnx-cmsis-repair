import onnxruntime as rt
import numpy as np
import json
import tempfile
import os
from collections import defaultdict


def profile_nodes(model_path, runs=200, warmup=20):

    print(f"\nProfiling model: {model_path}")
    print("-" * 60)

    options = rt.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = tempfile.mktemp()

    sess = rt.InferenceSession(model_path, sess_options=options)

    input_name = sess.get_inputs()[0].name
    input_shape = [d if isinstance(d, int) and d > 0 else 1
                   for d in sess.get_inputs()[0].shape]

    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    # Timed runs
    for _ in range(runs):
        sess.run(None, {input_name: dummy})

    prof_file = sess.end_profiling()

    with open(prof_file) as f:
        events = json.load(f)

    os.remove(prof_file)

    # Collect node timings
    node_times = defaultdict(float)
    node_counts = defaultdict(int)

    for e in events:
        if e.get("cat") == "Node" and "dur" in e:
            op = e.get("args", {}).get("op_name", "unknown")
            name = e.get("name", "unknown")
            key = f"{op}/{name}"

            node_times[key] += e["dur"]
            node_counts[key] += 1

    print(f"\nRuns analyzed: {runs}")
    print()

    print(f"{'Operation':40s} {'Avg (μs)':>10} {'Total (μs)':>12} {'Count':>8}")
    print("-" * 75)

    total_kernel_time = 0

    for op in sorted(node_times, key=node_times.get, reverse=True):

        total = node_times[op]
        count = node_counts[op]
        avg = total / count

        total_kernel_time += total

        print(f"{op:40s} {avg:10.2f} {total:12.0f} {count:8}")

    print("-" * 75)
    print(f"{'TOTAL KERNEL TIME':40s} {total_kernel_time / runs:10.2f} μs per run")

    return node_times


def compare_models(model_a, model_b):

    print("\nComparing models\n")

    a = profile_nodes(model_a)
    b = profile_nodes(model_b)


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("python node_profile.py model.onnx [patched.onnx]")
        exit()

    model = sys.argv[1]

    if len(sys.argv) == 2:
        profile_nodes(model)

    else:
        compare_models(sys.argv[1], sys.argv[2])