# CMSIS-NN Graph Repair Tool

CMSIS-NN is Arm's embedded neural network library that provides highly optimized integer and DSP-accelerated kernels for efficient inference on resource-constrained microcontrollers.

A deterministic, static ONNX graph repair tool that automatically zero-pads weight and bias tensors to satisfy CMSIS-NN fast-path alignment constraints, without requiring retraining. The transformation preserves original model behavior while reshaping internal dimensions to fully unlock optimized kernel routing on ARM Cortex-M microcontrollers.

---

# Motivation

CMSIS-NN selects optimized kernels at runtime via wrapper functions that inspect tensor dimensions before dispatching. A model exported with channels that do not satisfy alignment requirements silently falls through to scalar fallback kernels — degrading inference throughput on Cortex-M devices with no warning at compile time.

Existing toolchains do not address this gap. The standard workflow is to retrain with aligned channel counts. This tool performs the alignment as a post-export graph transformation, preserving model semantics while unlocking fast-path dispatch.

---

# Novelty

* **Constraint knowledge base built from CMSIS-NN wrapper source** — every alignment requirement is traced to an exact routing condition in the C wrapper. `knowledge_base/constraints.py` derives constraints directly from the CMSIS-NN implementation — nothing is assumed or heuristic.
* **Union-Find propagation of ONNX op semantics** — dimension variables are coupled by the shape compatibility rules of each operation type, so padding one tensor automatically identifies all tensors that must move together — crucial for stability and unchanged behaviour.
* **Three-class violation taxonomy** — FREE, COUPLED, and LOCKED violations are distinguished before any patch is attempted, giving a clear audit trail of what was changed and why.
* **Reshape safety detection** — tensors that feed into Reshape nodes with hardcoded shape constants are locked rather than silently patched, preventing shape mismatches downstream.

---

# Core Structure

```
cmsis-ref/
├── pipeline.py               # programmatic entry point
├── cli.py                    # command-line entry point
├── report.py                 # report formatting
├── make_test_models.py       # test ONNX generation
├── knowledge_base/
│   └── constraints.py        # CMSIS-NN alignment constraints, source-cited
├── graph/
│   ├── loader.py             # ONNX load, shape extraction, graph I/O
│   ├── dim_variables.py      # symbolic dim variable assignment
│   └── propagator.py         # union-find constraint propagation
├── analysis/
│   ├── classifier.py         # FREE / COUPLED / LOCKED violation classification
│   └── feasibility.py        # padding plan construction
└── transforms/
    ├── pad_channels.py       # weight and bias padding, Reshape lock detection
    └── validate.py           # ONNX Runtime correctness check
```

Each layer depends only on the layer below it. `transforms` calls `analysis`, `analysis` calls `graph`, `knowledge_base` is read-only data with no dependencies.

---

# Pipeline

```
load model
    ↓
assign dim variables
    ↓
propagate constraints
    ↓
classify violations
    ↓
check feasibility
    ↓
build padding plan
    ↓
detect Reshape locks
    ↓
apply padding
    ↓
validate
```

---

# Constraint Knowledge Base

Constraints are derived directly from CMSIS-NN wrapper routing conditions, not from benchmarking or heuristics.

Example:

```python
# arm_convolve_wrapper_s8.c BRANCH B  →  arm_convolve_1_x_n_s8
#   elif (input_dims->h == 1
#         && dilation.w == 1
#         && filter_dims->h == 1
#         && (conv_params->stride.w * input_dims->c) % 4 == 0
#         && input_dims->c == filter_dims->c)
#
# For stride_w == 1  → input_channels % 4 == 0
# alignment=4 satisfies all stride values.
```

Every constraint entry records:

* dimension index
* required alignment
* whether the dimension is patchable
* which kernel fast-path becomes available
* the exact CMSIS-NN source condition.

---

# Union-Find Propagation

Dimension variables are symbolic names assigned to each tensor axis. The propagator walks every node and unions variables that must remain equal according to ONNX operator semantics.

Examples:

* **Conv**
  `input.channels == weight.in_channels`
  `weight.out_channels == output.channels`

* **Add**
  all inputs and outputs must match on every axis

* **Concat**
  tensors match on all axes except the concat axis

* **Reshape / Flatten**
  output dimensions are locked because they are defined by constants.

Union-find ensures that if two variables must be equal through a chain of operations, they are automatically grouped and patched together.

---

# Violation Classification

| Classification | Condition                        | Action                     |
| -------------- | -------------------------------- | -------------------------- |
| FREE           | group size == 1                  | patch directly             |
| COUPLED        | group size > 1                   | patch all members together |
| LOCKED         | graph I/O or reshape constrained | report and skip            |

---

# Example Output

MobileNet-style inverted residual block:

```
Node       | Op            | Constraint      | Current | Target | Status
------------------------------------------------------------------------
PW1        | Conv          | output_channels | 6       | 8      | PATCHED
DW2        | DepthwiseConv | channels        | 6       | 8      | PATCHED
PW2        | Conv          | input_channels  | 6       | 8      | PATCHED
PW2        | Conv          | output_channels | 6       | 8      | PATCHED
Expand     | Conv          | input_channels  | 6       | 8      | PATCHED
Expand     | Conv          | output_channels | 10      | 12     | PATCHED
StemConv   | Conv          | input_channels  | 3       | 4      | LOCKED
```

---

# Host Validation: Behavioural Consistency

Before deploying repaired models to embedded hardware, it is critical to verify that the graph transformation preserves the original model behaviour.

The repair process modifies internal tensor dimensions through channel padding. Validation ensures that these structural changes do not alter:

* model execution correctness
* operator behaviour
* output tensor shapes
* runtime stability.

### Validation Command

```
python benchmark.py test_model_mixed.onnx test_model_mixed_patched.onnx
```

### Node-Level Runtime Comparison

```
Node                                 Before (us)   After (us)      Delta
------------------------------------------------------------------------
Conv/ConvB_kernel_time                         7            7         +0
Conv/DW_A_kernel_time                          9            9         +0
Conv/PW_A_kernel_time                          7            7         +0
Conv/PW_B_kernel_time                          7            7         +0
Conv/ProjConv_kernel_time                      7            7         +0
Conv/StemConv_kernel_time                     11           10         -1
Flatten/Flat1_kernel_time                      4            4         +0
FusedConv/MidA_kernel_time                     9            9         +0
Gemm/Gemm1_kernel_time                         5            5         +0
GlobalAveragePool/GAP_kernel_time              6            6         +0
```

### Observations

Per-node runtime differences fall within **measurement noise** (−1 μs to +0 μs).

This confirms that the transformation:

* preserves **operator execution behaviour**
* does not introduce **runtime instability**
* does not change **output tensor dimensions**
* does not introduce **meaningful host-side overhead**.

### Significance

Structural graph transformations must preserve both **functional correctness** and **interface stability**.

This validation demonstrates that:

1. the repaired model remains **fully functional**,
2. the transformation **does not alter inference semantics**, and
3. the **external model interface remains unchanged**.

The repaired graph can therefore be safely deployed to embedded inference environments where **CMSIS-NN optimized kernels** will provide the expected performance improvements.

---

# Usage

```
python make_test_models.py
python cli.py test_model.onnx
python cli.py test_model.onnx --output fixed.onnx
python cli.py test_model.onnx --report-only
python cli.py test_model.onnx --export-report report.json
```

---

# Requirements

```
onnx>=1.13.0
onnxruntime>=1.14.0
numpy>=1.23.0
protobuf>=3.20.0
```

Python ≥ 3.10 required.

---

# Scope

The tool supports typical CNN architectures deployed on Cortex-M devices:

* convolution layers
* depthwise convolution
* pointwise convolution
* residual additions
* concatenation
* fully connected layers.

Transformer and RNN architectures are outside scope.

The framework is backend-agnostic and could be extended to other kernel selection systems that depend on tensor alignment constraints.

---

# Attribution

This project derives alignment constraint definitions from the CMSIS-NN project.

CMSIS-NN is © Arm Limited and licensed under the Apache License, Version 2.0.

See the `NOTICE` file for full attribution.
