# This file derives alignment constraint definitions from the CMSIS-NN project.
# CMSIS-NN is © Arm Limited, licensed under the Apache License, Version 2.0.
# See the NOTICE file in the project root for full attribution.

"""
CMSIS-NN Kernel Routing Constraints
====================================
This file encodes the EXACT preconditions that CMSIS-NN wrapper functions
use to select optimized kernels over fallback implementations.

Every constraint entry is traced to a specific source file and the routing
condition that gates the fast path. Where multiple backends exist (DSP, MVE,
generic), the MOST RESTRICTIVE common condition is recorded so that padding
guarantees the fast path on ALL supported cores.

Source files analyzed:
  - arm_convolve_wrapper_s8.c        (V.2.5.1, 04 Nov 2024)
  - arm_convolve_wrapper_s4.c        (V.1.2.0, 27 May 2024)
  - arm_convolve_wrapper_s16.c       (V.3.0.0, 23 Apr 2024)
  - arm_depthwise_conv_wrapper_s8.c  (V.2.2.1, 17 Jan 2025)
  - arm_depthwise_conv_wrapper_s4.c  (V.1.0.0, 30 Oct 2023)
  - arm_depthwise_conv_wrapper_s16.c (V.1.1.0, 20 Jan 2023)
  - arm_convolve_1_x_n_s8.c         (stride*ch % 4 constraint detail)
  - arm_convolve_1_x_n_s4.c         (stride*ch % 4 constraint detail)
  - arm_depthwise_conv_s8_opt.c      (inner loop structure)
  - arm_depthwise_conv_3x3_s8.c     (inner loop structure)
  - arm_depthwise_conv_fast_s16.c   (inner loop structure)
  - arm_depthwise_conv_s4_opt.c     (inner loop structure)
  - arm_fully_connected_wrapper_s8.c (col_dim % 4 constraint detail)

How to read this file
---------------------
Each op type maps to a dict of named constraints. Each constraint has:

  dim_index   : which axis of the weight/filter tensor the constraint acts on
                  0 = output_channels  (N dim of weight [out_ch, in_ch, kH, kW])
                  1 = input_channels   (C dim of weight [out_ch, in_ch, kH, kW])
                NOTE: DepthwiseConv weight shape is [out_ch, 1, kH, kW].
                  Axis 0 IS the channel count. There is no meaningful axis 1
                  (it is always 1 per group). Only axis 0 is constrained.
                NOTE: Gemm weight shape is [out_features, in_features] with transB=1.
                  Axis 1 is the input feature count (col_dim in CMSIS-NN).

  alignment   : value % alignment == 0 required for the fast path.
                None = no alignment requirement.

  patchable   : True  = can be fixed by zero-padding the weight tensor axis.
                False = depends on runtime params (stride, dilation, padding,
                        kernel size) that cannot be changed by static graph repair.

  kernel_path : which optimised kernel is unlocked when this condition holds.

  description : source citation with the EXACT condition from the C source.
"""

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _c(dim_index, alignment, patchable, kernel_path, description):
    return {
        "dim_index":   dim_index,
        "alignment":   alignment,
        "patchable":   patchable,
        "kernel_path": kernel_path,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Constraint tables — one entry per op variant per constraint
# ---------------------------------------------------------------------------

_CONSTRAINTS = {

    # =========================================================================
    # Gemm  (arm_fully_connected_wrapper_s8)
    # =========================================================================
    #
    # CMSIS-NN maps ONNX Gemm to arm_fully_connected_s8.
    # The optimized MAC loop in arm_fully_connected_s8.c processes
    # col_dim (input features) in groups of 4 via SMLAD/MVE.
    # When col_dim % 4 != 0 a scalar tail loop handles the remainder.
    #
    # Gemm weight shape with transB=1: [out_features, in_features]
    # Axis 1 = in_features = col_dim in the CMSIS-NN kernel.
    # Axis 0 = out_features — no alignment constraint, not patchable here.

    "Gemm": {

        "input_features": _c(
            dim_index   = 1,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_fully_connected_s8",
            description = (
                "arm_fully_connected_wrapper_s8.c: col_dim % 4 == 0 required "
                "for optimized SMLAD/MVE MAC loop. Gemm weight axis 1 = in_features "
                "= col_dim with transB=1."
            ),
        ),
    },

    # =========================================================================
    # Conv_s8  (arm_convolve_wrapper_s8)
    # =========================================================================
    #
    # WRAPPER ROUTING (arm_convolve_wrapper_s8.c):
    #
    # BRANCH A  →  arm_convolve_1x1_s8_fast  or  arm_convolve_1x1_s8
    #   if (padding.w == 0 && padding.h == 0
    #       && filter_dims->w == 1 && filter_dims->h == 1
    #       && dilation.w == 1 && dilation.h == 1
    #       && input_dims->c == filter_dims->c)
    #     stride == 1  →  arm_convolve_1x1_s8_fast
    #     stride != 1  →  arm_convolve_1x1_s8
    #
    # BRANCH B  →  arm_convolve_1_x_n_s8
    #   elif (input_dims->h == 1
    #         && dilation.w == 1
    #         && filter_dims->h == 1
    #         && (conv_params->stride.w * input_dims->c) % 4 == 0   ← KEY
    #         && input_dims->c == filter_dims->c)
    #
    # BRANCH C  →  arm_convolve_s8  (generic fallback)

    "Conv_s8": {

        "input_channels": _c(
            dim_index   = 1,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_1_x_n_s8",
            description = (
                "arm_convolve_wrapper_s8.c BRANCH B: "
                "(conv_params->stride.w * input_dims->c) % 4 == 0. "
                "For stride_w=1 this is input_channels % 4 == 0. "
                "arm_convolve_s8.c: 'optimal when channels are multiples of 4'."
            ),
        ),

        "output_channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_s8",
            description = (
                "arm_convolve_s8.c: 'Optimal use case for DSP/MVE implementation "
                "is when input and output channels are multiples of 4 or at least "
                "greater than 4.' Output channels = RHS row count in mat-mult kernels."
            ),
        ),
    },

    # =========================================================================
    # Conv_s4  (arm_convolve_wrapper_s4)
    # =========================================================================

    "Conv_s4": {

        "input_channels": _c(
            dim_index   = 1,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_1_x_n_s4",
            description = (
                "arm_convolve_wrapper_s4.c BRANCH B: "
                "(conv_params->stride.w * input_dims->c) % 4 == 0. "
                "Identical gate to s8 variant."
            ),
        ),

        "output_channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_s4",
            description = (
                "arm_convolve_s4.c: 'Optimal use case for DSP/MVE is when input "
                "and output channels are multiples of 4 or at least greater than 4.' "
                "Mirrors the s8 recommendation."
            ),
        ),

        "rhs_cols_even": _c(
            dim_index   = 1,
            alignment   = 2,
            patchable   = True,
            kernel_path = "arm_convolve_even_s4",
            description = (
                "arm_convolve_wrapper_s4.c BRANCH C (MVE only): "
                "((filter_dims->h * filter_dims->w * input_dims->c) & 0x1) == 0. "
                "arm_convolve_even_s4 returns ARG_ERROR if rhs_cols is odd. "
                "When kH*kW is odd, input_channels must be even to satisfy this. "
                "When kH*kW is even, constraint is always met regardless of channels."
            ),
        ),
    },

    # =========================================================================
    # Conv_s16  (arm_convolve_wrapper_s16)
    # =========================================================================

    "Conv_s16": {

        "input_channels": _c(
            dim_index   = 1,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_s16",
            description = (
                "arm_convolve_wrapper_s16.c: pass-through, no routing branch. "
                "arm_convolve_s16.c: 'optimal when channels are multiples of 4'. "
                "No hard gate — alignment=4 is a performance recommendation."
            ),
        ),

        "output_channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_s16",
            description = (
                "arm_convolve_s16.c: same 'multiples of 4' performance recommendation "
                "for output channels."
            ),
        ),
    },

    # =========================================================================
    # DepthwiseConv_s8  (arm_depthwise_conv_wrapper_s8)
    # =========================================================================

    "DepthwiseConv_s8": {

        "channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_depthwise_conv_s8_opt / arm_depthwise_conv_3x3_s8",
            description = (
                "arm_depthwise_conv_wrapper_s8.c optimised branch: ch_mult==1, n==1, "
                "dilation==1. "
                "arm_depthwise_conv_3x3_s8.c: inner loop 'for (; in_ch <= input_ch-4; in_ch+=4)'. "
                "arm_depthwise_conv_s8_opt.c (DSP): row_count = output_ch / 4, tail at output_ch & 0x3. "
                "channels % 4 == 0 eliminates scalar tail in both kernels. "
                "Weight shape [out_ch, 1, kH, kW]; ch_mult==1 means in_ch==out_ch, both at axis 0."
            ),
        ),
    },

    # =========================================================================
    # DepthwiseConv_s4  (arm_depthwise_conv_wrapper_s4)
    # =========================================================================

    "DepthwiseConv_s4": {

        "channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_depthwise_conv_s4_opt",
            description = (
                "arm_depthwise_conv_wrapper_s4.c optimised branch: ch_mult==1, n==1, "
                "dilation==1. "
                "arm_depthwise_conv_s4_opt.c: row_count = output_ch / 4 (DSP loop). "
                "channels % 4 == 0 eliminates scalar tail. "
                "Weight shape [out_ch, 1, kH, kW]; channels at axis 0."
            ),
        ),
    },

    # =========================================================================
    # DepthwiseConv_s16  (arm_depthwise_conv_wrapper_s16)
    # =========================================================================

    "DepthwiseConv_s16": {

        "channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_depthwise_conv_fast_s16",
            description = (
                "arm_depthwise_conv_wrapper_s16.c: fast path via USE_FAST_DW_CONV_S16_FUNCTION "
                "(ch_mult==1, DSP/MVE present). "
                "arm_depthwise_conv_fast_s16.c DSP loop: row_count = output_ch / 4, "
                "tail at output_ch & 0x3. MVE path: 4-wide int16 vectors. "
                "channels % 4 == 0 eliminates scalar tail. "
                "Weight shape [out_ch, 1, kH, kW]; channels at axis 0."
            ),
        ),
    },
}


# ---------------------------------------------------------------------------
# Backwards-compatible aliases — ONNX op type strings → s8 default variant
# ---------------------------------------------------------------------------

_CONSTRAINTS["Conv"]          = _CONSTRAINTS["Conv_s8"]
_CONSTRAINTS["DepthwiseConv"] = _CONSTRAINTS["DepthwiseConv_s8"]
_CONSTRAINTS["MaxPool"]       = {}
_CONSTRAINTS["AveragePool"]   = {}


# ---------------------------------------------------------------------------
# Public accessor API
# ---------------------------------------------------------------------------

def supported_ops():
    """Return list of op type keys with constraint entries."""
    return list(_CONSTRAINTS.keys())


def get_constraints(op_type):
    """Return all constraints for op_type. Empty dict if unknown."""
    return _CONSTRAINTS.get(op_type, {})


def get_patchable_constraints(op_type):
    """Return only constraints where patchable==True."""
    return {
        name: c
        for name, c in _CONSTRAINTS.get(op_type, {}).items()
        if c["patchable"]
    }


def get_non_patchable_constraints(op_type: str) -> dict:
    return {
        name: c
        for name, c in _CONSTRAINTS.get(op_type, {}).items()
        if not c.get("patchable", True)
    }


def get_alignment(op_type, constraint_name):
    """Return alignment integer for a constraint, or None if not found."""
    c = _CONSTRAINTS.get(op_type, {}).get(constraint_name)
    return c["alignment"] if c else None


def get_dim_index(op_type, constraint_name):
    """Return weight tensor axis index for a constraint, or None."""
    c = _CONSTRAINTS.get(op_type, {}).get(constraint_name)
    return c["dim_index"] if c else None


def get_max_value(op_type, constraint_name):
    """Upper bound for a constraint value. Currently None for all entries."""
    return None


def is_patchable(op_type, constraint_name):
    """True if this constraint can be fixed by zero-padding the weight axis."""
    c = _CONSTRAINTS.get(op_type, {}).get(constraint_name)
    return c["patchable"] if c else False


def describe(op_type, constraint_name):
    """Return the source citation string for a constraint."""
    c = _CONSTRAINTS.get(op_type, {}).get(constraint_name)
    return c["description"] if c else ""


def get_kernel_path(op_type, constraint_name):
    """Return the name of the optimised kernel gated by this constraint."""
    c = _CONSTRAINTS.get(op_type, {}).get(constraint_name)
    return c["kernel_path"] if c else ""


CONSTRAINTS = _CONSTRAINTS  # backwards-compatible alias


# ---------------------------------------------------------------------------
# Self-test / documentation dump
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for op in supported_ops():
        print(f"\n{'='*72}")
        print(f"  Op: {op}")
        print(f"{'='*72}")
        for name, c in get_constraints(op).items():
            print(f"\n  [{name}]")
            print(f"    dim_index  : {c['dim_index']}")
            print(f"    alignment  : {c['alignment']}")
            print(f"    patchable  : {c['patchable']}")
            print(f"    kernel     : {c['kernel_path']}")
            print(f"    source     : {c['description'][:100]}...")