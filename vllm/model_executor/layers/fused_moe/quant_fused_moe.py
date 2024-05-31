import torch
from vllm.model_executor.layers.fused_moe import (moe_align_block_size,
                                                  fused_topk,
                                                  get_config_file_name)
from typing import Optional, Dict, Any
import triton
import triton.language as tl
from vllm._C import ops
import os
import functools
from vllm.logger import init_logger
import json

logger = init_logger(__name__)


@functools.lru_cache
def get_moe_configs(E: int, N: int, num_bits: int=8) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(E, N, dtype=f"int{num_bits}")

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                f"Using configuration from {config_file_path} for MoE layer.")
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    return None


@triton.jit()
def fused_moe_kernel(
    # Pointers to matrices
    x_ptr,
    qweight_ptr,
    qscales_ptr,
    qzeros_ptr,
    g_idx_ptr,
    out_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_xm,
    stride_xk,
    stride_qe,
    stride_qk,
    stride_qn,  # strides for qweight
    stride_om,
    stride_on,
    stride_se,
    stride_ze,
    # Meta-parameters
    num_bits: int,
    maxq: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % EM
    offs_token_id = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M),
                                      BLOCK_SIZE_M)

    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens  # (BLOCK_SIZE_M, )

    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N),
                                BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Adjust offsets for packed tensors: qweight, qzero
    numel_per_i32 = 32 // num_bits
    q_offs_k = offs_k // numel_per_i32
    qweight_shifts = (offs_k % numel_per_i32) * num_bits
    qzero_shifts = (offs_bn % numel_per_i32) * num_bits

    x_ptrs = x_ptr + (offs_token[:, None] // top_k * stride_xm +
                      offs_k[None, :] * stride_xk)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    qweight_ptrs = qweight_ptr + off_experts * stride_qe + (
        q_offs_k[:, None] * stride_qk + offs_bn[None, :] * stride_qn)

    # qscales and qzeros only have a dependence on the n dimension,
    # so we only have to do a single I/O
    scales = tl.load(
        qscales_ptr + off_experts * stride_se + offs_bn,
        mask=offs_bn < N,
    )

    # TODO: this currently assumes that there is a single quantization group
    qzeros = tl.load(
        qzeros_ptr + stride_ze * off_experts + (offs_bn // numel_per_i32),
        mask=offs_bn < N,
    )
    zeros = (qzeros >> qzero_shifts) & maxq
    zeros = (zeros + 1) * scales

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        x = tl.load(x_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        qweights = tl.load(qweight_ptrs)

        # Dequantize
        weights = (
            qweights >> qweight_shifts[:, None]) & maxq  # bit shift qweight
        weights = scales * weights - zeros

        accumulator += tl.dot(x, weights)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        qweight_ptrs += BLOCK_SIZE_K // numel_per_i32 * stride_qk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_om * offs_token[:, None] + stride_on * offs_on[
        None, :]
    c_mask = token_mask[:, None] & (offs_on[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(x: torch.Tensor,
                            qweights: torch.Tensor,
                            qscales: torch.Tensor,
                            qzeros: torch.Tensor,
                            g_idx: torch.Tensor,
                            out: torch.Tensor,
                            topk_weights: torch.Tensor,
                            topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            num_bits: int,
                            config: Dict[str, Any],
                            debug: bool = False) -> None:

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    infeatures, outfeatures = g_idx.shape[1], qscales.shape[2]
    # Since we avoid masking, infeatures (the reduced K dimension) will
    # need to be a multiple of BLOCK_SIZE_K
    assert infeatures % config['BLOCK_SIZE_K'] == 0

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(outfeatures, META['BLOCK_SIZE_N']), )
    maxq = 2**num_bits - 1

    fused_moe_kernel[grid](
        x,
        qweights,
        qscales,
        qzeros,
        g_idx,
        out,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        outfeatures,
        infeatures,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        x.stride(0),
        x.stride(1),
        qweights.stride(0),
        qweights.stride(2),
        qweights.stride(1),
        out.stride(1),
        out.stride(2),
        qscales.stride(0),
        qzeros.stride(0),
        num_bits=num_bits,
        maxq=maxq,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16,
        **config,
    )


def fused_moe(
    hidden_states: torch.Tensor,
    qweights_1: torch.Tensor,
    qscales_1: torch.Tensor,
    qzeros_1: torch.Tensor,
    g_idx_1: torch.Tensor,
    qweights_2: torch.Tensor,
    qscales_2: torch.Tensor,
    qzeros_2: torch.Tensor,
    g_idx_2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_bits: int = 8,
    inplace: bool = True,
    override_config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")
    # assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert gating_output.shape[1] == qweights_1.shape[
        0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    # assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    # assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    # permute qweights
    qweights_1 = qweights_1.permute(0, 2, 1)  # (E, K, N)
    qweights_2 = qweights_2.permute(
        0, 2, 1)  # (E, N, K // packing_factor // tp_size)

    M, _ = hidden_states.shape
    E, N, _ = qweights_1.shape

    topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk, renormalize)
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        # We need to multiply by 2 here, since
        configs = get_moe_configs(E, qweights_2.shape[2] * 2, num_bits)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            # print("using...")
            config = configs[min(configs.keys(),
                                 key=lambda x: abs(int(x) - M))]
            # print(config)
        else:
            # Else use the default config
            config = {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }

            if M <= E:
                config = {
                    'BLOCK_SIZE_M': 16,
                    'BLOCK_SIZE_N': 32,
                    'BLOCK_SIZE_K': 64,
                    'GROUP_SIZE_M': 1
                }

    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty(
        (M, topk_ids.shape[1], qweights_2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config['BLOCK_SIZE_M'], E)

    invoke_fused_moe_kernel(hidden_states, qweights_1, qscales_1, qzeros_1,
                            g_idx_1, intermediate_cache1, topk_weights,
                            topk_ids, sorted_token_ids, expert_ids,
                            num_tokens_post_padded, False, topk_ids.shape[1],
                            num_bits, config)

    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    invoke_fused_moe_kernel(intermediate_cache2, qweights_2, qscales_2,
                            qzeros_2, g_idx_2, intermediate_cache3,
                            topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, True, 1,
                            num_bits, config)
    if inplace:
        return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                         dim=1,
                         out=hidden_states)
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                     dim=1)