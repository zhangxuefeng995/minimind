"""可选的 KDA Triton 加速（源码是 Python，运行时编译成 GPU kernel）。

Triton 的 kernel 用 Python DSL 写，**不是**再引入一套 C++/CUDA 工程；
但 ``import triton`` 之后会在 GPU 上 JIT，CPU / 没装 Triton 时绝不能变成硬依赖。

本模块：
- 缺 Triton、不是 CUDA、head_dim 过大时全部返回 ``None``，由 ``k3_ops`` 回退 PyTorch；
- 实现的是与 naive 递推等价的 fused recurrent（把 token 循环搬进 GPU），
  不是 FLA 那套完整 WY/UT Tensor Core kernel。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # CPU、精简环境、未 pip install triton
    triton = None
    tl = None
    HAS_TRITON = False

# SRAM 里要放 D×D 的状态；过大就回退，避免 shared memory 爆掉
_MAX_HEAD_DIM = 128


def kda_triton_is_available(x: Optional[torch.Tensor] = None) -> bool:
    if not HAS_TRITON:
        return False
    if x is None:
        return True
    return bool(x.is_cuda) and x.shape[-1] <= _MAX_HEAD_DIM


if HAS_TRITON:

    @triton.jit
    def _kda_recurrent_kernel(
        q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, state_ptr, out_ptr, state_out_ptr, mask_ptr,
        T, D, H,
        stride_b, stride_h, stride_t, stride_d,
        stride_sb, stride_sh, stride_sd1, stride_sd2,
        stride_bb, stride_bh, stride_bt,
        stride_mb, stride_mt,
        HAS_MASK: tl.constexpr,
        HAS_INIT: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        offs = tl.arange(0, BLOCK_D)
        mask_d = offs < D

        if HAS_INIT:
            state = tl.load(
                state_ptr + b * stride_sb + h * stride_sh
                + offs[:, None] * stride_sd1 + offs[None, :] * stride_sd2,
                mask=mask_d[:, None] & mask_d[None, :],
                other=0.0,
            )
        else:
            state = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

        vec_base = b * stride_b + h * stride_h
        for t in range(T):
            t_off = vec_base + t * stride_t
            kt = tl.load(k_ptr + t_off + offs * stride_d, mask=mask_d, other=0.0).to(tl.float32)
            vt = tl.load(v_ptr + t_off + offs * stride_d, mask=mask_d, other=0.0).to(tl.float32)
            qt = tl.load(q_ptr + t_off + offs * stride_d, mask=mask_d, other=0.0).to(tl.float32)
            at = tl.load(alpha_ptr + t_off + offs * stride_d, mask=mask_d, other=1.0).to(tl.float32)
            bt = tl.load(beta_ptr + b * stride_bb + h * stride_bh + t * stride_bt).to(tl.float32)
            prev = state
            state = state * at[:, None]
            k_state = tl.sum(kt[:, None] * state, axis=0)
            state = state - bt * kt[:, None] * k_state[None, :]
            state = state + bt * kt[:, None] * vt[None, :]
            if HAS_MASK:
                keep = tl.load(mask_ptr + b * stride_mb + t * stride_mt).to(tl.float32)
                state = tl.where(keep > 0, state, prev)
            ot = tl.sum(qt[:, None] * state, axis=0)
            tl.store(out_ptr + t_off + offs * stride_d, ot, mask=mask_d)

        tl.store(
            state_out_ptr + b * stride_sb + h * stride_sh
            + offs[:, None] * stride_sd1 + offs[None, :] * stride_sd2,
            state,
            mask=mask_d[:, None] & mask_d[None, :],
        )


def kda_delta_rule_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """与 naive 递推同公式的 GPU fused loop；不可用时返回 None 让调用方回退。"""
    if not kda_triton_is_available(q):
        return None
    orig_dtype = q.dtype
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    alpha = torch.exp(log_decay.float()).contiguous()
    beta = beta.float().contiguous()
    bsz, n_heads, seq_len, dim = q.shape
    q32, k32, v32 = q.float(), k.float(), v.float()
    out = torch.empty_like(q32)
    state_out = q32.new_empty(bsz, n_heads, dim, dim)
    has_init = initial_state is not None
    if has_init:
        init_state = initial_state.float().contiguous()
    else:
        init_state = state_out  # 占位，HAS_INIT=False 时 kernel 不读
    mask = None
    has_mask = attention_mask is not None
    if has_mask:
        mask = attention_mask.to(dtype=q32.dtype, device=q.device).contiguous()

    block_d = triton.next_power_of_2(dim)
    _kda_recurrent_kernel[(bsz * n_heads,)](
        q32, k32, v32, alpha, beta, init_state, out, state_out, mask if has_mask else q32,
        seq_len, dim, n_heads,
        q32.stride(0), q32.stride(1), q32.stride(2), q32.stride(3),
        state_out.stride(0), state_out.stride(1), state_out.stride(2), state_out.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        mask.stride(0) if has_mask else 0, mask.stride(1) if has_mask else 0,
        HAS_MASK=has_mask,
        HAS_INIT=has_init,
        BLOCK_D=block_d,
    )
    return out.to(orig_dtype), state_out.to(orig_dtype)
