"""Kimi-K3 相关算子：KDA 的 WY/UT 分块扫描、Context Parallel 模拟、MXFP QAT。

本文件是 MiniMind 规模的纯 PyTorch 实现，对应 Kimi Linear / K3 论文中的算法骨架，
而不是 2.8T 训练栈里的 Triton/CUDA kernel、NCCL 1M Context Parallel 或真实 MX 硬件量化。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def kda_delta_rule_scan_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 token 递推的 KDA（用于解码一步、数值对照）。

    论文公式（Kimi Linear / K3）::

        S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
        o_t = S_t^T q_t

    其中 ``log_decay`` 已是通道级遗忘门的对数域值，``α_t = exp(log_decay_t) ∈ (0, 1]``。
    ``attention_mask==0`` 的位置不更新 S（保持上一时刻状态）。
    """
    orig_dtype = q.dtype
    bsz, n_heads, seq_len, dim = q.shape
    q, k, v = q.float(), k.float(), v.float()
    alpha = torch.exp(log_decay.float())
    beta = beta.float()
    state = initial_state.float() if initial_state is not None else q.new_zeros(bsz, n_heads, dim, dim)
    outputs = q.new_empty(bsz, n_heads, seq_len, dim)
    for t in range(seq_len):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        # 先按通道遗忘：Diag(α) S
        new_state = state * alpha[:, :, t].unsqueeze(-1)
        # Householder / delta rule：减去 β k (k^T (α S))，再写入 β k v^T
        k_state = torch.einsum('bhd,bhde->bhe', kt, new_state)
        b_t = beta[:, :, t]
        new_state = new_state - b_t[..., None, None] * torch.einsum('bhd,bhe->bhde', kt, k_state)
        new_state = new_state + b_t[..., None, None] * torch.einsum('bhd,bhe->bhde', kt, vt)
        if attention_mask is not None:
            keep = attention_mask[:, t].view(bsz, 1, 1, 1).to(dtype=new_state.dtype)
            new_state = keep * new_state + (1.0 - keep) * state
        state = new_state
        outputs[:, :, t] = torch.einsum('bhd,bhde->bhe', qt, state)
    return outputs.to(orig_dtype), state.to(orig_dtype)


def _prepare_kda_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 padding 位置改写成「零写入 + α=1」的恒等转移，便于走矩阵形式的 WY。

    不改 attention_mask 本身：WY 路径无法像 Python 循环那样「跳过」某步，
    但 β=0 且 α=1 时 S_t = S_{t-1}，与 naive 扫描在有效 token 上等价。
    """
    if attention_mask is None:
        return q, k, v, log_decay, beta
    keep = attention_mask.to(dtype=q.dtype)
    while keep.dim() < q.dim() - 1:
        keep = keep.unsqueeze(1)
    # keep: [B, T] -> 广播到 [B, 1, T, 1] / [B, 1, T]
    keep_qkv = keep[:, None, :, None]
    keep_beta = keep[:, None, :]
    q = q * keep_qkv
    k = k * keep_qkv
    v = v * keep_qkv
    beta = beta * keep_beta
    log_decay = log_decay * keep_qkv
    return q, k, v, log_decay, beta


def kda_delta_rule_scan_wy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kimi Linear §3.1 的分块 WY / UT 扫描，与 :func:`kda_delta_rule_scan_naive` 数值对齐。

    序列按 ``chunk_size`` 切开：

    1. **块内（intra-chunk）**：用 UT 变换把一串 rank-1 Householder 压成下三角系统
       ``M = (I + StrictTril(...))^{-1} Diag(β)``，再得到辅助矩阵 ``W, U``。
       输出为论文 (9) 式::

           O = (Γ ⊙ Q) S_0 + Tril((Γ ⊙ Q)(K / Γ)^T) (U - W S_0)

       其中 ``Γ^r = ∏_{i=1}^{r} α^i`` 是块内通道级累积衰减。
    2. **块间（inter-chunk）**：用论文 (8) 的紧凑状态更新，左因子取
       ``K ⊙ (γ^C / γ^r)``（从当前 token **之后** 乘到块尾的衰减），
       这才能与 ``(I - βkk^T) Diag(α)`` 的递推定义一致。

    说明：官方 kernel 在 Triton 里做二次分块、exp2 和 Tensor Core 融合；
    这里用 ``torch.linalg.solve_triangular`` 表达同一套代数，便于 CPU/小模型训练。
    """
    orig_dtype = q.dtype
    q, k, v, log_decay, beta = _prepare_kda_mask(q, k, v, log_decay, beta, attention_mask)
    q, k, v = q.float(), k.float(), v.float()
    log_g = log_decay.float().clamp(min=-40.0)
    beta = beta.float()
    bsz, n_heads, seq_len, dim = q.shape
    chunk = max(int(chunk_size), 1)
    pad = (chunk - seq_len % chunk) % chunk
    if pad:
        # 右侧 padding：q=k=v=0、β=0、log α=0 ⇒ α=1，状态在 pad 步保持不变
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        log_g = F.pad(log_g, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
    pad_len = q.shape[2]
    n_chunk = pad_len // chunk
    q = q.view(bsz, n_heads, n_chunk, chunk, dim)
    k = k.view(bsz, n_heads, n_chunk, chunk, dim)
    v = v.view(bsz, n_heads, n_chunk, chunk, dim)
    log_g = log_g.view(bsz, n_heads, n_chunk, chunk, dim)
    beta = beta.view(bsz, n_heads, n_chunk, chunk)

    state = initial_state.float() if initial_state is not None else q.new_zeros(bsz, n_heads, dim, dim)
    outputs = []
    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    strict_tril = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=torch.bool), diagonal=-1)
    incl_tril = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=torch.bool), diagonal=0)

    for ci in range(n_chunk):
        qn, kn, vn, gn, bn = q[:, :, ci], k[:, :, ci], v[:, :, ci], log_g[:, :, ci], beta[:, :, ci]
        # γ^r = ∏_{i≤r} α^i，在 log 域 cumsum 后 exp，避免连乘下溢
        log_gamma = gn.cumsum(dim=-2).clamp(min=-40.0)
        gamma = log_gamma.exp()
        k_scaled = kn * gamma          # Γ ⊙ K
        k_div = kn * (-log_gamma).exp()  # K / Γ（按通道，log 域完成除法）
        # 论文 (6)：StrictTril( Diag(β) (Γ⊙K)(K/Γ)^T )
        gram = torch.matmul(k_scaled, k_div.transpose(-1, -2))
        lower = (bn.unsqueeze(-1) * gram).masked_fill(~strict_tril, 0)
        # M = (I+L)^{-1} Diag(β)；I+L 是单位下三角，前代即可
        m_mat = torch.linalg.solve_triangular(eye + lower, torch.diag_embed(bn), upper=False)
        w_mat = torch.matmul(m_mat, k_scaled)  # 论文 (7) 的 W
        u_mat = torch.matmul(m_mat, vn)        # 论文 (7) 的 U
        log_g_tail = log_gamma[:, :, -1, :]
        # 从 r+1 乘到块尾：γ^C / γ^r ；r=C 时为 1
        gamma_after = (log_g_tail.unsqueeze(-2) - log_gamma).clamp(-40.0, 40.0).exp()
        k_left = kn * gamma_after
        state_in = state
        # 「伪 Value」：U - W S，同时用于块内输出和块间写状态
        pseudo = u_mat - torch.matmul(w_mat, state_in)
        q_scaled = qn * gamma
        # 论文 (9)：块间读旧状态 + 块内下三角「伪注意力」
        inter = torch.matmul(q_scaled, state_in)
        intra = torch.matmul(q_scaled, k_div.transpose(-1, -2)).masked_fill(~incl_tril, 0)
        outputs.append(inter + torch.matmul(intra, pseudo))
        # 论文 (8) 的状态推进；左因子必须是 K⊙(γ^C/γ^r) 才能对齐 naive 递推
        state = state_in * log_g_tail.exp().unsqueeze(-1) + torch.matmul(k_left.transpose(-1, -2), pseudo)

    output = torch.cat(outputs, dim=2)[:, :, :seq_len]
    return output.to(orig_dtype), state.to(orig_dtype)


def kda_delta_rule_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
    use_wy_scan: bool = True,
    context_parallel_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KDA 扫描入口。

    - ``seq_len==1``：走逐步递推（解码 / KV-RNN 状态）。
    - ``use_wy_scan``：训练/预填使用 WY/UT 分块，避免 Python 扫完整 T。
    - ``context_parallel_size>1``：把序列切成若干段，**段间顺序传递 S**。
      这是 K3 在多卡 Context Parallel（1M 上下文、NCCL 传状态）上的单卡等价物：
      代数上与一次扫完整序列相同，并不实现跨卡 all-to-all。
    """
    seq_len = q.shape[2]
    if seq_len == 1 or not use_wy_scan:
        return kda_delta_rule_scan_naive(
            q, k, v, log_decay, beta,
            initial_state=initial_state,
            attention_mask=attention_mask,
        )

    cp = max(int(context_parallel_size), 1)
    if cp <= 1 or cp >= seq_len:
        return kda_delta_rule_scan_wy(
            q, k, v, log_decay, beta,
            initial_state=initial_state,
            attention_mask=attention_mask,
            chunk_size=chunk_size,
        )

    # 模拟 CP：每段长度 ≈ cp，段尾状态交给下一段（多卡时这里会是 P2P send）
    outputs = []
    state = initial_state
    for t0 in range(0, seq_len, cp):
        t1 = min(t0 + cp, seq_len)
        mask_seg = attention_mask[:, t0:t1] if attention_mask is not None else None
        out_seg, state = kda_delta_rule_scan_wy(
            q[:, :, t0:t1], k[:, :, t0:t1], v[:, :, t0:t1],
            log_decay[:, :, t0:t1], beta[:, :, t0:t1],
            initial_state=state,
            attention_mask=mask_seg,
            chunk_size=chunk_size,
        )
        outputs.append(out_seg)
    return torch.cat(outputs, dim=2), state


def fake_mxfp_quant(x: torch.Tensor, bits: int, block_size: int = 32) -> torch.Tensor:
    """MX 风格的块缩放伪量化（STE），用于 QAT，不是真实 MXFP4/MXFP8 硬件格式。

    K3 只对 **routed expert 的权重（约 4bit）和激活（约 8bit）** 做量化，
    注意力、共享专家、路由器保持高精度。这里用「每 block 共享 scale + 对称整数格子」近似：

    - 4bit：量化到 [-7, 7]（16 档里去掉一个非对称点，便于 STE）
    - 8bit：量化到 [-127, 127]，接近 E4M3 的动态范围使用方式

    反传走 Straight-Through Estimator：前向用量化值，反向梯度当作恒等。
    """
    if bits <= 0 or block_size <= 0:
        return x
    qmax = 7.0 if int(bits) <= 4 else 127.0
    orig_shape = x.shape
    last = orig_shape[-1]
    pad = (block_size - last % block_size) % block_size
    xp = F.pad(x, (0, pad)) if pad else x
    blocked = xp.reshape(*xp.shape[:-1], -1, block_size)
    # scale 不回传，避免量化阈值抖动
    amax = blocked.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / qmax
    q = (blocked / scale).round().clamp(-qmax, qmax)
    y = q * scale
    y = y.reshape(*xp.shape[:-1], -1)[..., :last].reshape(orig_shape)
    return x + (y - x).detach()


def qat_linear(
    linear: torch.nn.Linear,
    x: torch.Tensor,
    quant_weight: bool,
    quant_act: bool,
    weight_bits: int = 4,
    act_bits: int = 8,
    block_size: int = 32,
) -> torch.Tensor:
    """带 MX 伪量化的 ``Linear``：只应包在 routed expert 上。"""
    weight = linear.weight
    if quant_weight:
        weight = fake_mxfp_quant(weight, bits=weight_bits, block_size=block_size)
    if quant_act:
        x = fake_mxfp_quant(x, bits=act_bits, block_size=block_size)
    return F.linear(x, weight, linear.bias)
