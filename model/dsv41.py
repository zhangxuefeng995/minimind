"""DeepSeek-V4.1-Flash 在 MiniMind 尺度上的架构组件。

对应论文 / 模型卡中的主要技术（均为教学级精简实现，不是 552B 原版）：

1. **CED（Causal Encoder-Decoder）**
   层数对半：前半因果编码器、后半解码器。解码器 Full 层的全局 KV 从
   **编码器最后一层隐状态**投影，而不是从本层 hidden 再算一遍。
   Prefill 只需跑编码器（外加解码器窗口内的 SWA）；Decode 才编解码一起跑。

2. **CSA2（Compressed Sparse Attention 2）**
   每层静态一种模式：Full / Reindex / Reuse。
   - Full：算全局 KV + indexer K + 新的 Top-K
   - Reindex：复用最近 Full 的全局 KV / indexer K，用本层 indexer Q 重打分
   - Reuse：连 Top-K 下标一起复用，不再跑 indexer
   所有模式都保留本层 **Q + 局部 SWA KV**。压缩比：编码器 CSA2=2（非重叠池化），
   解码器=1（不压缩，CSA 的特例）。CSA2 相对 CSA 去掉了重叠压缩与压缩端绝对位置编码，
   indexer K 改为从 main KV 投影。

3. **Hierarchical Sparse Indexer（仅解码器）**
   解码器第一个 Full 层对全上下文打分，再按 block 取 max 构成候选池；
   后续 Reindex 只在池内选 Top-K，打分代价与上下文长度解耦。

4. **SWA Bounded Replay**
   局部滑动窗口 KV 不持久化整段历史，推理时只保留最近 ``sliding_window`` 个 token。

5. **Single-Pass mHC**
   多残差流混合。输入混合系数错位一层（用上一层的 γ），从而残差更新、
   输入混合、系数预测可以在一次 hidden 遍历里做完。

6. **Engram**
   词表压缩 + 多头 n-gram 哈希查表 + 上下文门控。V4.1 去掉了短因果卷积。

7. **FP4 主 KV（E2M1 + 每 16 通道一个 scale）**
   训练可用 STE 伪量化；SWA KV 保持高精度。

8. **DeepSeekMoE（无辅助损失）**
   sigmoid 分数 + 无梯度 expert bias 做负载均衡（noaux_tc 风格）。

9. **DSpark**
   论文在骨干预训练之后单独训练投机解码头，本文件不实现草稿网络。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# E2M1 可表示的幅度（含符号由调用方处理）：0, 0.5, 1, 1.5, 2, 3, 4, 6
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, T, H, D]，cos/sin: [T, D]。"""
    def rotate_half(t):
        return torch.cat((-t[..., t.shape[-1] // 2:], t[..., : t.shape[-1] // 2]), dim=-1)

    if cos.dim() == 2:
        cos = cos.unsqueeze(1)  # [T, 1, D] 与 [B, T, H, D] 右对齐广播
        sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def compress_tokens(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """非重叠池化压缩序列维。ratio<=1 时原样返回。x: [B, T, ...]。"""
    if ratio is None or ratio <= 1:
        return x
    bsz, seqlen = x.shape[:2]
    pad = (ratio - seqlen % ratio) % ratio
    if pad:
        pad_spec = (0, 0) * (x.dim() - 2) + (0, pad)
        x = F.pad(x, pad_spec)
        seqlen = x.shape[1]
    new_len = seqlen // ratio
    rest = x.shape[2:]
    x = x.view(bsz, new_len, ratio, *rest).mean(dim=2)
    return x


def compressed_end_positions(seq_len: int, ratio: int, device) -> torch.Tensor:
    """每个压缩槽对应的最后一个原始 token 下标（因果边界）。"""
    if ratio is None or ratio <= 1:
        return torch.arange(seq_len, device=device)
    n = (seq_len + ratio - 1) // ratio
    return (torch.arange(n, device=device) + 1) * ratio - 1


def fake_quant_fp4_e2m1(x: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """主 KV 的 MX/NVFP4 风格伪量化：E2M1 尾数 + 每 group_size 通道一个 scale。

    推理时等价于先量化再反量化；训练时用 STE 让梯度穿过。
    """
    if x.numel() == 0:
        return x
    orig_shape = x.shape
    last = orig_shape[-1]
    pad = (group_size - last % group_size) % group_size
    if pad:
        x = F.pad(x, (0, pad))
    grouped = x.view(*x.shape[:-1], -1, group_size)
    # 尺度取组内最大幅度 / 6，避免溢出 E2M1 的 6
    amax = grouped.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = amax / 6.0
    y = grouped / scale
    levels = _E2M1_LEVELS.to(device=x.device, dtype=x.dtype)
    # 最近邻量化到 E2M1 幅度，再恢复符号
    mag = y.abs()
    idx = (mag.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    qmag = levels[idx]
    q = qmag * y.sign()
    deq = q * scale
    deq = deq.view(*x.shape)[..., :last].view(orig_shape)
    # STE
    return x + (deq - x).detach()


def lightning_index_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, head_w: torch.Tensor) -> torch.Tensor:
    """DSA/CSA 闪电 indexer：I[t,s] = Σ_h w_h · ReLU(q[t,h] · k[s,h])。

    q_idx: [B, T, Hi, Di]，k_idx: [B, S, Hi, Di] → [B, T, S]
    """
    dots = torch.einsum('bthd,bshd->btsh', q_idx, k_idx)
    return torch.relu(dots).mul(head_w.view(1, 1, 1, -1)).sum(dim=-1)


def build_candidate_pool(scores: torch.Tensor, block_size: int, top_blocks: int,
                         valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """按 block 内最大 indexer 分选块，展开为候选位置下标。scores: [B, T, S]。"""
    bsz, qlen, slen = scores.shape
    if valid is not None:
        if valid.shape[0] == 1 and bsz > 1:
            valid = valid.expand(bsz, -1, -1)
        scores = scores.masked_fill(~valid, float('-inf'))
    pad = (block_size - slen % block_size) % block_size
    if pad:
        scores = F.pad(scores, (0, pad), value=float('-inf'))
    n_blocks = scores.shape[-1] // block_size
    blocked = scores.view(bsz, qlen, n_blocks, block_size)
    block_score = blocked.amax(dim=-1)  # [B, T, n_blocks]
    k_blocks = min(top_blocks, n_blocks)
    _, block_idx = torch.topk(block_score, k=k_blocks, dim=-1, sorted=False)  # [B, T, k_blocks]
    offsets = torch.arange(block_size, device=scores.device).view(1, 1, 1, block_size)
    pool = block_idx.unsqueeze(-1) * block_size + offsets  # [B, T, k_blocks, bs]
    pool = pool.view(bsz, qlen, -1)
    pool = pool.clamp(max=slen - 1)
    return pool


def gather_by_index(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """src: [B, S, ...]，index: [B, T, K] → [B, T, K, ...]。"""
    extra = src.shape[2:]
    bsz, slen = src.shape[:2]
    tlen, klen = index.shape[1], index.shape[2]
    idx = index.clamp(0, max(slen - 1, 0))
    expand_src = src.unsqueeze(1).expand(bsz, tlen, slen, *extra)
    gather_idx = idx.view(bsz, tlen, klen, *([1] * len(extra))).expand(bsz, tlen, klen, *extra)
    return torch.gather(expand_src, 2, gather_idx)


def build_dsv41_layout(n_layers: int) -> Dict:
    """按 DeepSeek-V4.1-Flash 的 40 层布局缩放到任意偶数层。

    官方（40 层）：层 0–1 仅 SWA；编码器 CSA2 压缩比 2；解码器压缩比 1；
    kv_source = [2, 8, 14, 20]；index_source 再加解码器每隔 4 层的 Reindex；
    candidate_source = 20（解码器第一个 Full）。
    """
    n_layers = int(n_layers)
    n_enc = max(n_layers // 2, 1)
    n_swa = 2 if n_enc >= 2 else 0
    if n_layers < 4:
        n_swa = 0

    compress = []
    modes: List[str] = []
    for i in range(n_layers):
        if i < n_swa:
            compress.append(0)
            modes.append('swa')
        elif i < n_enc:
            compress.append(2)
            modes.append('reuse')
        else:
            compress.append(1)
            modes.append('reuse')

    kv_sources = list(range(n_swa, n_enc, 6))
    if n_swa < n_enc and not kv_sources:
        kv_sources = [n_swa]
    if n_enc < n_layers:
        kv_sources.append(n_enc)

    n_dec = n_layers - n_enc
    dec_stride = 4 if n_dec >= 8 else 2
    index_sources = list(kv_sources)
    for i in range(n_enc + dec_stride, n_layers, dec_stride):
        if i not in index_sources:
            index_sources.append(i)

    kv_set, idx_set = set(kv_sources), set(index_sources)
    for i in range(n_layers):
        if modes[i] == 'swa':
            continue
        if i in kv_set:
            modes[i] = 'full'
        elif i in idx_set:
            modes[i] = 'reindex'
        else:
            modes[i] = 'reuse'

    candidate_source = n_enc if n_enc < n_layers else None
    return {
        'n_enc': n_enc,
        'n_swa': n_swa,
        'compress_ratios': compress,
        'modes': modes,
        'kv_source_layer_ids': kv_sources,
        'index_source_layer_ids': index_sources,
        'candidate_source_layer_id': candidate_source,
    }


class MHCPredictor(nn.Module):
    """Single-Pass mHC 的系数头 H：从当前残差流预测 (α, β, γ)。

    论文把输入混合从 γ_ℓ(x_ℓ) 改成 γ_{ℓ-1}(x_ℓ)，打破「先算系数再混合」的依赖，
    从而 Mega-mHC 只需一次读写残差。
    """

    def __init__(self, hidden_size: int, n_stream: int, eps: float = 1e-5):
        super().__init__()
        self.n_stream = n_stream
        self.norm = _RMSNorm(hidden_size, eps)
        # γ: S×S 混合，α/β: 每个流一个缩放
        self.proj = nn.Linear(hidden_size, n_stream * n_stream + 2 * n_stream, bias=False)

    def forward(self, streams: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # streams: [B, T, S, H]
        pooled = streams.mean(dim=2)
        coef = self.proj(self.norm(pooled))
        s = self.n_stream
        gamma = coef[..., : s * s].view(*coef.shape[:-1], s, s)
        alpha = coef[..., s * s: s * s + s].unsqueeze(-1)
        beta = coef[..., s * s + s:].unsqueeze(-1)
        gamma = torch.softmax(gamma, dim=-1)
        alpha = torch.sigmoid(alpha)
        beta = torch.sigmoid(beta)
        return alpha, beta, gamma


def mhc_mix_streams(streams: torch.Tensor, gamma: Optional[torch.Tensor]) -> torch.Tensor:
    """用上一层的 γ 混合残差流。gamma 为 None 时（第 0 层）恒等。"""
    if gamma is None:
        return streams
    # streams [B,T,S,H]，gamma [B,T,S,S]：new[i] = Σ_j γ[i,j] old[j]
    return torch.einsum('btij,btjh->btih', gamma, streams)


class EngramMemory(nn.Module):
    """Engram 条件记忆：tokenizer 压缩 + 多头哈希 + 上下文门控。

    MiniMind 用很小的表（默认每头 1024 项），n-gram 阶 {2,3}。官方还有 4-gram、
    约 16M 项/头、FP8 表项以及 Sinkhorn 优化器，这里只保留查表与门控路径。
    """

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.n_heads = int(getattr(config, 'engram_n_heads', 2))
        self.head_dim = int(getattr(config, 'engram_head_dim', 32))
        self.table_size = int(getattr(config, 'engram_table_size', 1024))
        self.max_ngram = int(getattr(config, 'engram_max_ngram_size', 3))
        self.pad_id = int(getattr(config, 'engram_pad_token_id', getattr(config, 'eos_token_id', 2)))
        orders = list(range(2, self.max_ngram + 1))
        self.orders = orders
        # 每 (阶, 头) 一张表；表大小用互质偏移避免对齐碰撞
        self.tables = nn.ModuleList([
            nn.Embedding(self.table_size, self.head_dim)
            for _ in orders for _ in range(self.n_heads)
        ])
        out_dim = len(orders) * self.n_heads * self.head_dim
        self.out_proj = nn.Linear(out_dim, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden * 2, hidden, bias=True)
        # 头相关的哈希乘数（固定缓冲，确定性寻址，便于预取）
        multipliers = []
        for oi, _ in enumerate(orders):
            for h in range(self.n_heads):
                multipliers.append(int(1_000_003 + 7919 * (oi * self.n_heads + h + 1)))
        self.register_buffer('hash_mul', torch.tensor(multipliers, dtype=torch.long), persistent=False)

    def _ngram_ids(self, input_ids: torch.Tensor, order: int) -> torch.Tensor:
        # 因果 n-gram：位置 t 用 token[t-order+1 : t]
        padded = F.pad(input_ids, (order - 1, 0), value=self.pad_id)
        acc = torch.zeros_like(input_ids, dtype=torch.long)
        for k in range(order):
            acc = acc * 16777619 + padded[:, k: k + input_ids.size(1)].long()
        return acc

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        pieces = []
        tbl_i = 0
        for order in self.orders:
            ngram = self._ngram_ids(input_ids, order)
            for _h in range(self.n_heads):
                mul = int(self.hash_mul[tbl_i].item())
                hashed = (ngram * mul).remainder(self.table_size)
                pieces.append(self.tables[tbl_i](hashed))
                tbl_i += 1
        retrieved = torch.cat(pieces, dim=-1)
        retrieved = self.out_proj(retrieved)
        retrieved = retrieved[:, -hidden.size(1):]
        gate = torch.sigmoid(self.gate_proj(torch.cat([hidden, retrieved], dim=-1)))
        return gate * retrieved


class CEDDecoderKV(nn.Module):
    """CED 解码器 Full 的全局 KV：编码器出口 → 一次 Linear。

    官方 Compressor 在 compress_ratio=1 时不做门控池化，只保留 ``wkv``。
    从 CSA2Attention 拆出，避免解码器再挂编码器的 k/v 双投影。
    """

    def __init__(self, hidden_size: int, kv_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, kv_dim * 2, bias=False)

    def forward(self, encoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v = self.proj(encoder_hidden).chunk(2, dim=-1)
        return k, v


class CSA2Attention(nn.Module):
    """CSA2 公共部分：Q + 局部 SWA + 稀疏拼 softmax。

    编码器 / 解码器是两个子类，组网时按层选用，不要直接实例化本类。
    """
    is_decoder = False

    def __init__(self, layer_id: int, config, layout: Dict):
        super().__init__()
        self.layer_id = layer_id
        self.mode = layout['modes'][layer_id]
        self.compress_ratio = int(layout['compress_ratios'][layer_id])
        self.n_enc = layout['n_enc']
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        assert self.n_heads % self.n_kv == 0
        self.n_rep = self.n_heads // self.n_kv
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.window = int(getattr(config, 'sliding_window', 64))
        self.index_topk = int(getattr(config, 'csa2_index_topk', 16))
        self.index_n_heads = int(getattr(config, 'csa2_index_n_heads', 4))
        self.index_head_dim = int(getattr(config, 'csa2_index_head_dim', 32))
        self.candidate_block = int(getattr(config, 'csa2_candidate_block_size', 4))
        self.candidate_top_blocks = int(getattr(config, 'csa2_candidate_topk_blocks', 8))
        self.candidate_source = layout['candidate_source_layer_id']
        self.use_fp4_kv = bool(getattr(config, 'use_fp4_kv', False))
        self.fp4_group = int(getattr(config, 'fp4_kv_group_size', 16))
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.swa_k_proj = nn.Linear(config.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.swa_v_proj = nn.Linear(config.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        kv_dim = self.n_kv * self.head_dim
        if self.mode == 'full':
            self._init_main_kv(config, kv_dim)
            self.indexer_q_proj = nn.Linear(config.hidden_size, self.index_n_heads * self.index_head_dim, bias=False)
            self.indexer_k_proj = nn.Linear(kv_dim, self.index_n_heads * self.index_head_dim, bias=False)
            self.indexer_w = nn.Parameter(torch.ones(self.index_n_heads))
        elif self.mode == 'reindex':
            self.indexer_q_proj = nn.Linear(config.hidden_size, self.index_n_heads * self.index_head_dim, bias=False)
            self.indexer_w = nn.Parameter(torch.ones(self.index_n_heads))

    def _init_main_kv(self, config, kv_dim: int):
        raise NotImplementedError

    def _main_kv_src(self, x, encoder_hidden):
        return x

    def _project_main_kv(self, src, bsz, seq_len):
        raise NotImplementedError

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, n_kv, D] → [B, T, n_heads, D]
        if self.n_rep == 1:
            return x
        b, t, n_kv, d = x.shape
        return x[:, :, :, None, :].expand(b, t, n_kv, self.n_rep, d).reshape(b, t, n_kv * self.n_rep, d)

    def _swa_attend(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            start_pos: int,
            attention_mask: Optional[torch.Tensor],
    ):
        """局部窗口注意力分数。q 为当前 chunk，k/v 为含 cache 的序列。

        返回 scores [B,T,H,W]、v_win [B,T,H,W,D]、valid [B,T,W]。
        """
        bsz, qlen, _, dim = q.shape
        total = k.size(1)
        past_len = total - qlen
        k_abs_start = start_pos - past_len
        q_abs = start_pos + torch.arange(qlen, device=q.device)
        rel = torch.arange(self.window, device=q.device) - (self.window - 1)
        key_abs = q_abs[:, None] + rel[None, :]
        key_idx = key_abs - k_abs_start
        valid = (key_idx >= 0) & (key_idx < total) & (key_abs >= 0)
        key_idx_c = key_idx.clamp(0, max(total - 1, 0))
        k_h = self._repeat_kv(k)
        v_h = self._repeat_kv(v)
        k_win = k_h[:, key_idx_c, :, :].permute(0, 1, 3, 2, 4)  # [B, T, H, W, D]
        v_win = v_h[:, key_idx_c, :, :].permute(0, 1, 3, 2, 4)
        valid_b = valid.view(1, qlen, self.window).expand(bsz, qlen, self.window)
        if attention_mask is not None:
            key_in_chunk = (key_abs >= start_pos) & (key_abs < start_pos + qlen)
            local = (key_abs - start_pos).clamp(0, qlen - 1)
            mask_at = attention_mask[:, local].bool()
            valid_b = valid_b & (~key_in_chunk.unsqueeze(0) | mask_at)
        scores = (q.unsqueeze(3) * k_win).sum(-1) / math.sqrt(dim)
        scores = scores.masked_fill(~valid_b.unsqueeze(2), torch.finfo(scores.dtype).min)
        return scores, v_win, valid_b

    def _select_topk(
            self,
            scores: torch.Tensor,
            q_abs: torch.Tensor,
            end_pos: torch.Tensor,
            extra_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # scores [B,T,S]，只允许压缩槽的结束位置 <= 当前 query 位置
        causal = end_pos.view(1, 1, -1) <= q_abs.view(1, -1, 1)
        valid = causal
        if extra_valid is not None:
            valid = valid & extra_valid
        masked = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        k = min(self.index_topk, scores.size(-1))
        topv, topi = torch.topk(masked, k=k, dim=-1, sorted=False)
        # 全 -inf 时 topk 仍会返回下标，用有限值过滤
        topi = torch.where(torch.isfinite(topv), topi, torch.zeros_like(topi))
        return topi

    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_value=None,
            use_cache: bool = False,
            attention_mask: Optional[torch.Tensor] = None,
            csa2_bank: Optional[Dict] = None,
            encoder_hidden: Optional[torch.Tensor] = None,
            start_pos: int = 0,
    ):
        bsz, seq_len, _ = x.shape
        cos, sin = position_embeddings
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        swa_k = self.swa_k_proj(x).view(bsz, seq_len, self.n_kv, self.head_dim)
        swa_v = self.swa_v_proj(x).view(bsz, seq_len, self.n_kv, self.head_dim)
        q = apply_rope(q, cos, sin)
        swa_k = apply_rope(swa_k, cos, sin)

        past_swa_k = past_swa_v = past_mk = past_mv = None
        if past_key_value is not None:
            past_swa_k, past_swa_v = past_key_value[0], past_key_value[1]
            if len(past_key_value) >= 5:
                past_mk, past_mv = past_key_value[2], past_key_value[3]
            swa_k = torch.cat([past_swa_k, swa_k], dim=1)
            swa_v = torch.cat([past_swa_v, swa_v], dim=1)

        bank = csa2_bank if csa2_bank is not None else {}
        q_abs = start_pos + torch.arange(seq_len, device=x.device)

        # ----- 仅 SWA（前两层）-----
        if self.mode == 'swa':
            scores_w, v_win, valid_w = self._swa_attend(q, swa_k, swa_v, start_pos, attention_mask)
            attn = F.softmax(scores_w.float(), dim=-1).type_as(q)
            attn = self.attn_dropout(attn)
            ctx = (attn.unsqueeze(-1) * v_win).sum(dim=3)  # [B,T,H,D]
            out = self.resid_dropout(self.o_proj(ctx.reshape(bsz, seq_len, -1)))
            present = self._make_present(swa_k, swa_v, None, None, start_pos + seq_len, use_cache)
            return out, present, bank

        # ----- 准备全局 main KV -----
        main_k = bank.get('main_k')
        main_v = bank.get('main_v')
        indexer_k = bank.get('indexer_k')
        topk_idx = bank.get('topk')

        if self.mode == 'full':
            src = self._main_kv_src(x, encoder_hidden)
            if src.size(1) != seq_len:
                # 编码器隐状态应与当前 chunk 对齐；多出来的历史走 cache
                src = src[:, -seq_len:, :]
            mk, mv = self._project_main_kv(src, bsz, seq_len)
            mk = apply_rope(mk, cos, sin)
            if past_mk is not None:
                mk = torch.cat([past_mk, mk], dim=1)
                mv = torch.cat([past_mv, mv], dim=1)
            raw_mk, raw_mv = mk, mv
            main_k = compress_tokens(mk, self.compress_ratio)
            main_v = compress_tokens(mv, self.compress_ratio)
            if self.use_fp4_kv:
                main_k = fake_quant_fp4_e2m1(main_k, self.fp4_group)
                main_v = fake_quant_fp4_e2m1(main_v, self.fp4_group)
            flat_k = main_k.reshape(bsz, main_k.size(1), -1)
            indexer_k = self.indexer_k_proj(flat_k).view(bsz, main_k.size(1), self.index_n_heads, self.index_head_dim)
            indexer_q = self.indexer_q_proj(x).view(bsz, seq_len, self.index_n_heads, self.index_head_dim)
            scores_i = lightning_index_scores(indexer_q, indexer_k, self.indexer_w)
            end_pos = compressed_end_positions(mk.size(1), self.compress_ratio, x.device)
            topk_idx = self._select_topk(scores_i, q_abs, end_pos)
            if self.layer_id == self.candidate_source:
                causal = end_pos.view(1, 1, -1) <= q_abs.view(1, -1, 1)
                bank['candidate_pool'] = build_candidate_pool(
                    scores_i, self.candidate_block, self.candidate_top_blocks, valid=causal
                )
            bank['main_k'] = main_k
            bank['main_v'] = main_v
            bank['indexer_k'] = indexer_k
            bank['topk'] = topk_idx
            bank['n_raw'] = int(mk.size(1))
        else:
            raw_mk = past_mk
            raw_mv = past_mv
            if main_k is None or main_v is None:
                raise RuntimeError(f'CSA2 {self.mode} 层 {self.layer_id} 没有可用的共享 main KV，请检查 Full 层布局')
            if self.mode == 'reindex':
                indexer_q = self.indexer_q_proj(x).view(bsz, seq_len, self.index_n_heads, self.index_head_dim)
                pool = bank.get('candidate_pool') if self.is_decoder else None
                ratio = max(self.compress_ratio, 1)
                end_pos = (torch.arange(main_k.size(1), device=x.device) + 1) * ratio - 1
                if pool is not None:
                    k_pool = gather_by_index(indexer_k, pool)  # [B,T,P,Hi,Di]
                    dots = torch.einsum('bthd,btphd->btph', indexer_q, k_pool)
                    pool_scores = torch.relu(dots).mul(self.indexer_w.view(1, 1, 1, -1)).sum(-1)
                    k_use = min(self.index_topk, pool_scores.size(-1))
                    _, local = torch.topk(pool_scores, k=k_use, dim=-1, sorted=False)
                    topk_idx = torch.gather(pool, 2, local)
                else:
                    scores_i = lightning_index_scores(indexer_q, indexer_k, self.indexer_w)
                    topk_idx = self._select_topk(scores_i, q_abs, end_pos)
                bank['topk'] = topk_idx
            else:
                # Reuse：沿用最近 Full/Reindex 的 Top-K；decode 时 bank 已是当前 query 的下标
                if topk_idx is None:
                    raise RuntimeError(f'CSA2 reuse 层 {self.layer_id} 没有 Top-K 下标')
                if topk_idx.size(1) != seq_len:
                    topk_idx = topk_idx[:, -seq_len:, :]

        # ----- 拼 SWA + 选中的全局 KV，一次 softmax -----
        scores_w, v_win, valid_w = self._swa_attend(q, swa_k, swa_v, start_pos, attention_mask)
        k_sel = gather_by_index(self._repeat_kv(main_k), topk_idx)  # [B,T,K,H,D]
        v_sel = gather_by_index(self._repeat_kv(main_v), topk_idx)
        k_sel = k_sel.permute(0, 1, 3, 2, 4)  # [B,T,H,K,D]
        v_sel = v_sel.permute(0, 1, 3, 2, 4)
        scores_g = (q.unsqueeze(3) * k_sel).sum(-1) / math.sqrt(self.head_dim)
        # topk 在全为 -inf 时仍会返回下标，必须在 softmax 前再次按因果边界屏蔽
        ratio = max(int(self.compress_ratio), 1)
        n_raw = int(bank.get('n_raw', main_k.size(1) * ratio))
        end_pos = ((torch.arange(main_k.size(1), device=x.device) + 1) * ratio - 1).clamp(max=n_raw - 1)
        sel_end = end_pos[topk_idx.clamp(0, max(main_k.size(1) - 1, 0))]
        valid_g = sel_end <= q_abs.view(1, -1, 1)
        scores_g = scores_g.masked_fill(~valid_g.unsqueeze(2), torch.finfo(scores_g.dtype).min)
        scores = torch.cat([scores_g, scores_w], dim=-1)
        v_cat = torch.cat([v_sel, v_win], dim=3)
        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        attn = self.attn_dropout(attn)
        ctx = (attn.unsqueeze(-1) * v_cat).sum(dim=3)
        out = self.resid_dropout(self.o_proj(ctx.reshape(bsz, seq_len, -1)))
        present = self._make_present(swa_k, swa_v, raw_mk if self.mode == 'full' else None,
                                     raw_mv if self.mode == 'full' else None,
                                     start_pos + seq_len, use_cache)
        return out, present, bank

    def _make_present(self, swa_k, swa_v, mk, mv, total_len: int, use_cache: bool):
        if not use_cache:
            return None
        # SWA Bounded Replay：只持久化最近 n_win 的局部 KV
        w = self.window
        swa_k = swa_k[:, -w:, :, :].contiguous()
        swa_v = swa_v[:, -w:, :, :].contiguous()
        seq_t = torch.tensor([total_len], device=swa_k.device, dtype=torch.long)
        if mk is None:
            return (swa_k, swa_v, seq_t)
        return (swa_k, swa_v, mk, mv, seq_t)


class CSA2EncoderAttention(CSA2Attention):
    """编码器 CSA2：全局 KV 从本层 hidden 做 k/v 双投影，再按 compress_ratio 池化。"""
    is_decoder = False

    def _init_main_kv(self, config, kv_dim: int):
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)

    def _project_main_kv(self, src, bsz, seq_len):
        mk = self.k_proj(src).view(bsz, seq_len, self.n_kv, self.head_dim)
        mv = self.v_proj(src).view(bsz, seq_len, self.n_kv, self.head_dim)
        return mk, mv


class CSA2DecoderAttention(CSA2Attention):
    """解码器 CSA2：全局 KV 只用 CEDDecoderKV（一个 Linear），源是编码器出口。"""
    is_decoder = True

    def _init_main_kv(self, config, kv_dim: int):
        self.dec_kv = CEDDecoderKV(config.hidden_size, kv_dim)

    def _main_kv_src(self, x, encoder_hidden):
        return encoder_hidden if encoder_hidden is not None else x

    def _project_main_kv(self, src, bsz, seq_len):
        mk, mv = self.dec_kv(src)
        mk = mk.view(bsz, seq_len, self.n_kv, self.head_dim)
        mv = mv.view(bsz, seq_len, self.n_kv, self.head_dim)
        return mk, mv


def build_csa2_attn(layer_id: int, config, layout: Dict) -> CSA2Attention:
    """前半层编码器注意力，后半层解码器注意力。"""
    cls = CSA2DecoderAttention if layer_id >= layout['n_enc'] else CSA2EncoderAttention
    return cls(layer_id, config, layout)


def infer_dsv41_start_pos(past_key_values) -> int:
    if not past_key_values or past_key_values[0] is None:
        return 0
    p0 = past_key_values[0]
    last = p0[-1]
    if torch.is_tensor(last) and last.numel() == 1:
        return int(last.item())
    return p0[0].shape[1]


def identity_gamma(streams: torch.Tensor) -> torch.Tensor:
    s = streams.size(2)
    eye = torch.eye(s, device=streams.device, dtype=streams.dtype)
    return eye.view(1, 1, s, s).expand(streams.size(0), streams.size(1), s, s)
