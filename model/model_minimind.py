# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # 混合注意力：Kimi-K3 风格（默认 3 层 KDA + 1 层全局 MLA）
            # KDA 层拥有独立 QKVO / 短卷积 / 通道遗忘门，不与 MLA 共享权重
            # 下面 WY 扫描、Quantile Balancing、QAT、MoonViT 等也只在显式打开时生效
            ####################################################
            use_hybrid_attn: bool = False,
            linear_attn_ratio: int = 3,
            kda_conv_kernel_size: int = 4,
            kda_gate_lower_bound: float = -5.0,
            kda_chunk_size: int = 16,
            kda_use_wy_scan: bool = True,
            kda_use_triton: bool = True,
            kda_context_parallel_size: int = 1,
            mla_kv_lora_rank: int = None,
            mla_qk_nope_head_dim: int = None,
            mla_v_head_dim: int = None,
            use_attn_res: bool = None,
            attn_res_block_size: int = 4,
            latent_moe_dim: int = None,
            use_quantile_balancing: bool = None,
            use_qat: bool = False,
            qat_weight_bits: int = 4,
            qat_act_bits: int = 8,
            qat_block_size: int = 32,
            situ_beta1: float = 4.0,
            situ_beta2: float = 25.0,
            use_vision: bool = False,
            vision_hidden_size: int = 128,
            vision_num_layers: int = 2,
            vision_num_heads: int = 4,
            vision_patch_size: int = 16,
            vision_image_size: int = 224,
            vision_num_channels: int = 3,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        self.use_hybrid_attn = use_hybrid_attn
        self.linear_attn_ratio = linear_attn_ratio
        self.kda_conv_kernel_size = kda_conv_kernel_size
        self.kda_gate_lower_bound = kda_gate_lower_bound
        self.kda_chunk_size = kda_chunk_size
        self.kda_use_wy_scan = kda_use_wy_scan
        self.kda_use_triton = kda_use_triton
        self.kda_context_parallel_size = kda_context_parallel_size
        head_dim = hidden_size // num_attention_heads
        self.mla_kv_lora_rank = mla_kv_lora_rank if mla_kv_lora_rank is not None else max(head_dim, hidden_size // 8)
        self.mla_qk_nope_head_dim = mla_qk_nope_head_dim if mla_qk_nope_head_dim is not None else head_dim
        self.mla_v_head_dim = mla_v_head_dim if mla_v_head_dim is not None else head_dim
        self.use_attn_res = use_hybrid_attn if use_attn_res is None else use_attn_res
        self.attn_res_block_size = attn_res_block_size
        self.latent_moe_dim = latent_moe_dim
        self.use_qat = use_qat
        self.qat_weight_bits = qat_weight_bits
        self.qat_act_bits = qat_act_bits
        self.qat_block_size = qat_block_size
        self.situ_beta1 = situ_beta1
        self.situ_beta2 = situ_beta2
        self.use_vision = use_vision
        self.vision_hidden_size = vision_hidden_size
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_patch_size = vision_patch_size
        self.vision_image_size = vision_image_size
        self.vision_num_channels = vision_num_channels
        # 混合注意力 + MoE 时默认走 K3 的 Quantile Balancing；原版 MiniMind MoE 仍用 softmax+aux
        if use_quantile_balancing is None:
            use_quantile_balancing = bool(use_hybrid_attn and use_moe)
        self.use_quantile_balancing = use_quantile_balancing
        if self.use_quantile_balancing:
            self.scoring_func = 'sigmoid'
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        if not self.use_quantile_balancing:
            self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        if self.latent_moe_dim is None and self.use_hybrid_attn and self.use_moe:
            self.latent_moe_dim = max(64, hidden_size // 2)

    def is_kda_layer(self, layer_id: int) -> bool:
        """Kimi-K3 的层排布：每组 ``linear_attn_ratio`` 层 KDA + 1 层全局注意力。

        最后一层强制为全局注意力（MLA）；凑不齐一组的余数层也保持全局注意力。
        ``use_hybrid_attn=False`` 时全部仍是原版 GQA，权重布局与旧 checkpoint 兼容。
        """
        if not self.use_hybrid_attn:
            return False
        if layer_id == self.num_hidden_layers - 1:
            return False
        ratio = max(int(self.linear_attn_ratio), 0)
        group = ratio + 1
        n_complete = (self.num_hidden_layers // group) * group
        if layer_id >= n_complete:
            return False
        return (layer_id % group) != ratio


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.k3_ops import kda_delta_rule_scan, qat_linear


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


def infer_cache_start_pos(past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]) -> int:
    """RoPE offset from full-attn KV cache; KDA layers keep a recurrent state instead of KV."""
    if not past_key_values or past_key_values[0] is None:
        return 0
    for pkv in past_key_values:
        if pkv is None:
            continue
        cache_a, cache_b = pkv[0], pkv[1]
        # Full attention KV: both tensors share shape [B, seq, n_kv_heads, head_dim]
        if cache_a.dim() == 4 and cache_b.dim() == 4 and cache_a.shape == cache_b.shape:
            return cache_a.shape[1]
    return 0


class ShortConvolution(nn.Module):
    """KDA 的因果深度可分离短卷积（K3 / Kimi Linear：kernel=4 + SiLU）。

    对 Q/K/V 在投影之后、L2Norm 之前做局部混合，让线性注意力也能看见邻近 token。
    ``cache`` 保存最后 ``kernel_size-1`` 步，解码时与预填等价。
    """

    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = 'silu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        bsz, seq_len, hidden = x.shape
        x_c = x.transpose(1, 2)
        if cache is None:
            cache = x_c.new_zeros(bsz, hidden, self.kernel_size - 1)
        x_c = torch.cat([cache, x_c], dim=-1)
        y = F.conv1d(x_c, self.conv.weight, bias=self.conv.bias, groups=self.conv.groups)
        new_cache = x_c[:, :, -(self.kernel_size - 1):]
        y = y.transpose(1, 2)
        if self.activation == 'silu':
            y = F.silu(y)
        return y, new_cache


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.attn_type = "full"

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class GatedMLA(nn.Module):
    """Kimi-K3 的 Gated MLA：KV 潜空间压缩 + NoPE + 满秩输出门。

    混合注意力里「全局层」用这个，而不是原版 GQA+RoPE：
    - ``kv_a_proj`` 把 KV 压到 ``mla_kv_lora_rank``，RMSNorm 后再 ``kv_b_proj`` 展开成 K/V；
    - Q 仍按头投影；KDA 已经带了位置信息，所以这里 **不做 RoPE（NoPE）**；
    - ``g_proj`` 对注意力输出做 sigmoid 门控，再 ``o_proj`` 回到 hidden。
    """

    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.mla_qk_nope_head_dim
        self.v_head_dim = args.mla_v_head_dim
        self.kv_lora_rank = args.mla_kv_lora_rank
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.qk_nope_head_dim, bias=False)
        self.kv_a_proj = nn.Linear(args.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.g_proj = nn.Linear(args.hidden_size, self.n_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.attn_type = "mla"

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.qk_nope_head_dim)
        compressed = self.kv_a_layernorm(self.kv_a_proj(x))
        kv = self.kv_b_proj(compressed).view(bsz, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # NoPE：位置由同一组里的 KDA 承担；MLA 只做全局内容检索

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        past_kv = (k, v) if use_cache else None

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        kv_seq = k.shape[-2]
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True, scale=self.qk_nope_head_dim ** -0.5
            )
        else:
            scores = (q @ k.transpose(-2, -1)) * (self.qk_nope_head_dim ** -0.5)
            causal = torch.ones(seq_len, kv_seq, device=scores.device, dtype=torch.bool).tril(diagonal=kv_seq - seq_len)
            scores = scores.masked_fill(~causal, float('-inf'))
            if attention_mask is not None:
                scores = scores + (1.0 - attention_mask[:, None, None, :].to(scores.dtype)) * -1e9
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.attn_dropout(scores)
            output = scores @ v

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = output * torch.sigmoid(self.g_proj(x))
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


def apply_attn_res(prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: RMSNorm):
    """Block AttnRes：用可学习伪 query 对「历史 block 摘要 + 当前前缀和」做 softmax 混合。

    K3 每隔 ``attn_res_block_size`` 层把当前残差前缀登记进 ``block_residual``，
    后续层不再用普通 skip，而是注意力式地读取这些 block 记忆，减轻深堆叠时的梯度稀释。
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    k = v_float * torch.rsqrt(v_float.pow(2).mean(-1, keepdim=True) + norm.eps)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = torch.softmax(scores, dim=-1).unsqueeze(1)
    return torch.matmul(probs, v_float).squeeze(1).to(dtype=v.dtype)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)


class KimiDeltaAttention(nn.Module):
    """Kimi-K3 / Kimi Linear 的 KDA：每层独立 QKVO，短卷积 + 通道遗忘门 + delta rule。

    布局刻意对齐原版 MiniMindBlock（每层自己的投影），不与 MLA 共享 QKVO。

    数据流（每个头）::

        x → Linear → ShortConv+SiLU → (Q/K 再 L2Norm)
        α = g_min * sigmoid(exp(A) * (lowrank(x) + dt_bias))   # 有下界的通道遗忘
        β = sigmoid(b_proj(x))                                 # delta 步长
        S, o = WY/UT 分块扫描(q, k, v, log α, β)
        o ← RMSNorm(o) ⊙ sigmoid(lowrank_gate(x)) → o_proj

    ``past_key_value`` 存的是 RNN 状态 S 以及 Q/K/V 卷积 cache，不是 KV 序列。
    """

    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.hidden_size = args.hidden_size
        proj_size = self.n_heads * self.head_dim
        conv_kernel = args.kda_conv_kernel_size

        self.q_proj = nn.Linear(args.hidden_size, proj_size, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, proj_size, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, proj_size, bias=False)
        self.q_conv1d = ShortConvolution(proj_size, kernel_size=conv_kernel, activation='silu')
        self.k_conv1d = ShortConvolution(proj_size, kernel_size=conv_kernel, activation='silu')
        self.v_conv1d = ShortConvolution(proj_size, kernel_size=conv_kernel, activation='silu')

        self.A_log = nn.Parameter(torch.log(torch.empty(self.n_heads).uniform_(1, 16)))
        self.f_a_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, proj_size, bias=False)
        dt = torch.exp(
            torch.rand(proj_size) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.b_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)

        self.g_a_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, proj_size, bias=False)
        self.o_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(proj_size, args.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.gate_lower_bound = args.kda_gate_lower_bound
        self.chunk_size = args.kda_chunk_size
        self.use_wy_scan = args.kda_use_wy_scan
        self.use_triton = getattr(args, 'kda_use_triton', True)
        self.context_parallel_size = args.kda_context_parallel_size
        self.attn_type = "kda"

    def _forget_gate(self, x: torch.Tensor):
        """K3 带下界的通道遗忘门，输出 log 域的 α ∈ (g_min, 0)。

        旧版 Linear/Mamba 常用 ``-exp(A) * softplus(z)``，值域 (-∞, 0)，长序列上容易把状态
        乘到数值零。K3 改为 ``g = g_min * sigmoid(exp(A) * z)``，默认 ``g_min=-5``，
        即 α ≥ exp(-5)，记忆不会被一次性清掉。
        """
        g = self.f_b_proj(self.f_a_proj(x))
        g = g.view(*x.shape[:2], self.n_heads, self.head_dim)
        dt_bias = self.dt_bias.view(self.n_heads, self.head_dim)
        z = g.float() + dt_bias
        a_scale = self.A_log.float().exp().view(1, 1, self.n_heads, 1)
        return self.gate_lower_bound * torch.sigmoid(a_scale * z)

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        conv_q = conv_k = conv_v = None
        recurrent_state = None
        if past_key_value is not None:
            recurrent_state, conv_pack = past_key_value
            conv_q, conv_k, conv_v = conv_pack.unbind(dim=1)

        q, conv_q = self.q_conv1d(self.q_proj(x), cache=conv_q)
        k, conv_k = self.k_conv1d(self.k_proj(x), cache=conv_k)
        v, conv_v = self.v_conv1d(self.v_proj(x), cache=conv_v)

        q = l2_normalize(q.view(bsz, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        k = l2_normalize(k.view(bsz, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        log_decay = self._forget_gate(x).transpose(1, 2)  # [B, H, T, D] 通道级 log α
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)  # [B, H, T]
        output, state = kda_delta_rule_scan(
            q, k, v, log_decay, beta,
            initial_state=recurrent_state,
            attention_mask=attention_mask,
            chunk_size=self.chunk_size,
            use_wy_scan=self.use_wy_scan,
            context_parallel_size=self.context_parallel_size,
            use_triton=self.use_triton,
        )

        past_kv = None
        if use_cache:
            conv_pack = torch.stack([conv_q, conv_k, conv_v], dim=1)
            past_kv = (state, conv_pack)

        output = output.transpose(1, 2)
        g_out = self.g_b_proj(self.g_a_proj(x)).view(bsz, seq_len, self.n_heads, self.head_dim)
        output = self.o_norm(output) * torch.sigmoid(g_out)
        output = output.reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, hidden_size: int = None, intermediate_size: int = None,
                 use_situ: bool = False, use_qat: bool = False):
        super().__init__()
        hidden_size = config.hidden_size if hidden_size is None else hidden_size
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            if hidden_size == config.hidden_size and config.intermediate_size is None:
                config.intermediate_size = intermediate_size
            elif hidden_size == config.hidden_size:
                intermediate_size = config.intermediate_size
        self.hidden_size = hidden_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.use_situ = use_situ or config.hidden_act == 'situ'
        self.act_fn = ACT2FN['silu'] if self.use_situ else ACT2FN[config.hidden_act]
        # K3 SiTU：β1=4 作用在 gate，β2=25 作用在 up；普通 SwiGLU 不走这条
        self.situ_beta1 = float(getattr(config, 'situ_beta1', 4.0))
        self.situ_beta2 = float(getattr(config, 'situ_beta2', 25.0))
        self.use_qat = use_qat
        self.qat_weight_bits = int(getattr(config, 'qat_weight_bits', 4))
        self.qat_act_bits = int(getattr(config, 'qat_act_bits', 8))
        self.qat_block_size = int(getattr(config, 'qat_block_size', 32))

    def _linear(self, proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if not self.use_qat:
            return proj(x)
        return qat_linear(
            proj, x,
            quant_weight=True,
            quant_act=True,
            weight_bits=self.qat_weight_bits,
            act_bits=self.qat_act_bits,
            block_size=self.qat_block_size,
        )

    def forward(self, x):
        gate, up = self._linear(self.gate_proj, x), self._linear(self.up_proj, x)
        if self.use_situ:
            # SiTU(x, y) = [β1 tanh(x/β1) ⊙ σ(x)] ⊙ [β2 tanh(y/β2)]
            gate_f, up_f = gate.float(), up.float()
            b1, b2 = self.situ_beta1, self.situ_beta2
            hidden = (b1 * torch.tanh(gate_f / b1) * torch.sigmoid(gate_f) * (b2 * torch.tanh(up_f / b2))).to(x.dtype)
        else:
            hidden = self.act_fn(gate) * up
        return self.dropout(self._linear(self.down_proj, hidden) if self.use_qat else self.down_proj(hidden))


class MoEGate(nn.Module):
    """MoE 路由器。

    - 原版 MiniMind：softmax 分数 + aux load-balancing loss。
    - K3 Quantile Balancing：sigmoid 分数、无梯度 expert bias ``b``，用 ``s+b`` 做 Top-k，
      混合权重仍取 **原始 s**（不含 b）。训练时
      ``b_j ← -quantile_{1-k/n}(s_{:,j} - α)`` 再减均值；推理冻结 ``b``，不再使用 aux loss。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.use_quantile_balancing = bool(getattr(config, 'use_quantile_balancing', False))
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # 非训练参数：不进 Adam；eval 时保持上次更新的负荷均衡偏置
        self.register_buffer('expert_bias', torch.zeros(self.n_routed_experts), persistent=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _quantile_balancing_route(self, logits: torch.Tensor):
        # s ∈ (0,1)^{n}，排序用 s+b，权重用裸 s
        scores = torch.sigmoid(logits.float()).type_as(logits)
        biased = scores + self.expert_bias.to(dtype=scores.dtype)
        k = self.top_k
        n_exp = self.n_routed_experts
        k_plus = min(k + 1, n_exp)
        topk_vals, topk_idx_full = torch.topk(biased, k=k_plus, dim=-1, sorted=True)
        topk_idx = topk_idx_full[:, :k].contiguous()
        # α：每个 token 上第 (k+1) 大的 s+b，作为负荷均衡的截断阈值
        alpha_cut = topk_vals[:, k - 1:k] if k_plus == k else topk_vals[:, k:k + 1]
        topk_weight = scores.gather(-1, topk_idx)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        if self.training and scores.shape[0] > 1:
            with torch.no_grad():
                residual = scores.float() - alpha_cut.float()  # [tokens, experts]
                q = max(min(1.0 - (k / max(n_exp, 1)), 1.0), 0.0)
                # 每个专家一列做分位数；token 太少时退回均值
                try:
                    bias = -torch.quantile(residual, q, dim=0)
                except RuntimeError:
                    bias = -residual.mean(dim=0)
                bias = bias - bias.mean()
                self.expert_bias.copy_(bias.to(dtype=self.expert_bias.dtype))
        aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.use_quantile_balancing or self.scoring_func in ('sigmoid', 'quantile_balancing'):
            return self._quantile_balancing_route(logits)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_moe_dim
        self.use_latent_moe = self.latent_dim is not None and self.latent_dim > 0
        expert_hidden = self.latent_dim if self.use_latent_moe else config.hidden_size
        # QAT 只打在 routed expert 上，共享专家 / 路由器保持高精度
        self.experts = nn.ModuleList([
            FeedForward(
                config,
                hidden_size=expert_hidden,
                use_situ=self.use_latent_moe,
                use_qat=bool(getattr(config, 'use_qat', False)),
            )
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])
        if self.use_latent_moe:
            self.routed_down_proj = nn.Linear(config.hidden_size, self.latent_dim, bias=False)
            self.routed_up_proj = nn.Linear(self.latent_dim, config.hidden_size, bias=False)
            self.routed_norm = RMSNorm(self.latent_dim, eps=config.rms_norm_eps)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        if self.use_latent_moe:
            x = self.routed_down_proj(x)
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1))
        if self.use_latent_moe:
            y = self.routed_up_proj(self.routed_norm(y))
        y = y.view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """标准 MiniMind 层：Attn/KDA/MLA + FFN/MoE。

    混合模式下按 ``is_kda_layer`` 在 KDA 与 Gated MLA 之间切换；
    ``use_attn_res`` 时残差改为 K3 的 block 级 AttnRes，而不是 x+attn+mlp。
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_id = layer_id
        self.use_attn_res = config.use_attn_res
        self.attn_res_block_size = config.attn_res_block_size
        if config.is_kda_layer(layer_id):
            self.self_attn = KimiDeltaAttention(config)
        elif config.use_hybrid_attn:
            self.self_attn = GatedMLA(config)
        else:
            self.self_attn = Attention(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        if self.use_attn_res:
            self.self_attention_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False,
                attention_mask=None, block_residual=None):
        if self.use_attn_res:
            return self._forward_attn_res(
                hidden_states, position_embeddings, past_key_value, use_cache, attention_mask, block_residual
            )
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value, block_residual

    def _forward_attn_res(self, hidden_states, position_embeddings, past_key_value, use_cache,
                          attention_mask, block_residual):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states
        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = apply_attn_res(
                prefix_sum.reshape(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)
        if self.layer_id % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.reshape(-1, hidden_size).unsqueeze(1)], dim=1
            )
            prefix_sum = None

        attn_out, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out
        hidden_states = apply_attn_res(
            prefix_sum.reshape(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)
        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
        prefix_sum = mlp_out if prefix_sum is None else prefix_sum + mlp_out
        return prefix_sum, present_key_value, block_residual


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_attn_res = config.use_attn_res
        if self.use_attn_res:
            self.output_attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                vision_embeds: Optional[torch.Tensor] = None,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = infer_cache_start_pos(past_key_values)

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 视觉 token 拼在文本左侧；NTP 时这些位置的 label 在 CausalLM 里填 -100
        if vision_embeds is not None:
            hidden_states = torch.cat([self.dropout(vision_embeds), hidden_states], dim=1)
            seq_length = hidden_states.shape[1]
            if attention_mask is not None:
                vis_mask = attention_mask.new_ones(batch_size, vision_embeds.shape[1])
                attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        block_residual = None
        if self.use_attn_res:
            block_residual = hidden_states.new_zeros(batch_size * seq_length, 0, self.config.hidden_size)
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present, block_residual = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
                block_residual=block_residual,
            )
            presents.append(present)

        if self.use_attn_res:
            hidden_states = apply_attn_res(
                hidden_states.reshape(-1, self.config.hidden_size),
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
            ).view(batch_size, seq_length, -1)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        # K3 的 MoonViT-V2 与 LLM 一起做 NTP；默认关闭以免改变纯文本 checkpoint 形状
        self.vision_tower = None
        if getattr(self.config, 'use_vision', False):
            from model.model_moonvit import MoonViTV2
            self.vision_tower = MoonViTV2(self.config)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.Tensor] = None,
                **args):
        vision_embeds = None
        if pixel_values is not None or pixel_values_videos is not None:
            if self.vision_tower is None:
                raise ValueError('传入了像素输入，但 config.use_vision=False，未构建 MoonViT-V2')
            vision_embeds = self.vision_tower.encode(pixel_values, pixel_values_videos)
            if labels is not None:
                pad = labels.new_full((labels.size(0), vision_embeds.size(1)), -100)
                labels = torch.cat([pad, labels], dim=1)
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            vision_embeds=vision_embeds,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
