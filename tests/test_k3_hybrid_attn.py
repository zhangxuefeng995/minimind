"""Sanity checks for MiniMind Kimi-K3 hybrid attention."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.model_minimind import (
    Attention,
    GatedMLA,
    KimiDeltaAttention,
    MiniMindConfig,
    MiniMindForCausalLM,
)


def _tiny_config(**kwargs):
    cfg = dict(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
        flash_attn=False,
        dropout=0.0,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )
    cfg.update(kwargs)
    return MiniMindConfig(**cfg)


def test_default_model_keeps_full_attention():
    model = MiniMindForCausalLM(_tiny_config(use_hybrid_attn=False))
    assert all(isinstance(layer.self_attn, Attention) for layer in model.model.layers)
    keys = model.state_dict().keys()
    assert 'model.layers.0.self_attn.q_proj.weight' in keys
    assert 'model.layers.0.self_attn.q_conv1d.conv.weight' not in keys
    assert 'model.layers.0.self_attn.kv_a_proj.weight' not in keys


def test_k3_layer_pattern_and_independent_params():
    model = MiniMindForCausalLM(_tiny_config(use_hybrid_attn=True, linear_attn_ratio=3))
    types = [layer.self_attn.attn_type for layer in model.model.layers]
    assert types == ['kda', 'kda', 'kda', 'mla']
    assert isinstance(model.model.layers[0].self_attn, KimiDeltaAttention)
    assert isinstance(model.model.layers[3].self_attn, GatedMLA)
    kda_q = model.model.layers[0].self_attn.q_proj.weight
    mla_q = model.model.layers[3].self_attn.q_proj.weight
    assert kda_q.data_ptr() != mla_q.data_ptr()
    keys = model.state_dict().keys()
    assert 'model.layers.0.self_attn.q_conv1d.conv.weight' in keys
    assert 'model.layers.3.self_attn.kv_a_proj.weight' in keys
    assert 'model.layers.3.self_attn.g_proj.weight' in keys
    assert 'model.layers.0.self_attention_res_proj.weight' in keys
    assert 'model.output_attn_res_proj.weight' in keys


def test_last_layer_is_always_full_attention():
    model = MiniMindForCausalLM(_tiny_config(use_hybrid_attn=True, num_hidden_layers=5, linear_attn_ratio=3))
    types = [layer.self_attn.attn_type for layer in model.model.layers]
    assert types[-1] == 'mla'
    assert types == ['kda', 'kda', 'kda', 'mla', 'mla']


def test_forward_backward_and_causal():
    torch.manual_seed(0)
    model = MiniMindForCausalLM(_tiny_config(use_hybrid_attn=True))
    model.train()
    ids = torch.randint(0, 128, (2, 8))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 8, 128)
    out.loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.model.layers[0].self_attn.b_proj.weight.grad is not None
    assert model.model.layers[3].self_attn.kv_a_proj.weight.grad is not None
    assert model.model.layers[3].self_attn.g_proj.weight.grad is not None

    model.eval()
    with torch.no_grad():
        logits_a = model(ids).logits
        ids_b = ids.clone()
        ids_b[:, -1] = (ids_b[:, -1] + 1) % 128
        logits_b = model(ids_b).logits
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-4, rtol=1e-4)


def test_kda_cache_matches_full_prefill():
    torch.manual_seed(1)
    model = MiniMindForCausalLM(_tiny_config(use_hybrid_attn=True))
    model.eval()
    ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        full = model(ids, use_cache=True)
        past = None
        chunks = []
        for t in range(ids.shape[1]):
            step = model(ids[:, t:t + 1], past_key_values=past, use_cache=True)
            past = step.past_key_values
            chunks.append(step.logits)
        stepped = torch.cat(chunks, dim=1)
    assert torch.allclose(full.logits, stepped, atol=2e-4, rtol=2e-4)


def test_k3_bounded_forget_gate():
    model = MiniMindForCausalLM(_tiny_config(use_hybrid_attn=True))
    kda = model.model.layers[0].self_attn
    x = torch.randn(2, 5, 64)
    log_decay = kda._forget_gate(x)
    assert log_decay.max() <= 0
    assert log_decay.min() >= kda.gate_lower_bound - 1e-5


def test_latent_moe_hybrid():
    torch.manual_seed(0)
    model = MiniMindForCausalLM(_tiny_config(use_hybrid_attn=True, use_moe=True))
    mlp = model.model.layers[0].mlp
    assert mlp.use_latent_moe
    assert mlp.gate.use_quantile_balancing
    assert mlp.experts[0].situ_beta1 == 4.0
    assert mlp.experts[0].situ_beta2 == 25.0
    assert 'routed_down_proj.weight' in dict(mlp.named_parameters())
    ids = torch.randint(0, 128, (2, 6))
    out = model(ids, labels=ids)
    out.loss.backward()
    assert out.logits.shape == (2, 6, 128)
    assert mlp.gate.expert_bias.abs().sum() > 0


def test_wy_scan_matches_naive():
    from model.k3_ops import kda_delta_rule_scan_naive, kda_delta_rule_scan_wy, kda_delta_rule_scan
    torch.manual_seed(0)
    bsz, heads, seq, dim = 2, 2, 13, 8
    q = torch.nn.functional.normalize(torch.randn(bsz, heads, seq, dim), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(bsz, heads, seq, dim), dim=-1)
    v = torch.randn(bsz, heads, seq, dim)
    log_decay = -torch.rand(bsz, heads, seq, dim) * 3
    beta = torch.rand(bsz, heads, seq)
    state0 = torch.randn(bsz, heads, dim, dim) * 0.05
    o_n, s_n = kda_delta_rule_scan_naive(q, k, v, log_decay, beta, state0)
    o_w, s_w = kda_delta_rule_scan_wy(q, k, v, log_decay, beta, state0, chunk_size=4)
    assert torch.allclose(o_n, o_w, atol=1e-4, rtol=1e-4)
    assert torch.allclose(s_n, s_w, atol=1e-4, rtol=1e-4)
    o_cp, s_cp = kda_delta_rule_scan(
        q, k, v, log_decay, beta, state0, chunk_size=4, use_wy_scan=True, context_parallel_size=5
    )
    assert torch.allclose(o_n, o_cp, atol=1e-4, rtol=1e-4)
    assert torch.allclose(s_n, s_cp, atol=1e-4, rtol=1e-4)


def test_triton_is_optional_python_fallback():
    """没装 Triton / 没有 CUDA 时必须安静回退，不能 import 失败。"""
    from model.k3_triton import HAS_TRITON, kda_delta_rule_scan_triton, kda_triton_is_available
    from model.k3_ops import kda_delta_rule_scan_naive, kda_delta_rule_scan
    q = torch.nn.functional.normalize(torch.randn(1, 1, 4, 8), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(1, 1, 4, 8), dim=-1)
    v = torch.randn(1, 1, 4, 8)
    g = -torch.rand(1, 1, 4, 8)
    b = torch.rand(1, 1, 4)
    if not kda_triton_is_available(q):
        assert kda_delta_rule_scan_triton(q, k, v, g, b) is None
        o_a, s_a = kda_delta_rule_scan(q, k, v, g, b, use_triton=True)
        o_b, s_b = kda_delta_rule_scan_naive(q, k, v, g, b)
        assert torch.allclose(o_a, o_b, atol=1e-4, rtol=1e-4)
        assert torch.allclose(s_a, s_b, atol=1e-4, rtol=1e-4)
        return
    o_t, s_t = kda_delta_rule_scan_triton(q.cuda(), k.cuda(), v.cuda(), g.cuda(), b.cuda())
    o_n, s_n = kda_delta_rule_scan_naive(q.cuda(), k.cuda(), v.cuda(), g.cuda(), b.cuda())
    assert torch.allclose(o_t, o_n, atol=2e-4, rtol=2e-4)
    assert torch.allclose(s_t, s_n, atol=2e-4, rtol=2e-4)
    _ = HAS_TRITON


def test_quantile_balancing_eval_freezes_bias():
    model = MiniMindForCausalLM(_tiny_config(use_hybrid_attn=True, use_moe=True))
    gate = model.model.layers[0].mlp.gate
    ids = torch.randint(0, 128, (2, 8))
    model.train()
    model(ids)
    bias_after_train = gate.expert_bias.clone()
    model.eval()
    with torch.no_grad():
        model(ids)
    assert torch.equal(bias_after_train, gate.expert_bias)


def test_qat_routed_expert_forward():
    torch.manual_seed(0)
    model = MiniMindForCausalLM(_tiny_config(use_hybrid_attn=True, use_moe=True, use_qat=True))
    assert model.model.layers[0].mlp.experts[0].use_qat
    assert not model.model.layers[0].mlp.shared_experts[0].use_qat
    ids = torch.randint(0, 128, (2, 6))
    out = model(ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss)


def test_per_head_muon_step():
    from trainer.trainer_utils import PerHeadMuonAdamW
    torch.manual_seed(0)
    model = MiniMindForCausalLM(_tiny_config(use_hybrid_attn=True))
    opt = PerHeadMuonAdamW(model, lr=1e-3)
    assert len(opt.muon_params) > 0
    ids = torch.randint(0, 128, (2, 6))
    out = model(ids, labels=ids)
    out.loss.backward()
    q_before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    opt.step()
    assert not torch.equal(q_before, model.model.layers[0].self_attn.q_proj.weight.detach())


def test_moonvit_prepends_vision_tokens():
    torch.manual_seed(0)
    cfg = _tiny_config(
        use_hybrid_attn=True,
        use_vision=True,
        vision_hidden_size=32,
        vision_num_layers=1,
        vision_num_heads=2,
        vision_patch_size=8,
        vision_image_size=16,
        max_position_embeddings=128,
    )
    model = MiniMindForCausalLM(cfg)
    ids = torch.randint(0, 128, (2, 4))
    pixels = torch.randn(2, 3, 16, 16)
    labels = ids.clone()
    out = model(ids, pixel_values=pixels, labels=labels)
    # 16/8=2 grid, 2x2 shuffle -> 1 vision token, logits 比纯文本长 1
    assert out.logits.shape[1] == 5
    out.loss.backward()
    videos = torch.randn(2, 2, 3, 16, 16)
    out_v = model(ids, pixel_values_videos=videos)
    assert out_v.logits.shape[1] == 4 + 2


if __name__ == '__main__':
    tests = [
        test_default_model_keeps_full_attention,
        test_k3_layer_pattern_and_independent_params,
        test_last_layer_is_always_full_attention,
        test_forward_backward_and_causal,
        test_kda_cache_matches_full_prefill,
        test_k3_bounded_forget_gate,
        test_latent_moe_hybrid,
        test_wy_scan_matches_naive,
        test_triton_is_optional_python_fallback,
        test_quantile_balancing_eval_freezes_bias,
        test_qat_routed_expert_forward,
        test_per_head_muon_step,
        test_moonvit_prepends_vision_tokens,
    ]
    for fn in tests:
        fn()
        print(f'ok  {fn.__name__}')
    print('all tests passed')
