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
    assert 'routed_down_proj.weight' in dict(mlp.named_parameters())
    ids = torch.randint(0, 128, (2, 6))
    out = model(ids, labels=ids)
    out.loss.backward()
    assert out.logits.shape == (2, 6, 128)


if __name__ == '__main__':
    tests = [
        test_default_model_keeps_full_attention,
        test_k3_layer_pattern_and_independent_params,
        test_last_layer_is_always_full_attention,
        test_forward_backward_and_causal,
        test_kda_cache_matches_full_prefill,
        test_k3_bounded_forget_gate,
        test_latent_moe_hybrid,
    ]
    for fn in tests:
        fn()
        print(f'ok  {fn.__name__}')
    print('all tests passed')
