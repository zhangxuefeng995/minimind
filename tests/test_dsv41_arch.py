"""DeepSeek-V4.1 风格 MiniMind 架构自检（CPU）。"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.dsv41 import (
    CEDDecoderKV, CSA2DecoderAttention, CSA2EncoderAttention,
    build_dsv41_layout, fake_quant_fp4_e2m1, compress_tokens,
)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, Attention


def _tiny_cfg(**kwargs):
    defaults = dict(
        hidden_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=256,
        dropout=0.0,
        flash_attn=False,
        sliding_window=8,
        csa2_index_topk=4,
        csa2_index_n_heads=2,
        csa2_index_head_dim=16,
        csa2_candidate_block_size=2,
        csa2_candidate_topk_blocks=2,
        hc_mult=2,
        engram_table_size=64,
        engram_n_heads=2,
        engram_head_dim=8,
        engram_max_ngram_size=3,
        use_dsv41=True,
    )
    defaults.update(kwargs)
    return MiniMindConfig(**defaults)


def test_layout_8_and_40():
    lay8 = build_dsv41_layout(8)
    assert lay8['n_enc'] == 4
    assert lay8['modes'][:2] == ['swa', 'swa']
    assert lay8['modes'][2] == 'full'
    assert lay8['modes'][3] == 'reuse'
    assert lay8['modes'][4] == 'full'
    assert 'reindex' in lay8['modes']
    assert lay8['compress_ratios'][2] == 2
    assert lay8['compress_ratios'][4] == 1
    assert lay8['candidate_source_layer_id'] == 4

    lay40 = build_dsv41_layout(40)
    assert lay40['n_enc'] == 20
    assert lay40['kv_source_layer_ids'][0] == 2
    assert 20 in lay40['kv_source_layer_ids']
    assert lay40['candidate_source_layer_id'] == 20
    assert lay40['modes'][20] == 'full'
    assert lay40['compress_ratios'][:2] == [0, 0]
    assert lay40['compress_ratios'][2] == 2
    assert lay40['compress_ratios'][20] == 1


def test_default_gqa_unchanged():
    cfg = MiniMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=32)
    model = MiniMindForCausalLM(cfg)
    assert not cfg.use_dsv41
    assert isinstance(model.model.layers[0].self_attn, Attention)
    ids = torch.randint(0, 32, (2, 7))
    out = model(ids)
    assert out.logits.shape == (2, 7, 32)


def test_encoder_decoder_attn_split():
    cfg = _tiny_cfg()
    model = MiniMindForCausalLM(cfg)
    n_enc = cfg.dsv41_layout['n_enc']
    modes = cfg.dsv41_layout['modes']
    for i, layer in enumerate(model.model.layers):
        if i < n_enc:
            assert isinstance(layer.self_attn, CSA2EncoderAttention)
            assert not isinstance(layer.self_attn, CSA2DecoderAttention)
        else:
            assert isinstance(layer.self_attn, CSA2DecoderAttention)
    enc_full = next(i for i, m in enumerate(modes) if m == 'full' and i < n_enc)
    dec_full = next(i for i, m in enumerate(modes) if m == 'full' and i >= n_enc)
    enc_attn = model.model.layers[enc_full].self_attn
    dec_attn = model.model.layers[dec_full].self_attn
    assert hasattr(enc_attn, 'k_proj') and hasattr(enc_attn, 'v_proj')
    assert not hasattr(enc_attn, 'dec_kv')
    assert isinstance(dec_attn.dec_kv, CEDDecoderKV)
    assert isinstance(dec_attn.dec_kv.proj, torch.nn.Linear)
    assert not hasattr(dec_attn, 'k_proj')
    assert not hasattr(dec_attn, 'v_proj')
    kv_dim = dec_attn.n_kv * dec_attn.head_dim
    assert dec_attn.dec_kv.proj.out_features == kv_dim * 2


def test_dsv41_forward_and_backward():
    cfg = _tiny_cfg()
    model = MiniMindForCausalLM(cfg)
    assert cfg.dsv41_layout['modes'][2] == 'full'
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is not None
    out.loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert sum(grads) > 0


def test_dsv41_moe_noaux():
    cfg = _tiny_cfg(use_moe=True, n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1)
    assert cfg.scoring_func == 'sigmoid'
    assert cfg.aux_loss_alpha == 0.0
    model = MiniMindForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(ids, labels=ids)
    assert out.logits.shape[-1] == cfg.vocab_size
    # 无辅助损失路径仍返回标量 aux_loss（为 0）
    assert float(out.aux_loss) == 0.0 or out.aux_loss.numel() == 1
    out.loss.backward()


def test_fp4_and_compress():
    x = torch.randn(2, 8, 4, 16)
    q = fake_quant_fp4_e2m1(x, group_size=16)
    assert q.shape == x.shape
    y = compress_tokens(torch.randn(2, 7, 3, 4), ratio=2)
    assert y.shape[1] == 4  # pad to 8 then /2


def test_cache_last_token_matches():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = MiniMindForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full = model(ids).logits[:, -1]
        pre = model(ids[:, :-1], use_cache=True)
        dec = model(ids[:, -1:], past_key_values=pre.past_key_values, use_cache=True).logits[:, -1]
    max_err = (full - dec).abs().max().item()
    assert max_err < 5e-3, max_err


def test_generate_smoke():
    cfg = _tiny_cfg()
    model = MiniMindForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    out = model.generate(ids, max_new_tokens=4, do_sample=False)
    assert out.shape[1] == 10


if __name__ == '__main__':
    tests = [
        test_layout_8_and_40,
        test_encoder_decoder_attn_split,
        test_default_gqa_unchanged,
        test_dsv41_forward_and_backward,
        test_dsv41_moe_noaux,
        test_fp4_and_compress,
        test_cache_last_token_matches,
        test_generate_smoke,
    ]
    for fn in tests:
        fn()
        print('ok', fn.__name__)
    print('all dsv41 tests passed')
