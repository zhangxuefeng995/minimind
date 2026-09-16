"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM

HYBRID_ATTN_ARG_HELP = "是否启用Kimi-K3风格混合注意力（3层KDA+1层全注意力；KDA层独立参数，默认0）"


def add_hybrid_attn_args(parser):
    parser.add_argument('--use_hybrid_attn', default=0, type=int, choices=[0, 1], help=HYBRID_ATTN_ARG_HELP)
    parser.add_argument('--linear_attn_ratio', default=3, type=int, help="每组中KDA层数（默认3，即3:1）")
    parser.add_argument('--use_per_head_muon', default=0, type=int, choices=[0, 1],
                        help="是否对注意力 Q/K/V 按头做 Newton–Schulz Muon（K3 Per-Head Muon）")
    parser.add_argument('--use_qat', default=0, type=int, choices=[0, 1],
                        help="是否对 routed expert 做 MXFP4/MXFP8 伪量化 QAT")
    parser.add_argument('--use_vision', default=0, type=int, choices=[0, 1],
                        help="是否构建 MoonViT-V2 视觉塔（需同时传入 pixel_values）")
    parser.add_argument('--kda_use_triton', default=1, type=int, choices=[0, 1],
                        help="KDA 在 CUDA+Triton 可用时走 fused recurrent；未安装则自动回退 PyTorch")


def hybrid_attn_kwargs(args):
    kwargs = dict(
        use_hybrid_attn=bool(getattr(args, 'use_hybrid_attn', 0)),
        linear_attn_ratio=int(getattr(args, 'linear_attn_ratio', 3)),
    )
    if hasattr(args, 'use_qat'):
        kwargs['use_qat'] = bool(args.use_qat)
    if hasattr(args, 'use_vision'):
        kwargs['use_vision'] = bool(args.use_vision)
    if hasattr(args, 'kda_use_triton'):
        kwargs['kda_use_triton'] = bool(args.kda_use_triton)
    return kwargs

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


def newton_schulz_(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Muon 用的 Newton–Schulz 正交化（近似 zeroth power / SVD 符号函数）。

    把 2D 梯度推到接近半正交，使更新步长与谱范数解耦。K3 的 Per-Head Muon
    是对 **每个头的 Q/K/V 矩阵** 分别做这一步，而不是整张 [n_heads*d, hidden]。
    """
    x = grad.float()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        a_mat = x @ x.T
        x = a * x + (b * a_mat + c * a_mat @ a_mat) @ x
    if transposed:
        x = x.T
    return x.to(dtype=grad.dtype)


def _is_per_head_attn_weight(name: str, param: torch.nn.Parameter, n_heads: int) -> bool:
    if param.ndim != 2 or 'self_attn' not in name or not name.endswith('.weight'):
        return False
    if 'conv' in name:
        return False
    # 只拆 Q/K/V/O 以及 MLA 的 g_proj / kv_b_proj；kv_a 是共享潜空间，不能按头切
    leaf = name.rsplit('.', 1)[0].rsplit('.', 1)[-1]
    if leaf not in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'g_proj', 'kv_b_proj'):
        return False
    return param.shape[0] % n_heads == 0


class PerHeadMuonAdamW:
    """K3 风格混合优化器：注意力投影按头 Muon，其余参数 AdamW。

    Muon 参数仍挂在 AdamW 的 param_groups 里（lr=0），这样 GradScaler.unscale_
    和梯度裁剪能看见它们；真正的更新在 ``step`` 里用 Newton–Schulz 完成。
    """

    def __init__(self, model, lr: float, weight_decay: float = 0.1, n_heads: int = None):
        raw = model.module if hasattr(model, 'module') else model
        raw = getattr(raw, '_orig_mod', raw)
        cfg = getattr(raw, 'config', None)
        n_heads = n_heads or getattr(cfg, 'num_attention_heads', 8)
        muon_params, adam_params = [], []
        for name, param in raw.named_parameters():
            if not param.requires_grad:
                continue
            if _is_per_head_attn_weight(name, param, n_heads):
                muon_params.append(param)
            else:
                adam_params.append(param)
        self.n_heads = n_heads
        self.muon_params = muon_params
        if not adam_params:
            # AdamW 至少需要一个 param group；Muon 全覆盖时仍保留空组供 scaler 接口
            self.adam = torch.optim.AdamW([{'params': muon_params[:1], 'lr': 0.0, 'weight_decay': 0.0}])
            groups = []
        else:
            self.adam = torch.optim.AdamW(adam_params, lr=lr, weight_decay=weight_decay)
            groups = list(self.adam.param_groups)
        if muon_params:
            groups.append({'params': muon_params, 'lr': lr, 'weight_decay': 0.0})
        self.param_groups = groups

    def zero_grad(self, set_to_none: bool = True):
        self.adam.zero_grad(set_to_none=set_to_none)
        for p in self.muon_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def state_dict(self):
        return {'adam': self.adam.state_dict(), 'n_heads': self.n_heads}

    def load_state_dict(self, state):
        if isinstance(state, dict) and 'adam' in state:
            self.adam.load_state_dict(state['adam'])
        else:
            self.adam.load_state_dict(state)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            closure()
        self.adam.step()
        for p in self.muon_params:
            if p.grad is None:
                continue
            grad = p.grad
            n_heads = self.n_heads
            if grad.shape[0] % n_heads == 0:
                per_head = grad.shape[0] // n_heads
                g = grad.view(n_heads, per_head, grad.shape[1])
                updates = [newton_schulz_(g[h]) for h in range(n_heads)]
                update = torch.stack(updates, dim=0).view_as(p)
            else:
                update = newton_schulz_(grad)
            scale = max(1.0, (p.shape[1] / max(p.shape[0], 1)) ** 0.5)
            muon_lr = self.param_groups[-1]['lr'] if self.param_groups else 0.0
            p.add_(update, alpha=-muon_lr * scale)


def build_optimizer(model, args):
    """默认 AdamW；``--use_per_head_muon 1`` 时对注意力矩阵走 Per-Head Muon。"""
    lr = args.learning_rate
    if int(getattr(args, 'use_per_head_muon', 0)) == 1:
        Logger('Per-Head Muon enabled for attention projections')
        return PerHeadMuonAdamW(model, lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr)