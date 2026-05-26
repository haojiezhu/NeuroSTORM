"""Unified TPT (Task-specific Parameter-efficient Tuning) module.

Strategies:
    - 'prompt'  : VPT-Deep, train per-block prompt tokens + output head
    - 'ln'      : LN-tuning, train all LayerNorm gamma/beta + output head
    - 'linear'  : Linear probe, train output head only

Use via apply_tpt(lightning_model, strategy). Returns a dict with parameter
counts and prints a summary.
"""
from __future__ import annotations


def _is_trainable_param(name: str, strategy: str) -> bool:
    if strategy == 'prompt':
        return 'prompt' in name
    if strategy == 'ln':
        n = name.lower()
        return 'norm' in n or '.ln' in n
    if strategy == 'linear':
        return False
    if strategy == 'prompt_ln':
        n = name.lower()
        return 'prompt' in name or 'norm' in n or '.ln' in n
    raise ValueError(f"unknown TPT strategy: {strategy}")


def apply_tpt(lightning_model, strategy: str) -> dict:
    """Freeze backbone; mark per-strategy params trainable. Output head stays
    trainable since it's a separate module under `lightning_model`.
    """
    backbone = lightning_model.model
    backbone.eval()

    n_tunable_bk, n_frozen_bk = 0, 0
    for name, p in backbone.named_parameters():
        trainable = _is_trainable_param(name, strategy)
        p.requires_grad = trainable
        if trainable:
            n_tunable_bk += p.numel()
        else:
            n_frozen_bk += p.numel()

    n_head = 0
    head = getattr(lightning_model, 'output_head', None)
    if head is not None:
        n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)

    n_trainable = n_tunable_bk + n_head
    n_full = n_tunable_bk + n_frozen_bk + n_head
    pct = (100.0 * n_trainable / n_full) if n_full > 0 else 0.0

    tag = f'[TPT={strategy}]'
    print(f'{tag} tunable backbone: {n_tunable_bk:>12,}')
    print(f'{tag} head params:      {n_head:>12,}')
    print(f'{tag} trainable total:  {n_trainable:>12,}')
    print(f'{tag} frozen backbone:  {n_frozen_bk:>12,}')
    print(f'{tag} full model:       {n_full:>12,}')
    print(f'{tag} trainable ratio:  {pct:.2f}%')

    if strategy == 'prompt' and n_tunable_bk == 0:
        print(f'{tag} WARNING: no params with "prompt" in name. '
              f'Set --prompt_len > 0 with --model neurostorm.')

    return {
        'strategy': strategy,
        'n_tunable_backbone': n_tunable_bk,
        'n_head': n_head,
        'n_trainable': n_trainable,
        'n_frozen': n_frozen_bk,
        'n_full': n_full,
        'ratio': pct,
    }
