"""
Confidence-Gated Teacher Distillation (SDAR) utilities.

Token-level gated distillation loss where the gate is derived from
the reference-target log-probability gap. In dual-view variants, the
target is whichever prompt view is being optimized, and the reference
is detached.
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_sdar_loss(
    target_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gate_delta_sign: torch.Tensor | None = None,
    loss_weight: torch.Tensor | None = None,
    direction_ids: torch.Tensor | None = None,
    gate_enable: bool = True,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Confidence-Gated Teacher Distillation loss.

    L_SDAR = agg( g_t * (log pi_reference - log pi_target) )
    where g_t = sigmoid(beta * delta_t).
    By default delta_t = log pi_reference - log pi_target.
    ``gate_delta_sign`` can flip delta_t for dual-view training while leaving
    the optimized target/reference direction unchanged.

    The gate g_t and reference log-probs are detached, so gradients only flow
    through the target log-probs.

    Args:
        target_log_probs: (bs, response_length) - current prompt-view log probs.
            This is the policy view being optimized and retains gradients.
        reference_log_probs: (bs, response_length) - opposite/reference-view
            log probs. This tensor is detached.
        response_mask: (bs, response_length) - mask for valid response tokens.
        loss_weight: optional per-sample or per-token distillation weight.
        direction_ids: optional per-sample or per-token direction ids.
            1 = no-skill learns from skill.
            2 = skill learns from no-skill.
            3 = successful no-skill RLRT against skill.
            4 = successful skill RLRT against no-skill.
        gate_beta: temperature for the sigmoid gate. Higher = sharper gating.
        loss_agg_mode: aggregation mode passed to agg_loss.

    Returns:
        sdar_loss: scalar loss.
        metrics: dict with gating statistics.
    """
    reference_log_probs = reference_log_probs.detach()

    delta_t = reference_log_probs - target_log_probs.detach()
    if gate_delta_sign is not None:
        if gate_delta_sign.dim() == 1:
            gate_delta_sign = gate_delta_sign.unsqueeze(-1)
        delta_t = delta_t * gate_delta_sign.to(device=delta_t.device, dtype=delta_t.dtype)

    gate = torch.sigmoid(gate_beta * delta_t).detach()
    if not gate_enable:
        gate = torch.ones_like(gate)

    kl_per_token = reference_log_probs - target_log_probs

    gated_kl = gate * kl_per_token
    if loss_weight is not None:
        if loss_weight.dim() == 1:
            loss_weight = loss_weight.unsqueeze(-1)
        loss_weight = loss_weight.to(device=gated_kl.device, dtype=gated_kl.dtype)
        gated_kl = gated_kl * loss_weight

    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        gate_mean = (gate * response_mask).sum() / mask_sum
        gate_active = ((gate > 0.5).float() * response_mask).sum() / mask_sum
        gap_mean = (delta_t * response_mask).sum() / mask_sum
        if loss_weight is None:
            loss_active_ratio = torch.ones((), device=response_mask.device, dtype=response_mask.dtype)
        else:
            loss_active_ratio = ((loss_weight > 0).float() * response_mask).sum() / mask_sum

    metrics = {
        "sdar/gate_mean": gate_mean.item(),
        "sdar/gate_active_ratio": gate_active.item(),
        "sdar/gate_delta_mean": gap_mean.item(),
        "sdar/loss_active_ratio": loss_active_ratio.item(),
        "sdar/teacher_gap_mean": gap_mean.item(),
        "sdar/loss": loss.detach().item(),
    }

    if direction_ids is not None:
        if direction_ids.dim() == 1:
            direction_ids = direction_ids.unsqueeze(-1)
        direction_ids = direction_ids.to(device=response_mask.device)
        if direction_ids.shape != response_mask.shape:
            direction_ids = direction_ids.expand_as(response_mask)

        def _direction_loss(direction_id: int) -> float:
            direction_mask = response_mask * (direction_ids == direction_id).to(dtype=response_mask.dtype)
            if direction_mask.sum().item() == 0:
                return 0.0
            return agg_loss(loss_mat=gated_kl, loss_mask=direction_mask, loss_agg_mode=loss_agg_mode).detach().item()

        metrics["sdar/loss_no_skill_from_skill"] = _direction_loss(1)
        metrics["sdar/loss_skill_from_no_skill"] = _direction_loss(2)
        metrics["sdar/loss_no_skill_rlrt_from_skill"] = _direction_loss(3)
        metrics["sdar/loss_skill_rlrt_from_no_skill"] = _direction_loss(4)
        metrics["sdar/loss_correct"] = metrics["sdar/loss_no_skill_from_skill"] + metrics["sdar/loss_skill_from_no_skill"]
        metrics["sdar/loss_rlrt"] = metrics["sdar/loss_no_skill_rlrt_from_skill"] + metrics["sdar/loss_skill_rlrt_from_no_skill"]

    return loss, metrics


def compute_dsdar_topk_loss(
    target_logits_k: torch.Tensor,
    target_logsumexp: torch.Tensor,
    reference_logits_k: torch.Tensor,
    reference_logsumexp: torch.Tensor,
    response_mask: torch.Tensor,
    target_log_probs: torch.Tensor | None = None,
    reference_log_probs: torch.Tensor | None = None,
    gate_delta_sign: torch.Tensor | None = None,
    loss_weight: torch.Tensor | None = None,
    direction_ids: torch.Tensor | None = None,
    gate_enable: bool = True,
    gate_beta: float = 5.0,
    norm_to_one: bool = True,
    clip_log_ratio: bool = False,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Teacher-TopK correction loss for adaptive DSDAR.

    This loss is used only for correction directions:
        1 = no-skill learns from skill.
        2 = skill learns from no-skill.

    The reference/teacher top-k distribution is detached. Gradients only update
    the target prompt-view logits gathered at the teacher's top-k token ids.
    A sampled-token confidence gate can be applied to avoid treating every
    token in a winning trajectory as equally reliable.
    """
    assert target_logits_k.shape == reference_logits_k.shape
    assert target_logits_k.dim() == 3
    assert target_logsumexp.shape == response_mask.shape
    assert reference_logsumexp.shape == response_mask.shape

    reference_logits_k = reference_logits_k.detach()
    reference_logsumexp = reference_logsumexp.detach()

    target_logits_k = target_logits_k.float()
    reference_logits_k = reference_logits_k.float()
    target_logsumexp = target_logsumexp.float()
    reference_logsumexp = reference_logsumexp.float()

    if norm_to_one:
        target_log_probs_k = torch.nn.functional.log_softmax(target_logits_k, dim=-1)
        reference_log_probs_k = torch.nn.functional.log_softmax(reference_logits_k, dim=-1)
    else:
        target_log_probs_k = target_logits_k - target_logsumexp.unsqueeze(-1)
        reference_log_probs_k = reference_logits_k - reference_logsumexp.unsqueeze(-1)

    reference_probs_k = reference_log_probs_k.exp().detach()
    log_ratio = reference_log_probs_k.detach() - target_log_probs_k
    if clip_log_ratio:
        log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)

    topk_kl = (reference_probs_k * log_ratio).sum(dim=-1)

    gate = None
    if gate_enable and target_log_probs is not None and reference_log_probs is not None:
        delta_t = reference_log_probs.detach() - target_log_probs.detach()
        if gate_delta_sign is not None:
            if gate_delta_sign.dim() == 1:
                gate_delta_sign = gate_delta_sign.unsqueeze(-1)
            gate_delta_sign = gate_delta_sign.to(device=delta_t.device, dtype=delta_t.dtype)
            if gate_delta_sign.shape != delta_t.shape:
                gate_delta_sign = gate_delta_sign.expand_as(delta_t)
            delta_t = delta_t * gate_delta_sign
        gate = torch.sigmoid(float(gate_beta) * delta_t).detach()
        topk_kl = topk_kl * gate

    if direction_ids is not None:
        if direction_ids.dim() == 1:
            direction_ids = direction_ids.unsqueeze(-1)
        direction_ids = direction_ids.to(device=response_mask.device)
        if direction_ids.shape != response_mask.shape:
            direction_ids = direction_ids.expand_as(response_mask)
        correction_mask = ((direction_ids == 1) | (direction_ids == 2)).to(dtype=response_mask.dtype)
    else:
        correction_mask = torch.ones_like(response_mask)

    weighted_topk_kl = topk_kl * correction_mask
    if loss_weight is not None:
        if loss_weight.dim() == 1:
            loss_weight = loss_weight.unsqueeze(-1)
        loss_weight = loss_weight.to(device=response_mask.device, dtype=weighted_topk_kl.dtype)
        weighted_topk_kl = weighted_topk_kl * loss_weight
        active_mask = correction_mask * (loss_weight > 0).to(dtype=response_mask.dtype)
    else:
        active_mask = correction_mask

    loss = agg_loss(loss_mat=weighted_topk_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        active_ratio = (active_mask * response_mask).sum() / mask_sum
        if gate is None:
            gate_mean = torch.ones((), device=response_mask.device, dtype=weighted_topk_kl.dtype)
            gate_active_ratio = torch.ones((), device=response_mask.device, dtype=weighted_topk_kl.dtype)
        else:
            gate_mean = (gate * response_mask).sum() / mask_sum
            gate_active_ratio = ((gate > 0.5).to(dtype=response_mask.dtype) * response_mask).sum() / mask_sum

    metrics = {
        "sdar/topk_loss": loss.detach().item(),
        "sdar/topk_active_ratio": active_ratio.item(),
        "sdar/topk_gate_mean": gate_mean.item(),
        "sdar/topk_gate_active_ratio": gate_active_ratio.item(),
    }

    if direction_ids is not None:
        def _direction_loss(direction_id: int) -> float:
            direction_mask = response_mask * (direction_ids == direction_id).to(dtype=response_mask.dtype)
            if direction_mask.sum().item() == 0:
                return 0.0
            return agg_loss(
                loss_mat=weighted_topk_kl,
                loss_mask=direction_mask,
                loss_agg_mode=loss_agg_mode,
            ).detach().item()

        metrics["sdar/topk_loss_no_skill_from_skill"] = _direction_loss(1)
        metrics["sdar/topk_loss_skill_from_no_skill"] = _direction_loss(2)

    return loss, metrics
