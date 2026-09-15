"""Numerically checked utilities shared by the positive-control producers."""
from __future__ import annotations

import torch

from .null import gaussian_haar_frame


def _append_projected_columns(
    basis: torch.Tensor,
    candidates: torch.Tensor,
    target: int,
    tolerance: float,
) -> torch.Tensor:
    """Order-preserving, twice-reorthogonalised Gram--Schmidt completion."""
    for column in candidates.T:
        if basis.shape[1] >= target:
            break
        vector = column
        if basis.numel():
            vector = vector - basis @ (basis.T @ vector)
            vector = vector - basis @ (basis.T @ vector)
        norm = torch.linalg.vector_norm(vector)
        if float(norm) > tolerance:
            basis = torch.cat((basis, (vector / norm).unsqueeze(1)), dim=1)
    return basis


def plant_right_directions(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    planted: int,
    k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Replace leading right directions while preserving candidate SVD data.

    Completion starts at ``candidate_v[:, planted:]``: the candidate directions
    being replaced are discarded, the rest retain their order, and only a true
    rank deficiency is filled with a Haar frame in the remaining complement.
    """
    if planted == 0:
        return candidate
    if planted < 0 or planted > k:
        raise ValueError(f"planted={planted} must lie in [0, k={k}]")
    if not (reference.is_floating_point() and candidate.is_floating_point()):
        raise TypeError("reference and candidate must be floating-point matrices")

    source_eps = torch.finfo(candidate.dtype).eps
    reference_eps = torch.finfo(reference.dtype).eps
    work_dtype = torch.float64
    ref = reference.to(work_dtype)
    cand = candidate.to(work_dtype)
    _, _, reference_vh = torch.linalg.svd(ref, full_matrices=False)
    candidate_u, candidate_s, candidate_vh = torch.linalg.svd(cand, full_matrices=False)

    if candidate_s.numel() == 0 or float(candidate_s[0]) == 0.0:
        raise ValueError("candidate has numerical rank zero")
    rank_tolerance = max(cand.shape) * source_eps * float(candidate_s[0])
    rank = int((candidate_s > rank_tolerance).sum())
    if rank < k or planted > rank:
        raise ValueError(
            f"candidate numerical rank {rank} cannot support k={k}, planted={planted}"
        )

    reference_s = torch.linalg.svdvals(ref)
    if reference_s.numel() == 0 or float(reference_s[0]) == 0.0:
        raise ValueError("reference has numerical rank zero")
    reference_tolerance = max(ref.shape) * reference_eps * float(reference_s[0])
    reference_rank = int((reference_s > reference_tolerance).sum())
    if reference_rank < planted:
        raise ValueError(
            f"reference numerical rank {reference_rank} cannot supply planted={planted}"
        )

    keep = reference_vh.T[:, :planted]
    candidate_v = candidate_vh.T[:, :rank]
    eye_keep = torch.eye(planted, dtype=work_dtype, device=keep.device)
    torch.testing.assert_close(keep.T @ keep, eye_keep, rtol=1e-7, atol=1e-8)

    eps = torch.finfo(work_dtype).eps
    scale = max(1.0, float(torch.linalg.matrix_norm(candidate_v, ord=2)))
    tolerance = 100.0 * eps * max(candidate_v.shape) * scale
    basis = _append_projected_columns(keep, candidate_v[:, planted:], rank, tolerance)

    while basis.shape[1] < rank:
        missing = rank - basis.shape[1]
        gaussian = torch.randn(
            candidate.shape[1], missing, generator=generator,
            device=candidate.device, dtype=work_dtype,
        )
        gaussian = gaussian - basis @ (basis.T @ gaussian)
        frame = gaussian_haar_frame(gaussian)
        before = basis.shape[1]
        basis = _append_projected_columns(basis, frame, rank, tolerance)
        if basis.shape[1] == before:
            raise RuntimeError("failed to Haar-complete the planted right basis")

    candidate_u = candidate_u[:, :rank]
    candidate_s = candidate_s[:rank]
    planted_matrix = ((candidate_u * candidate_s) @ basis.T).to(reference)

    identity = torch.eye(rank, dtype=work_dtype, device=basis.device)
    torch.testing.assert_close(basis.T @ basis, identity, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        basis[:, :planted].T @ keep, eye_keep, rtol=2e-6, atol=2e-7
    )
    torch.testing.assert_close(
        planted_matrix.double().norm(), candidate_s.norm(), rtol=2e-5,
        atol=2e-7 * max(1.0, float(candidate_s[0])),
    )
    return planted_matrix


def materialize_in_adapter_row_space(
    proposed: torch.Tensor, adapter_a: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a planted matrix into row(A), then rederive its LoRA B factor."""
    if scale == 0:
        raise ValueError("LoRA scale must be nonzero")
    pinv = torch.linalg.pinv(adapter_a.double()).to(proposed)
    proposed_b = (proposed @ pinv) / scale
    materialized = scale * (proposed_b @ adapter_a.to(proposed))
    materialized_b = (materialized @ pinv) / scale
    reconstructed = scale * (materialized_b @ adapter_a.to(proposed))
    torch.testing.assert_close(materialized, reconstructed, rtol=3e-4, atol=3e-6)
    row_residual = materialized - (materialized @ pinv) @ adapter_a.to(proposed)
    relative = float(row_residual.norm() / torch.clamp(materialized.norm(), min=1e-30))
    if relative > 5e-5:
        raise AssertionError(f"materialized plant is not in row(A): relative residual {relative:g}")
    return materialized, materialized_b


def initial_coupling_destroying_null(
    dW: torch.Tensor,
    V: torch.Tensor,
    n_blocks: int,
    generator: torch.Generator,
    tail_rank: int = 128,
    blocks: list[list[int]] | None = None,
) -> torch.Tensor:
    """Activation null as first written, with only its two intended defects.

    Singleton blocks are skipped and only a ``tail_rank``-dimensional slice of
    the complement is rotated. All Gaussian QR draws nevertheless use the same
    sign-corrected Haar sampler as the current null, so this comparison does not
    confound those historical defects with an implementation artefact.
    """
    _, d_in = dW.shape
    m = V.shape[0]
    vd = V.to(dW)
    coordinates = dW @ vd.T
    residual = dW - coordinates @ vd
    if blocks is None:
        edges = torch.linspace(0, m, n_blocks + 1).round().long().tolist()
        blocks = [list(range(start, stop)) for start, stop in zip(edges[:-1], edges[1:])]
    for block in blocks:
        if len(block) < 2:
            continue
        index = torch.tensor(block, device=dW.device)
        rotation = gaussian_haar_frame(
            torch.randn(
                len(block), len(block), generator=generator,
                device=dW.device, dtype=torch.float32,
            )
        )
        coordinates[:, index] = coordinates[:, index] @ rotation.to(dW)

    width = min(tail_rank, max(d_in - m, 0))
    if width >= 2:
        frame = torch.randn(
            d_in, width, generator=generator, device=dW.device, dtype=torch.float32
        )
        frame = frame - vd.T.float() @ (vd.float() @ frame)
        frame = gaussian_haar_frame(frame).to(dW)
        rotation = gaussian_haar_frame(
            torch.randn(
                width, width, generator=generator,
                device=dW.device, dtype=torch.float32,
            )
        ).to(dW)
        projected = residual @ frame
        residual = residual - projected @ frame.T + (projected @ rotation) @ frame.T
    return coordinates @ vd + residual


def initial_lora_shared_init_null(
    adapter_a: torch.Tensor,
    adapter_b: torch.Tensor,
    scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """LoRA null as first written: iid B matched only in Frobenius norm."""
    gaussian = torch.randn(
        adapter_b.shape, generator=generator,
        device=adapter_b.device, dtype=torch.float32,
    )
    gaussian = gaussian * (adapter_b.norm() / torch.clamp(gaussian.norm(), min=1e-12))
    return (gaussian @ adapter_a) * scale


__all__ = [
    "initial_coupling_destroying_null",
    "initial_lora_shared_init_null",
    "materialize_in_adapter_row_space",
    "plant_right_directions",
]
