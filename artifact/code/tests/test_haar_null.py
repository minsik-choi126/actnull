#!/usr/bin/env python3
"""Fast deterministic checks for the real Gaussian Haar/Stiefel sampler."""

import torch

from actnull.null import coupling_destroying_null, gaussian_haar_frame


def test_sign_symmetry_and_twirl() -> None:
    gen = torch.Generator().manual_seed(20260915)
    draws, n, projector_rank = 8192, 5, 2
    q = gaussian_haar_frame(torch.randn(draws, n, n, generator=gen))

    positive_fraction = float((q[:, 0, 0] > 0).float().mean())
    assert 0.47 < positive_fraction < 0.53, positive_fraction

    projector = torch.zeros(n, n)
    projector[:projector_rank, :projector_rank] = torch.eye(projector_rank)
    twirl = (q.transpose(-2, -1) @ projector @ q).mean(0)
    target = torch.eye(n) * (projector_rank / n)
    assert torch.allclose(twirl, target, atol=0.015, rtol=0.0), (twirl, target)


def test_frame_and_coupling_invariants() -> None:
    gen = torch.Generator().manual_seed(7)
    frame = gaussian_haar_frame(torch.randn(13, 5, generator=gen))
    assert torch.allclose(frame.T @ frame, torch.eye(5), atol=2e-6, rtol=0.0)

    d_out, d_in, m = 4, 11, 5
    V = gaussian_haar_frame(torch.randn(d_in, m, generator=gen)).T
    dW = torch.randn(d_out, d_in, generator=gen)
    blocks = [[0, 1], [2], [3, 4]]
    randomized = coupling_destroying_null(
        dW, V, n_blocks=len(blocks), gen=gen, blocks=blocks
    )

    assert torch.allclose(randomized @ randomized.T, dW @ dW.T,
                          atol=2e-5, rtol=2e-6)
    for block in blocks:
        vb = V[block]
        projector = vb.T @ vb
        before = dW @ projector @ dW.T
        after = randomized @ projector @ randomized.T
        assert torch.allclose(after, before, atol=2e-5, rtol=2e-6)


if __name__ == "__main__":
    test_sign_symmetry_and_twirl()
    test_frame_and_coupling_invariants()
    print("Haar/Stiefel symmetry, twirl, and coupling invariants: PASS")
