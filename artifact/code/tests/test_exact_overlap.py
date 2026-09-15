"""Sanity checks for the analytic Haar block-null expectation."""
from __future__ import annotations

import unittest

import torch

from scripts.exact_overlap import exact_block_null, projector_block_masses


def haar_qr(matrix: torch.Tensor) -> torch.Tensor:
    """QR Haar draw with the signs of diag(R) transferred to Q."""
    q, r = torch.linalg.qr(matrix)
    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    one = torch.ones((), dtype=diagonal.dtype, device=diagonal.device)
    phase = torch.where(diagonal >= 0, one, -one)
    return q * phase.unsqueeze(-2)


def random_frame(d: int, k: int, generator: torch.Generator) -> torch.Tensor:
    return haar_qr(torch.randn(d, k, dtype=torch.float64, generator=generator))


class ExactBlockNullTest(unittest.TestCase):
    def test_qr_sign_correction_removes_householder_orientation_bias(self) -> None:
        generator = torch.Generator().manual_seed(31)
        matrices = torch.randn(4096, 2, 2, dtype=torch.float64, generator=generator)
        raw_q = torch.linalg.qr(matrices)[0]
        corrected_q = haar_qr(matrices)
        # LAPACK's Householder convention fixes one hemisphere.  Moving the
        # random signs of diag(R) into Q restores the Q -> -Q symmetry of Haar.
        self.assertGreater(abs(float(raw_q[:, 0, 0].mean())), 0.4)
        self.assertLess(abs(float(corrected_q[:, 0, 0].mean())), 0.04)

    def test_single_full_block_reduces_to_isotropic_null(self) -> None:
        d, k = 13, 4
        # A rank-k projector has total block mass k for the one full block.
        self.assertAlmostEqual(exact_block_null([k], [k], [d], k), k / d, places=14)

    def test_masses_include_the_full_complement(self) -> None:
        generator = torch.Generator().manual_seed(7)
        d, k, m = 12, 3, 5
        ambient_basis = random_frame(d, d, generator)
        activation_basis = ambient_basis[:, :m].T
        subspace = random_frame(d, k, generator)
        masses, dims = projector_block_masses(
            subspace, activation_basis, [[0], [1, 2], [3, 4]], d
        )
        self.assertEqual(dims, [1, 2, 2, 7])
        self.assertAlmostEqual(sum(masses), k, places=11)

    def test_formula_matches_sign_corrected_haar_monte_carlo(self) -> None:
        generator = torch.Generator().manual_seed(20260915)
        d, k, m = 11, 3, 5
        blocks = [[0, 1], [2, 3, 4]]
        dims = [2, 3, d - m]
        ambient_basis = random_frame(d, d, generator)
        activation_basis = ambient_basis[:, :m].T
        a = random_frame(d, k, generator)
        b = random_frame(d, k, generator)
        mass_a, got_dims = projector_block_masses(a, activation_basis, blocks, d)
        mass_b, _ = projector_block_masses(b, activation_basis, blocks, d)
        self.assertEqual(got_dims, dims)
        expected = exact_block_null(mass_a, mass_b, dims, k)

        # Draw independent corrected-Haar maps for the two tasks inside every
        # activation block and the entire complement.  Coordinate blocks suffice:
        # conjugating the construction by a fixed orthogonal basis changes neither
        # side of the identity under test.
        draws = 12_000
        qa = torch.zeros(draws, d, d, dtype=torch.float64)
        qb = torch.zeros_like(qa)
        offset = 0
        for width in dims:
            sl = slice(offset, offset + width)
            qa[:, sl, sl] = haar_qr(
                torch.randn(draws, width, width, dtype=torch.float64, generator=generator)
            )
            qb[:, sl, sl] = haar_qr(
                torch.randn(draws, width, width, dtype=torch.float64, generator=generator)
            )
            offset += width
        # Work in the random activation basis.  This also checks the row/column
        # convention used by projector_block_masses rather than testing only the
        # special case where activation eigenvectors are coordinate axes.
        a_coordinates = ambient_basis.T @ a
        b_coordinates = ambient_basis.T @ b
        ar = qa.transpose(-2, -1) @ a_coordinates
        br = qb.transpose(-2, -1) @ b_coordinates
        cross = ar.transpose(-2, -1) @ br
        monte_carlo = float(cross.square().sum(dim=(-2, -1)).mean() / k)

        # The statistic lies in [0,1]; this tolerance is conservative relative
        # to the 12k-draw standard error.  The preceding test separately guards
        # the QR sign correction itself.
        self.assertAlmostEqual(monte_carlo, expected, delta=0.012)


if __name__ == "__main__":
    unittest.main()
