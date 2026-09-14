#!/usr/bin/env python3
"""ACT: does cross-specialist read-subspace overlap exceed the base activation opportunity?

THE QUESTION. Independently fine-tuned specialists from one base have task vectors whose
right (read) singular subspaces overlap. The merging literature measures this overlap and
builds on it -- TSV Fig. 3 reports V_i^T V_j across 8 tasks, Demystifying Mergeability
(2601.22285 Sec. 3.3) defines "Right Subspace Overlap" over 190 pairs and uses it to PREDICT
merging success, NUFILT's projector is built from it. None of them calibrates it. Since
dW = sum_t g_t h_t^T puts every row space inside the span of the base's input activations
(WUDI Prop. 1, ICML 2025; NUFILT Thm. 1), specialists from a common parent share that span
by construction, and an isotropic k/d null cannot tell shared capability from shared
opportunity.

WHAT THIS COMPUTES, per (layer, submodule):
  raw        mean squared canonical correlation between top-k right subspaces of dW_i, dW_j
  cond       the same after truncated activation whitening (see below)
  null_iso   k/d, the isotropic expectation the field currently uses
  null_act   ACTIVATION-MATCHED: dW replaced by R C_H^{1/2} with matched row norms, so the
             null task vector has the base's activation geometry and no task coupling
  excess     cond - null_act, the quantity the paper is about

⛔ THE `raw` COLUMN IS THE RESULT; `cond` IS THE ROBUSTNESS VARIANT. Measured on the real
CLIP ViT-B/32 activation covariance over 256 images from four of the eight task datasets, on
task vector pairs drawn to live ENTIRELY inside the activation span -- i.e. sharing no
capability at all -- the two nulls behave like this, averaged over all 72 modules:

    excess_raw       (activation-matched null)   +0.00090
    excess_over_iso  (the isotropic k/d null)    +0.58963

The calibrated null returns zero where the field's null reports a large effect, and it does so
across participation ratios from 3.7 to 16.4, i.e. WITHOUT depending on the truncation at all.
That is why `raw` leads: it needs no m, so it cannot inherit the whitening pathology below.

⛔ WHITENING IS TRUNCATED, NOT TIKHONOV, AND THIS IS NOT A STYLE CHOICE. Whitening two task
vectors by a shared C_H manufactures overlap: on independent task vectors whose true overlap
is chance 0.0400, Tikhonov at lambda = 1e-4 * mean(w) returns 0.7199, and RegMean's own
published alpha = 0.9 off-diagonal shrinkage returns 0.0931. C_H^{-1/2} amplifies small
eigendirections and pushes both task vectors onto the same few of them. Whitening only the
top-m eigendirections and leaving the complement alone holds type-I at chance (0.0394 at
m=8, 0.0405 at m=32, 0.0422 at m=64) while preserving power. See
analysis/act_whitening_calibration.py; --m must be picked there, per submodule, and reported.

⛔ DO NOT SUBSTITUTE A DATA-FREE COVARIANCE. ACTMat uses C = dW^T dW and ACE-Merging a
mean-centred version. With dW = U S V^T, dW(dW^T dW + lambda I)^{-1/2} = U S(S^2+lambda I)^{-1/2} V^T,
whose right singular vectors are exactly V. Whitening by the task vector's own Gram is an
exact no-op on the subspace being measured. C_H must come from
analysis/extract_activation_cov.py, on the frozen base, independent of every task vector.

Usage:
  python analysis/act_test.py --base <base_ckpt> \\
      --experts Code=<ckpt> Math=<ckpt> IF=<ckpt> Mem=<ckpt> Tool=<ckpt> \\
      --cov results/act_cov/qwen25_7b --k 8 --m 32 --out results/act/qwen25_7b.csv
"""
from __future__ import annotations

import argparse, csv, json, itertools, math
from pathlib import Path

import torch


def top_right(dW: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k right singular vectors as columns, (d_in, k). dW is (d_out, d_in).

    ⛔ REFUSES A DEGENERATE INPUT. The SVD of a zero matrix returns an arbitrary orthonormal
    basis, and it returns the SAME arbitrary basis every time, so two zero task vectors score
    overlap = 1.0000 -- the maximum possible value, reported as perfect sharing. That is not
    hypothetical: in a LoRA checkpoint every module the adapter does not target has dW
    identically zero, so an unguarded run over a LoRA pool would fill its averages with 1.0s
    from exactly the modules that carry no task vector at all."""
    n = float(dW.norm())
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError("top_right on a zero or non-finite matrix: its singular vectors are "
                         "arbitrary and two such matrices score overlap 1.0. The caller must "
                         "exclude degenerate task vectors, not silently score them.")
    return torch.linalg.svd(dW, full_matrices=False)[2][:k].T


def overlap(Va: torch.Tensor, Vb: torch.Tensor) -> float:
    """Mean squared canonical correlation. Equals k/d in expectation for independent
    isotropic k-subspaces, which is the isotropic null the field uses."""
    return float((torch.linalg.svdvals(Va.T @ Vb) ** 2).mean())


def truncated_whiten(dW: torch.Tensor, w: torch.Tensor, V: torch.Tensor, m: int):
    """Whiten the top-m activation eigendirections; leave the complement at unit scale.

    dW -> dW (I - V_m^T V_m + V_m^T D V_m) with D = diag((w_m / mean(w_m))^{-1/2}).
    Written as a low-rank correction so it never materialises a d x d matrix."""
    Vm = V[:m]                                            # (m, d)
    blk = torch.clamp(w[:m], min=1e-12)
    scale = torch.rsqrt(blk / blk.mean())                 # (m,)
    P = dW @ Vm.T                                         # (d_out, m)
    return dW + P * (scale - 1.0) @ Vm


def act_matched_null(dW: torch.Tensor, w: torch.Tensor, V: torch.Tensor,
                     trace_full: float, gen):
    """GENERATIVE null: a task vector synthesised to have the base's activation geometry.

    ⛔ RETAINED FOR COMPARISON, NOT AS THE PRIMARY NULL. It assumes the update is one step
    with an isotropic output side, and real task vectors do not look like that. Measured on
    CLIP ViT-B/32 over 576 (module, expert) pairs: a real task vector's top-8 read subspace
    puts a MEDIAN of 3.94x chance mass on the top-m activation eigendirections, whereas this
    null puts 0.986 to 0.999 of it there -- essentially all. On layers.0.mlp.fc1 the top four
    activation directions carry 63.7% of the activation energy and only 5.2% of the real read
    mass, while 74.9% of the real read mass sits outside the top 64. Comparing the two gave
    excess_raw = -0.812, i.e. the null overlapping ninefold more than the data.

    Adam is the likely reason: per-parameter normalisation divides out much of the activation
    anisotropy, so the update is not proportional to C_H even though its row space still lies
    in the activation span."""
    d_out, d_in = dW.shape
    m = V.shape[0]
    Vd = V.to(dW)
    G = torch.randn(d_out, m, generator=gen, device=dW.device, dtype=dW.dtype)
    N = (G * torch.sqrt(torch.clamp(w[:m], min=0)).to(dW)) @ Vd
    tail = max(float(trace_full) - float(w.sum()), 0.0) / max(d_in - m, 1)
    if tail > 0:
        E = torch.randn(d_out, d_in, generator=gen, device=dW.device, dtype=dW.dtype)
        E = E - (E @ Vd.T) @ Vd
        N = N + E * (tail ** 0.5)
    return N * (dW.norm() / torch.clamp(N.norm(), min=1e-12))


def h0_pair(spectrum: torch.Tensor, w: torch.Tensor, V: torch.Tensor, trace_full: float,
            d_out: int, gen):
    """A matched TRUE null: read geometry from C_H, spectrum from the real task vector, and
    nothing else shared.

    ⛔ WHY THIS EXISTS. A null model has to be checked against a case where the answer is
    known. Two task vectors generated from the same activation geometry with independent
    randomness share the span and share nothing else, so the truth is explained = 100% and
    excess = 0. The shipped coupling null does not return that. Measured over spectra with
    participation ratio 12.3 and 65.7:

        n_blocks    explained (truth 100%)   false excess (truth 0)
               4              65.1% / 81.8%        +0.245 / +0.032
               8              95.8% / 91.2%        +0.029 / +0.015
              16              99.3% / 87.7%        +0.005 / +0.022
              32              97.4% / 97.8%        +0.018 / +0.004

    At the shipped eight blocks the false excess is a quarter to a half of the effect being
    reported, so the block count cannot be a global constant and the excess cannot be read
    without subtracting this. Fewer blocks rotate harder, push the null below where it belongs
    and inflate the false excess; the bias is monotone enough to calibrate against.

    The spectrum is matched to the real task vector because the false excess depends on it:
    an H0 built with an arbitrary spectrum calibrates the wrong quantity."""
    d_in = V.shape[1]
    m = V.shape[0]
    G = torch.randn(d_out, m, generator=gen, device=V.device, dtype=torch.float32)
    M = (G * torch.sqrt(torch.clamp(w[:m], min=0))) @ V
    tail = max(float(trace_full) - float(w.sum()), 0.0) / max(d_in - m, 1)
    if tail > 0:
        E = torch.randn(d_out, d_in, generator=gen, device=V.device, dtype=torch.float32)
        M = M + (E - (E @ V.T) @ V) * (tail ** 0.5)
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(spectrum.shape[0], U.shape[1])
    return (U[:, :r] * spectrum[:r].to(U)) @ Vh[:r]


def resolvable_blocks(w: torch.Tensor, se: torch.Tensor, z: float = 2.0):
    """Group eigendirections the calibration data cannot tell apart.

    ⛔ WHY THIS REPLACES TWO FREE PARAMETERS. The null rotates within blocks of the activation
    eigenbasis, and a block count has no principled setting: sweeping it moved the explained
    fraction from 48% to 23%, and the complement's rotation rank moved the excess from +0.032 to
    +0.046 -- a factor of two on knobs a reader would rightly ask about. The obstruction is
    structural. The orthogonal maps that commute with C_H exactly are block-diagonal on its
    eigenvalue MULTIPLICITIES, and for a generic C_H every eigenvalue is distinct, so that group
    is trivial and no exactly invariant randomisation exists.

    What does exist is a statistical version of the same idea. Eigenvalues are estimated from a
    finite calibration set, so two directions whose estimates are not separated by more than
    their uncertainty are directions we cannot order. Rotating within such a group changes
    nothing we can claim to know; rotating across a resolved gap destroys structure we can.
    Blocks are therefore runs of consecutive eigendirections whose z-sigma intervals overlap.

    On CLIP ViT-B/16 with an eight-fold jackknife over 224 calibration images this yields, per
    module, a median of 17 blocks over 64 directions: the leading eigenvalues are resolved and
    sit in blocks of one, and the tail is unresolvable and merges into a single block of about
    40. The tail-rank parameter disappears with it, since the complement is by construction one
    unresolvable block."""
    keep, cur = [], [0]
    for i in range(1, len(w)):
        if float(w[i - 1] - z * se[i - 1]) <= float(w[i] + z * se[i]):
            cur.append(i)
        else:
            keep.append(cur); cur = [i]
    keep.append(cur)
    return keep


def coupling_destroying_null(dW: torch.Tensor, V: torch.Tensor, n_blocks: int, gen,
                             tail_rank: int = 128, blocks=None):
    """THE PRIMARY NULL. Randomise a real task vector WITHIN the activation eigenbasis.

    The generative null above fails because it must invent the update's relationship to C_H,
    and it invents the wrong one. This one never invents it: it takes the observed task
    vector, expresses it in the activation eigenbasis, and applies a random orthogonal map
    that is BLOCK-DIAGONAL in that basis. What survives is exactly what we are not testing --
    this task vector's own spectrum, and its own profile of read mass across activation
    directions. What is destroyed is the only thing we are testing: whether two DIFFERENT
    specialists pick the SAME directions inside that profile.

    Blocks are contiguous runs of the stored eigendirections, so a rotation never moves mass
    between directions of very different activation energy.

    ⛔ THE COMPLEMENT IS ROTATED INSIDE ITSELF, NOT PERMUTED. The first version permuted the
    complement's STANDARD-BASIS coordinates, R[:, perm]. A coordinate permutation is
    orthogonal, but it is not equivariant with the activation eigenbasis, so it does not map
    the complement of span(V) to itself: measured on a random task vector, R's component
    inside span(V) went from 6.2e-06 (zero, by construction) to 17.93 after permuting, a leak
    of 72.5% of R's norm, and the read-mass profile the null is supposed to preserve moved
    0.5055 -> 0.8276. Instead a random orthonormal frame is drawn INSIDE the complement and R
    is rotated within it, which stays in the complement exactly and costs O(d * j) rather
    than the (d-m) x (d-m) QR that is unaffordable at d_in = 18944.

    This is a CALIBRATION-CONDITIONED null, not a training-trajectory one. No gradients from
    training are used and none are claimed."""
    d_out, d_in = dW.shape
    m = V.shape[0]
    Vd = V.to(dW)

    C = dW @ Vd.T                                   # (d_out, m) coordinates in the eigenbasis
    R = dW - C @ Vd                                 # the complement part, untouched basis

    if blocks is None:
        edges = torch.linspace(0, m, n_blocks + 1).round().long().tolist()
        blocks = [list(range(a, b)) for a, b in zip(edges[:-1], edges[1:])]
    for blk in blocks:
        n = len(blk)
        idx = torch.tensor(blk, device=dW.device)
        if n < 2:
            # ⛔ Skipping singletons left the directions the calibration set resolves BEST
            # untouched and bit-identical across two task vectors. On ViT-B/16 the singleton
            # blocks carry 71% of the stored eigenvalue mass, so most of span(V) was exempt.
            # A one-dimensional eigenspace has exactly {+1,-1} as its orthogonal group and Haar
            # measure on it is a fair coin, so the sign flip is the correct randomisation here,
            # not a token one. It leaves C C^T and therefore the spectrum of dW untouched while
            # flipping the sign of every cross term between this direction and the others.
            sgn = 1.0 - 2.0 * float(torch.randint(0, 2, (1,), generator=gen, device=dW.device))
            C[:, idx] = C[:, idx] * sgn
            continue
        Q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen,
                                           device=dW.device, dtype=torch.float32))
        C[:, idx] = C[:, idx] @ Q.to(dW)

    # Haar-rotate R inside the WHOLE complement of span(V).
    #
    # ⛔ An earlier version rotated R inside a random tail_rank-dimensional frame of the
    # complement and left the rest of it bit-identical between two task vectors. The complement
    # holds about three quarters of the real read mass, so that null preserved most of the
    # cross-task agreement it was supposed to destroy. On planted pairs that share 2 of 8 top
    # read directions outright, where the truth is that nothing is inherited, it reported 41%
    # (d_in=768) to 70% (d_in=3072) of that planted structure as inherited. The H0 check cannot
    # catch this: H0 measures over-destruction, and an identity null scores a perfect zero on it.
    #
    # Rotating the full complement does not need a (d_in-m)-square Haar matrix. R's rows lie in
    # the complement and its rank is at most min(d_out, d_in-m), so a Haar map on the complement
    # carries R's right singular vectors to a UNIFORM frame of that rank there. Drawing the frame
    # directly is exact and costs one thin SVD plus one QR.
    #
    # dW dW^T = C C^T + R R^T because R Vd^T = 0, and both terms are invariant under this, so the
    # singular values of dW are preserved exactly rather than approximately.
    rank = int(min(R.shape[0], max(d_in - m, 0)))
    if rank >= 2 and float(R.norm()) > 0:
        # ⛔ gesdd fails to converge on some of these residuals ("too many repeated singular
        # values"), so float32 SVD alone is not safe here. float64 converges where float32 does
        # not, and a QR of R^T is the fallback: its leading columns span row(R) just as well, and
        # only R R^T has to be preserved, not the factorisation itself.
        # Only R R^T has to be reproduced, not any particular factorisation, so the fallback is
        # a symmetric eigendecomposition of R R^T. eigh converges where gesdd does not, and
        # taking its top `rank` eigenpairs is exact because rank(R) <= rank by construction.
        # A QR of R^T is NOT a valid fallback: R is rank-deficient here, an unpivoted QR does not
        # put the null rows last, and truncating its triangular factor drops part of R.
        try:
            Ur, Sr, _ = torch.linalg.svd(R.double(), full_matrices=False)
            M = (Ur[:, :rank] * Sr[:rank])
        except torch._C._LinAlgError:
            Rd = R.double()
            ev, U = torch.linalg.eigh(Rd @ Rd.T)
            ev, U = ev.flip(0)[:rank].clamp(min=0.0), U.flip(1)[:, :rank]
            M = U * ev.sqrt()
        G = torch.randn(d_in, rank, generator=gen, device=dW.device, dtype=torch.float32).double()
        G = G - Vd.T.double() @ (Vd.double() @ G)        # into the complement
        G, _ = torch.linalg.qr(G)                        # uniform frame of the complement
        R = (M @ G.T).to(dW)
    return C @ Vd + R


class LazyCheckpoint:
    """Reads ONE tensor at a time out of a checkpoint, by suffix rather than exact name.

    ⛔ WHY LAZY. The first version of this file called
    AutoModelForCausalLM.from_pretrained(..., torch_dtype=float32).state_dict() once per
    checkpoint and held them all. At 7.6B parameters that is 28.3 GiB per model, and a base
    plus five specialists is 170 GiB resident. It would OOM on any node this project has.
    Reading per module keeps at most six weight matrices alive: 271 MB each at the widest
    (18944 x 3584 fp32), so 1.6 GiB total.

    ⛔ WHY BY SUFFIX. Module paths and checkpoint keys do not agree, and the disagreement is
    silent. transformers' CLIPVisionModel.named_modules() yields
    `encoder.layers.0.mlp.fc1`, while the checkpoints on the hub store
    `vision_model.encoder.layers.0.mlp.fc1.weight`. An exact-match lookup finds nothing for
    all 72 modules, and the only reason that surfaces at all is the empty-result guard at the
    end of main(). Suffix matching with a uniqueness check handles any prefix difference and
    refuses when a suffix is ambiguous."""

    # Loading a whole checkpoint into RAM is fine for a ViT and fatal for a 7B. The bound is
    # generous enough for any vision pool here and far below one fp32 language model.
    BIN_LIMIT_GIB = 4.0

    def __init__(self, path: str, prefix: str = ""):
        p = Path(path)
        self.prefix = prefix
        self.dense = None
        if not p.is_dir():
            p = self._snapshot(path)
        idx = p / "model.safetensors.index.json"
        single = p / "model.safetensors"
        if idx.exists():
            from safetensors import safe_open
            self._open = safe_open
            wm = json.loads(idx.read_text())["weight_map"]
            self.map = {k: str(p / v) for k, v in wm.items()}
        elif single.exists():
            from safetensors import safe_open
            self._open = safe_open
            with safe_open(str(single), framework="pt") as f:
                self.map = {k: str(single) for k in f.keys()}
        else:
            bins = sorted(p.glob("*.bin"))
            if not bins:
                raise SystemExit(f"[refuse] {path} has no safetensors and no .bin")
            gib = sum(q.stat().st_size for q in bins) / 2 ** 30
            if gib > self.BIN_LIMIT_GIB:
                raise SystemExit(
                    f"[refuse] {path} is {gib:.1f} GiB of .bin, over the {self.BIN_LIMIT_GIB} "
                    f"GiB limit for dense loading. Convert it to safetensors so it can be read "
                    f"per tensor; loading it whole is what the lazy path exists to avoid.")
            self.dense = {}
            for q in bins:
                self.dense.update(torch.load(q, map_location="cpu", weights_only=True))
            self.map = {k: None for k in self.dense}

        # suffix -> full key. A suffix that matches more than one key is AMBIGUOUS and is
        # removed rather than resolved to an arbitrary winner. ⛔ This is not pedantry: the
        # full CLIP checkpoint stores both text_model.encoder.layers.0.mlp.fc1.weight and
        # vision_model.encoder.layers.0.mlp.fc1.weight, so a "pick the first" rule would
        # silently build task vectors from the TEXT encoder and compare them against a VISION
        # activation covariance. Pass --key_prefix vision_model. to disambiguate.
        keys = [k for k in self.map if k.startswith(prefix)]
        if prefix and not keys:
            raise SystemExit(f"[refuse] no key in {path} starts with {prefix!r}. "
                             f"Sample keys: {sorted(self.map)[:3]}")
        self.by_suffix: dict[str, str] = {}
        self.ambiguous: set[str] = set()
        for k in keys:
            parts = k.split(".")
            for a in range(len(parts)):
                sfx = ".".join(parts[a:])
                if sfx in self.by_suffix and self.by_suffix[sfx] != k:
                    self.ambiguous.add(sfx)
                self.by_suffix.setdefault(sfx, k)
        for sfx in self.ambiguous:
            self.by_suffix.pop(sfx, None)

    @staticmethod
    def _snapshot(repo: str) -> Path:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo, allow_patterns=["*.safetensors*", "*.bin", "*.json"]))

    def resolve(self, key: str):
        if key in self.map and (not self.prefix or key.startswith(self.prefix)):
            return key
        return self.by_suffix.get(key)

    def why_missing(self, key: str) -> str:
        if key in self.ambiguous:
            return (f"{key!r} is ambiguous; it matches several namespaces. Pass --key_prefix "
                    f"(for CLIP, 'vision_model.') to pin one.")
        return f"{key!r} not found. Sample keys: {sorted(self.map)[:3]}"

    def __contains__(self, key):
        return self.resolve(key) is not None

    def get(self, key, device):
        k = self.resolve(key)
        if k is None:
            raise KeyError(key)
        if self.dense is not None:
            return self.dense[k].to(device=device, dtype=torch.float32)
        with self._open(self.map[k], framework="pt") as f:
            return f.get_tensor(k).to(device=device, dtype=torch.float32)


class AdapterCheckpoint:
    """A PEFT LoRA adapter, which stores the task vector directly rather than a full model.

    dW = (alpha / r) * B @ A for each targeted module, and EXACTLY ZERO everywhere else. That
    zero is the whole reason act_test refuses degenerate task vectors: the tanganke LoRA pools
    target only q_proj and v_proj, so 48 of a CLIP ViT-B/16's 72 linear modules are untouched,
    and scoring them would report overlap 1.0000 from modules that never changed.

    ⛔ COMPARE ONLY ON THE INTERSECTION. A full-fine-tuning arm covers all 72 modules and a
    LoRA arm covers 24. Averaging each over its own module set and calling the difference a
    regime effect measures module composition, not regime."""

    def __init__(self, path: str):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        cfg = json.loads(Path(hf_hub_download(path, "adapter_config.json")).read_text())
        self.rank = int(cfg.get("r", 1))
        self.scale = float(cfg.get("lora_alpha", 1)) / self.rank
        f = None
        for cand in ("adapter_model.safetensors", "linearized_adapter_model.safetensors"):
            try:
                f = hf_hub_download(path, cand); self.kind = cand; break
            except Exception:
                continue
        if f is None:
            raise SystemExit(f"[refuse] {path} has no adapter_model.safetensors")
        with safe_open(f, framework="pt") as h:
            self.t = {k: h.get_tensor(k) for k in h.keys()}
        # module path -> (A, B), keyed by the suffix after the PEFT wrapper prefix
        self.mods: dict[str, dict] = {}
        for k, v in self.t.items():
            for tag in ("lora_A", "lora_B"):
                if f".{tag}." in k:
                    mod = k.split(f".{tag}.")[0]
                    mod = mod.split("base_model.model.", 1)[-1]
                    # ⛔ The linearized adapters wrap one level deeper: their keys read
                    # ...q_proj.model.lora_A.default.weight, so a naive split leaves a
                    # trailing ".model" that matches no module. Every target then resolves to
                    # None, every task vector is zero, and the run reports "dropped 192 cells"
                    # and dies rather than producing a silently wrong number -- which is only
                    # true because the zero guard exists.
                    if mod.endswith(".model"):
                        mod = mod[: -len(".model")]
                    self.mods.setdefault(mod, {})[tag] = v
        self.targets = {m for m, d in self.mods.items() if "lora_A" in d and "lora_B" in d}

    def __contains__(self, key):
        """Every module of the base is 'present': an untargeted one has a zero task vector,
        which act_test drops explicitly rather than skipping here, so that the drop is counted
        and reported instead of silently shrinking the module set."""
        return True

    def randomized(self, mod: str, device, gen):
        d = self.mods[mod]
        return lora_shared_init_null(d["lora_A"].to(device, torch.float32),
                                     d["lora_B"].to(device, torch.float32), self.scale, gen)

    def delta(self, mod: str, device):
        d = self.mods.get(mod)
        if not d or "lora_A" not in d or "lora_B" not in d:
            return None                      # untouched: caller treats as a zero task vector
        A = d["lora_A"].to(device=device, dtype=torch.float32)
        B = d["lora_B"].to(device=device, dtype=torch.float32)
        return (B @ A) * self.scale


def lora_shared_init_null(A: torch.Tensor, B: torch.Tensor, scale: float, gen):
    """THE NULL FOR A LoRA POOL: keep this adapter's A, draw its B afresh.

    For dW = (alpha/r) B A the row space lies inside row(A), and equals it whenever B has full
    column rank, so adapters trained from a shared initialisation inherit that space rather than
    learning it. The null keeps A exactly as the adapter has it and redraws B i.i.d. at the
    observed Frobenius norm: an adapter that shares everything inherited and nothing learned.

    ⛔ THREE EARLIER CONSTRUCTIONS FAILED, each caught by asking what the null returns where the
    answer is known -- two adapters with the same A and an independent B, so the truth is zero
    excess:

      B -> B Q for a random orthogonal Q does not preserve dW's spectrum; its singular values
      moved by 19.7%.

      dW = U S V^T -> U S (V Q)^T with uniform Q inside row(A) preserves span and spectrum but
      destroys more than the cross-task agreement, because row(A) is not isotropic: A carries its
      own spectrum and both adapters inherit that too. Two true-null draws overlap at 0.706, not
      the k/r = 0.5 a uniform rotation assumes, and the false excess was +0.2023.

      The same rotation restricted to contiguous blocks is inert at these shapes. With r = 16 and
      k = 8 the top-k subspace is exactly the first block, so rotating inside it leaves the span
      unchanged and both the false excess and the measured excess collapse to 0.0000.

    Redrawing B avoids all three and needs no rotation, no block count and no calibration.

    ⛔ MATCHING ONLY THE FROBENIUS NORM IS NOT ENOUGH, and this was a fourth failure. An i.i.d.
    Gaussian G has a Marchenko-Pastur spectrum, which is nearly flat, while a trained B decays.
    That violates requirement (i) of the null -- same spectrum -- which is the requirement this
    same docstring invokes to reject B -> BQ. It is not a cosmetic violation: on pools that share
    A and have genuinely independent B, so the true excess is zero, the flat-spectrum null reports
    an excess that runs from +0.09 to -0.39 as B's decay is varied, i.e. an unmatched nuisance
    parameter sets even the SIGN of the answer.

    Redrawing B at its OBSERVED singular values fixes this: U diag(S_B) W^T with U and W drawn
    Haar keeps the spectrum exactly and randomises only the orientation, which is the thing the
    adapters could have learned."""
    r = min(B.shape)
    Sb = torch.linalg.svdvals(B.float())[:r]
    U = torch.linalg.qr(torch.randn(B.shape[0], r, generator=gen,
                                    device=B.device, dtype=torch.float32))[0]
    W = torch.linalg.qr(torch.randn(B.shape[1], r, generator=gen,
                                    device=B.device, dtype=torch.float32))[0]
    G = (U * Sb) @ W.T
    return (G @ A) * scale

def load_delta(base: LazyCheckpoint, expert, key, device):
    if isinstance(expert, AdapterCheckpoint):
        dW = expert.delta(key[:-len(".weight")] if key.endswith(".weight") else key, device)
        if dW is None:
            return torch.zeros_like(base.get(key, device))
        return dW
    return expert.get(key, device) - base.get(key, device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--experts", nargs="+", required=True, help="NAME=path ...")
    ap.add_argument("--cov", required=True, help="dir written by extract_activation_cov.py")
    ap.add_argument("--experts_are_adapters", action="store_true",
                    help="experts are PEFT LoRA adapter repos storing dW = (alpha/r) B A "
                         "rather than full checkpoints. Untargeted modules are exactly zero "
                         "and get dropped, so the comparison arm must be restricted to the "
                         "same module set or the contrast measures composition, not regime.")
    ap.add_argument("--only_modules", default="",
                    help="comma-separated submodule names to score, e.g. q_proj,v_proj. Use it "
                         "to match a LoRA arm's target set in the full-fine-tuning arm.")
    ap.add_argument("--key_prefix", default="",
                    help="restrict checkpoint keys to this namespace before matching module "
                         "paths by suffix. Required whenever one checkpoint holds more than "
                         "one tower: the full CLIP model stores text_model.* and "
                         "vision_model.* under identical suffixes, so without it every lookup "
                         "is ambiguous and refused. For a vision pool pass 'vision_model.'.")
    ap.add_argument("--k", type=int, default=8, help="read subspace budget")
    ap.add_argument("--m", default="auto",
                    help="whitened eigendirections for the `cond` column. An integer, or "
                         "'auto' to take each module's own participation ratio, floored. "
                         "⛔ A single global m is not safe: on CLIP ViT-B/32 over 256 real "
                         "images the participation ratio runs 2.5 to 16.4 across the 72 "
                         "modules, so m=32 would whiten directions the data never determined. "
                         "This flag does not affect the `raw` column, which needs no m at all.")
    ap.add_argument("--null_draws", type=int, default=20)
    ap.add_argument("--min_rel_norm", type=float, default=1e-8,
                    help="drop a (module, expert) whose ||dW||/||W_0|| falls below this. Zero "
                         "task vectors have arbitrary singular vectors and two of them score "
                         "overlap 1.0, so scoring them would report perfect sharing from "
                         "modules that changed not at all. Common in LoRA pools.")
    ap.add_argument("--null", default="coupling", choices=["coupling", "generative", "lora"],
                    help="'coupling' randomises the observed task vectors inside the "
                         "activation eigenbasis and is the primary null; 'generative' "
                         "synthesises one from C_H and is kept only for comparison, since it "
                         "assumes a one-step isotropic-output update the data does not show. "
                         "'lora' requires --experts_are_adapters and preserves A plus B's "
                         "spectrum while randomising the selection inside row(A); use it "
                         "whenever the adapters may share an initialisation.")
    ap.add_argument("--fold_cov", default="",
                    help="directory of per-fold covariance runs (f0..fK-1) from the same "
                         "extractor on disjoint splits of the calibration set. With it the "
                         "rotation blocks are the groups of eigendirections the data cannot "
                         "resolve, and --null_blocks and --tail_rank are ignored. Without it "
                         "the null falls back to a fixed block count, which has no principled "
                         "setting: sweeping tail_rank from 32 to 512 moved the explained "
                         "fraction from 48%% to 23%%.")
    ap.add_argument("--null_m", type=int, default=0,
                    help="truncate the stored eigenbasis to this many directions BEFORE the null "
                         "sees it. The null protects span(V) with blocks and destroys the "
                         "complement outright, so this sets that boundary. 0 keeps all stored "
                         "directions, which is what every reported number uses. Exposed so the "
                         "boundary can be swept rather than asserted to be irrelevant.")
    ap.add_argument("--block_z", type=float, default=2.0,
                    help="how many standard errors two eigenvalues must be apart before the "
                         "null is allowed to rotate across them")
    ap.add_argument("--tail_rank", type=int, default=128,
                    help="dimension of the random frame the coupling null rotates the "
                         "complement within. Larger scrambles more of the tail; the cost is "
                         "O(d_in * tail_rank).")
    ap.add_argument("--h0_draws", type=int, default=8,
                    help="matched true-null pairs PER MODULE used to measure how much excess "
                         "the null reports where the answer is known to be zero. The measured "
                         "excess minus this is the corrected quantity; 0 disables. "
                         "⚠️ The precision that matters is on the AGGREGATE, whose sample size "
                         "is draws x modules, so a run over many modules needs few draws each. "
                         "40 draws over 12 modules gave a 95%% interval of +/-0.0020 on the "
                         "bias; 8 over 72 modules is a larger sample for less than a fifth of "
                         "the cost. Going the other way is what burns: 10 draws over 4 modules "
                         "put the bias at a value that later proved to be noise, and the "
                         "explained fraction read 92.7%% instead of 72.2%%.")
    ap.add_argument("--null_blocks", type=int, default=8,
                    help="contiguous eigendirection blocks the coupling null rotates within. "
                         "Fewer blocks destroy more structure; 1 block mixes the strongest and "
                         "weakest stored directions and is too aggressive.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cov_dir = Path(args.cov)
    manifest = json.loads((cov_dir / "manifest.json").read_text())
    if args.null == "lora" and not args.experts_are_adapters:
        raise SystemExit("[refuse] --null lora needs --experts_are_adapters: it works from the "
                         "adapter's A and B, not from a materialised weight difference.")
    auto_m = (str(args.m).lower() == "auto")
    if not auto_m:
        args.m = int(args.m)
        if args.m > manifest["m_out"]:
            raise SystemExit(f"[refuse] --m {args.m} exceeds the {manifest['m_out']} directions "
                             f"stored in {cov_dir}. Re-extract with a larger --m_out.")
        thin = [n for n, v in manifest["modules"].items()
                if v.get("participation_ratio", float("inf")) < args.m]
        if thin:
            print(f"[act] ⚠️  {len(thin)}/{len(manifest['modules'])} modules have a "
                  f"participation ratio below --m {args.m}. Those whitened directions are "
                  f"arbitrary, not estimated. Consider --m auto.", flush=True)

    print(f"[act] indexing base {args.base}", flush=True)
    base = LazyCheckpoint(args.base, args.key_prefix)
    experts = {}
    for spec in args.experts:
        name, path = spec.split("=", 1)
        print(f"[act] indexing expert {name}", flush=True)
        experts[name] = (AdapterCheckpoint(path) if args.experts_are_adapters
                         else LazyCheckpoint(path, args.key_prefix))
    if args.experts_are_adapters:
        tg = sorted(set.intersection(*(e.targets for e in experts.values())))
        print(f"[act] adapters target {len(tg)} modules in common; the rest have a zero task "
              f"vector and will be dropped. Restrict the comparison arm to these.", flush=True)

    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    rows = []
    degenerate: dict = {}
    skipped_modules: list = []
    modules = sorted(manifest["modules"])
    if args.only_modules:
        keep = tuple(x.strip() for x in args.only_modules.split(",") if x.strip())
        modules = [m for m in modules if m.split(".")[-1] in keep]
        print(f"[act] --only_modules {keep}: {len(modules)} modules retained", flush=True)
    for mi, mod in enumerate(modules):
        key = mod + ".weight"
        if key not in base or any(key not in e for e in experts.values()):
            continue
        blob = torch.load(cov_dir / f"{mod.replace('.', '__')}.pt", map_location=args.device,
                           weights_only=False)
        w, V = blob["eigvals"].to(args.device), blob["eigvecs"].to(args.device)
        if args.null_m:
            w, V = w[:args.null_m], V[:args.null_m]
        d_in = blob["d"]
        blocks = None
        if args.fold_cov:
            fw = []
            for fd in sorted(Path(args.fold_cov).glob("f*")):
                q = fd / f"{mod.replace('.', '__')}.pt"
                if q.exists():
                    fw.append(torch.load(q, map_location="cpu",
                                         weights_only=False)["eigvals"].double())
            if len(fw) < 4:
                raise SystemExit(f"[refuse] {mod}: only {len(fw)} folds under {args.fold_cov}. "
                                 f"Eigenvalue standard errors need at least four.")
            L = min(len(x) for x in fw)
            St = torch.stack([x[:L] for x in fw])
            se = (St.std(0) / len(fw) ** 0.5).float().to(args.device)
            L = min(L, V.shape[0])
            blocks = resolvable_blocks(w[:L], se[:L], args.block_z)
        trace_full = blob.get("trace_full")
        if trace_full is None:
            raise SystemExit(
                f"[refuse] {cov_dir} predates trace_full. Without tr(C_H) the null cannot be "
                f"given a tail and is confined to the stored subspace, which inverts the "
                f"result. Re-extract with the current analysis/extract_activation_cov.py.")
        # Per-module truncation. The participation ratio (sum w)^2 / sum w^2 counts the
        # directions the calibration inputs actually determine; beyond it the eigenvectors
        # are arbitrary and two extractions on the same data disagree about them.
        m_mod = (max(1, min(int(manifest["modules"][mod]["participation_ratio"]), V.shape[0]))
                 if auto_m else args.m)

        raws, conds, anchor, deltas = {}, {}, None, {}
        w0 = float(base.get(key, args.device).norm())
        for name, e in experts.items():
            dW = load_delta(base, e, key, args.device)
            if dW.shape[1] != d_in:
                raise SystemExit(f"[refuse] {mod}: dW has d_in {dW.shape[1]} but C_H has {d_in}. "
                                 f"The covariance is for the wrong side of this module.")
            # ⛔ A task vector that is (numerically) zero has arbitrary singular vectors, and two
            # of them score overlap 1.0. LoRA checkpoints leave every untargeted module exactly
            # at the base, so this is the common case there, not an edge case. Drop the module
            # for this expert and count it; never score it.
            rel = float(dW.norm()) / max(w0, 1e-30)
            if not math.isfinite(rel) or rel < args.min_rel_norm:
                degenerate[(mod, name)] = rel
                del dW
                continue
            raws[name] = top_right(dW, args.k)
            conds[name] = top_right(truncated_whiten(dW, w, V, m_mod), args.k)
            if anchor is None:
                anchor = dW.clone()
            if args.null == "coupling":
                deltas[name] = dW
            else:
                del dW
        live = sorted(raws)
        if len(live) < 2:
            skipped_modules.append(mod)
            continue

        # ⛔ THE NULL GOES THROUGH THE SAME TRANSFORM AS WHAT IT IS COMPARED AGAINST.
        # The first version measured the null on RAW subspaces and subtracted it from the
        # WHITENED `cond`, which is apples to oranges: on a synthetic check the same null
        # reads 0.5275 raw and 0.5071 whitened. Both columns are now paired with their own
        # null, and both are reported, because they answer different questions:
        #   raw   vs null_act_raw   -- is the overlap more than base geometry gives for free?
        #   cond  vs null_act_cond  -- the same question after removing the top-m directions.
        # The raw comparison needs no whitening at all and so is immune to the type-I
        # inflation documented in analysis/act_whitening_calibration.py. Lead with it.
        # ⛔ THE NULL IS DRAWN PER PAIR, NOT ONCE PER MODULE. The first version drew ~20
        # nulls from cyclically adjacent experts in alphabetical order and used their mean as
        # a single constant for all 190 pairs. Those source pairs are not a random sample: on
        # ViT-B/32 their raw overlap averages 0.10637 against 0.09531 over all pairs, and the
        # explained fraction moved 32.68% -> 28.97% when the null was restricted to the pairs
        # it was actually built from. Randomising every expert once per draw and then scoring
        # all pairs from those costs 20 randomisations per draw instead of 380, so the correct
        # bookkeeping is also the cheap one.
        nulls_r: dict = {(i, j): [] for i, j in itertools.combinations(live, 2)}
        nulls_c: dict = {k: [] for k in nulls_r}
        for _ in range(args.null_draws):
            rnd = {}
            for nm in live:
                if args.null == "lora":
                    rnd[nm] = experts[nm].randomized(mod, args.device, gen)
                elif args.null == "coupling":
                    rnd[nm] = coupling_destroying_null(deltas[nm], V, args.null_blocks, gen,
                                                       args.tail_rank, blocks)
                else:
                    rnd[nm] = act_matched_null(anchor, w, V, trace_full, gen)
            tr = {nm: top_right(x, args.k) for nm, x in rnd.items()}
            tc = {nm: top_right(truncated_whiten(x, w, V, m_mod), args.k)
                  for nm, x in rnd.items()}
            for i, j in nulls_r:
                nulls_r[(i, j)].append(overlap(tr[i], tr[j]))
                nulls_c[(i, j)].append(overlap(tc[i], tc[j]))
        _anchor_keep = anchor

        # ---- matched H0: how much excess does the null invent when the truth is zero? -----
        h0_ex = 0.0
        if args.null == "lora":
            # The null IS the true null here -- same A, independent B -- so there is no separate
            # H0 term to subtract and no knob to sweep.
            pass
        elif args.h0_draws > 0:
            spec = torch.linalg.svdvals(next(iter(deltas.values())) if deltas
                                        else anchor)
            d_out = anchor.shape[0]
            vals = []
            for _ in range(args.h0_draws):
                A0 = h0_pair(spec, w, V, trace_full, d_out, gen)
                B0 = h0_pair(spec, w, V, trace_full, d_out, gen)
                nA = coupling_destroying_null(A0, V, args.null_blocks, gen, args.tail_rank, blocks)
                nB = coupling_destroying_null(B0, V, args.null_blocks, gen, args.tail_rank, blocks)
                vals.append(overlap(top_right(A0, args.k), top_right(B0, args.k))
                            - overlap(top_right(nA, args.k), top_right(nB, args.k)))
            h0_ex = float(sum(vals) / len(vals))

        for i, j in itertools.combinations(live, 2):
            r_ = overlap(raws[i], raws[j])
            c_ = overlap(conds[i], conds[j])
            null_raw = sum(nulls_r[(i, j)]) / len(nulls_r[(i, j)])
            null_cond = sum(nulls_c[(i, j)]) / len(nulls_c[(i, j)])
            rows.append(dict(module=mod, d_in=d_in, a=i, b=j, k=args.k, m=m_mod,
                             null_kind=args.null,
                             n_blocks=(len(blocks) if blocks else args.null_blocks),
                             raw=r_, cond=c_,
                             null_iso=args.k / d_in,
                             null_act_raw=null_raw, null_act_cond=null_cond,
                             excess_over_iso=r_ - args.k / d_in,
                             excess_raw=r_ - null_raw,
                             excess_cond=c_ - null_cond,
                             h0_excess=h0_ex,
                             excess_raw_corrected=r_ - null_raw - h0_ex))
        if (mi + 1) % 20 == 0:
            print(f"[act] {mi+1}/{len(modules)} modules", flush=True)

    if degenerate:
        byexp: dict = {}
        for (mo, na) in degenerate:
            byexp[na] = byexp.get(na, 0) + 1
        print(f"[act] ⛔ dropped {len(degenerate)} (module, expert) cells whose task vector is "
              f"below --min_rel_norm={args.min_rel_norm:g}. Per expert: "
              f"{dict(sorted(byexp.items(), key=lambda x: -x[1])[:6])}", flush=True)
    if skipped_modules:
        print(f"[act] ⛔ {len(skipped_modules)} modules had fewer than 2 live experts and were "
              f"skipped entirely: {skipped_modules[:4]}", flush=True)
    if not rows:
        probe = sorted(manifest["modules"])[0] + ".weight"
        raise SystemExit(
            "[refuse] no module matched between the covariance directory and the checkpoints.\n"
            f"  base:   {base.why_missing(probe)}\n"
            + "".join(f"  {n}: {e.why_missing(probe)}\n" for n, e in list(experts.items())[:1])
            + "  Check that --cov was extracted from the same base, and see --key_prefix.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader(); wtr.writerows(rows)
    print(f"[act] wrote {len(rows)} rows to {args.out}")

    import statistics as st
    for fld in ("raw", "cond", "null_iso", "null_act_raw", "null_act_cond",
                "excess_over_iso", "excess_raw", "h0_excess", "excess_raw_corrected",
                "excess_cond"):
        print(f"  mean {fld:16s} {st.mean(r[fld] for r in rows):+.5f}")
    print("\n  THE PAPER'S NUMBER is `excess_raw` (and `excess_cond` as the robustness")
    print("  variant). `excess_over_iso` is what the literature currently reports against")
    print("  its isotropic null; printing both side by side is the whole argument.")


if __name__ == "__main__":
    main()
