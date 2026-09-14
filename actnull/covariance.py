#!/usr/bin/env python3
"""Streaming sketch of the FROZEN BASE model's input second moment, per linear module.

WHY THIS FILE IS THE CRITICAL PATH. Every published route to an activation covariance in the
merging literature estimates a TASK-CONDITIONED object and none of them gives what a null
test needs:

    RegMean      (Jin et al., ICLR 2023)   X_i^T X_i on each FINE-TUNED model's own data
    ACTMat       (2604.01329)              C_hat = Delta_i^T Delta_i, from the task vector
    ACE-Merging  (2603.02945)              mean-centred rows of Delta W_i

The last two are worse than merely task-conditioned. With dW = U S V^T,

    dW (dW^T dW + lambda I)^{-1/2} = U S (S^2 + lambda I)^{-1/2} V^T

whose right singular vectors are EXACTLY V for every lambda > 0 (verified to 1e-10 at
lambda >= 1e-3). Whitening a task vector by its own Gram is an exact no-op on the read
subspace, so a null test built on it would run on unwhitened task vectors while appearing
not to. We need C_H from the base model, computed on data, independent of every task vector.

WHY A SKETCH. C_H is d_in x d_in. At d_in = 18944 that is 1.4 GB in fp32 per module and
~280 GB across 196 modules. The test only ever uses the top-m eigendirections (m in the
tens), so we keep a Frequent Directions sketch B of shape (2m, d) with C_H ~= B^T B. At
m = 128 that is 19 MB per module, ~3.8 GB for the whole model.

FREQUENT DIRECTIONS (Liberty, KDD 2013) is used rather than a random projection because its
error bound is on the SPECTRUM we are about to threshold: ||C - B^T B||_2 <= ||H||_F^2 / m,
deterministic, no failure probability. A Gaussian sketch would put a random rotation inside
the very geometry the null test is trying to isolate.

⛔ AND FD ALONE IS NOT ENOUGH, WHICH IS WHY THIS RUNS TWO PASSES. FD's shrinkage subtracts
the m-th squared singular value at every flush, so its eigenVALUES are biased DOWNWARD:
0 <= sigma_i^2(H) - sigma_i^2(B) <= ||H||_F^2 / m. Measured on synthetic anisotropic data
(d=400, N=20000), the sketched subspace at m=128 is excellent (top-8 and top-32 overlap with
the exact eigenspace = 0.998) while the eigenvalues are off by up to 49% of the leading one.
That bias is not survivable here: whitening applies C_H^{-1/2}, so understated eigenvalues
over-amplify their directions, which is precisely the mechanism that inflates the null (see
analysis/act_whitening_calibration.py). Pass 1 therefore uses FD only to FIND the subspace;
pass 2 re-reads the data and accumulates the exact m x m matrix E[(V^T h)(V^T h)^T] inside
it, giving unbiased eigenvalues at negligible cost.

SIZING RULE, from the same experiment: the sketch must be much larger than the number of
directions you intend to keep. top-32 overlap was 0.998 at sketch m=128 but 0.505 at m=64.
Use --sketch >= 4 x --m_out, and read the printed subspace diagnostics before trusting a run.

⛔ THE CALIBRATION SET IS PART OF THE MEASUREMENT. C_H is an expectation over some input
distribution and the answer depends on which. Pass it explicitly, and the manifest records
the file, its sha256, the token count and the sequence length so a reader can tell what
"the base model's activation geometry" meant here. Do not use a task's own training data:
that reintroduces exactly the conditioning this file exists to avoid.

Usage:
  python analysis/extract_activation_cov.py \
      --model /131_data/geeho/minsik/Qwen2.5-7B-Instruct \
      --data benchmarks/controlled_if/mathif.jsonl --text_key prompt \
      --out results/act_cov/qwen25_7b --sketch 128 --max_seqs 512
"""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

import torch
import torch.nn as nn


class FrequentDirections:
    """Deterministic sketch B (ell x d) with C = H^T H approximated by B^T B.

    Rows arrive in blocks. When the buffer fills, one SVD shrinks it back to half capacity by
    subtracting the (ell/2)-th squared singular value from every squared singular value. The
    shrinkage is what bounds the error; simply truncating would bias the top of the spectrum
    upward, which is the part the test thresholds on."""

    def __init__(self, d: int, m: int, device, dtype=torch.float32):
        self.d, self.m, self.ell = d, m, 2 * m
        self.B = torch.zeros(self.ell, d, device=device, dtype=dtype)
        self.next = 0
        self.n_rows = 0
        self.fro2 = 0.0          # running ||H||_F^2, the scale the FD bound is stated in

    @torch.no_grad()
    def add(self, rows: torch.Tensor):
        rows = rows.to(self.B.dtype)
        self.n_rows += rows.shape[0]
        self.fro2 += float(rows.pow(2).sum())
        i = 0
        while i < rows.shape[0]:
            take = min(self.ell - self.next, rows.shape[0] - i)
            self.B[self.next:self.next + take] = rows[i:i + take]
            self.next += take
            i += take
            if self.next == self.ell:
                self._shrink()

    @torch.no_grad()
    def _shrink(self):
        # torch.linalg.svd on (2m x d) costs O(m^2 d); at m=128, d=18944 this is milliseconds.
        _, S, Vh = torch.linalg.svd(self.B, full_matrices=False)
        delta = S[self.m - 1] ** 2
        S2 = torch.clamp(S ** 2 - delta, min=0.0)
        self.B.zero_()
        keep = min(self.m, S2.shape[0])
        self.B[:keep] = torch.sqrt(S2[:keep]).unsqueeze(1) * Vh[:keep]
        self.next = keep

    @torch.no_grad()
    def eig(self, m_out: int):
        """Top-m_out directions, plus their DOWNWARD-BIASED sketch eigenvalues.

        Use the vectors. Do not report the values: pass 2 replaces them. They are returned
        only so the bias can be printed next to the corrected spectrum."""
        B = self.B[:max(self.next, 1)]
        _, S, Vh = torch.linalg.svd(B, full_matrices=False)
        k = min(m_out, S.shape[0])
        return (S[:k] ** 2 / max(self.n_rows, 1)).cpu(), Vh[:k]


class ExactInSubspace:
    """Pass 2. Accumulates M = sum_t (V h_t)(V h_t)^T for a FIXED basis V (m x d).

    m x m per module, so 128x128 floats even at d = 18944. Its eigendecomposition gives
    unbiased eigenvalues, and rotating V by its eigenvectors gives the corrected basis."""

    def __init__(self, V: torch.Tensor):
        self.V = V                                   # (m, d), on device
        self.M = torch.zeros(V.shape[0], V.shape[0], device=V.device, dtype=torch.float32)
        self.n = 0

    @torch.no_grad()
    def add(self, rows: torch.Tensor):
        Z = rows.to(torch.float32) @ self.V.T.to(torch.float32)   # (n, m)
        self.M += Z.T @ Z
        self.n += rows.shape[0]

    @torch.no_grad()
    def eig(self):
        C = self.M / max(self.n, 1)
        w, U = torch.linalg.eigh(C)                  # ascending
        w, U = w.flip(0), U.flip(1)
        return torch.clamp(w, min=0.0).cpu(), (U.T @ self.V).cpu()   # (m,), (m, d)


class ExactFull:
    """The full d x d second moment, no sketch. Only for small d.

    WHY IT EXISTS. At CLIP ViT-B/32's widths (768 and 3072) the exact covariance is 2.4 MB
    and 37.7 MB, so the whole 72-module encoder fits in ~594 MB. That makes the vision pool
    the one place the sketch path can be validated against ground truth rather than against
    another approximation. Run both and compare before trusting the 7B numbers."""

    def __init__(self, d: int, device):
        self.d = d
        self.M = torch.zeros(d, d, device=device, dtype=torch.float32)
        self.n_rows = 0
        self.fro2 = 0.0

    @torch.no_grad()
    def add(self, rows: torch.Tensor):
        r = rows.to(torch.float32)
        self.M += r.T @ r
        self.n_rows += r.shape[0]
        self.fro2 += float(r.pow(2).sum())

    @torch.no_grad()
    def pr_full(self) -> float:
        """Participation ratio from the FULL spectrum, without an eigendecomposition.

        (sum w)^2 / sum w^2 = tr(C)^2 / ||C||_F^2. ⛔ Computed from only the stored top-m_out
        eigenvalues it is bounded above by m_out (Cauchy-Schwarz), so a gate comparing the two
        can never be quiet and the value act_test consumes for --m auto is biased down. On the
        real CLIP encoder the stored value understated it by up to 8.7x."""
        C = self.M / max(self.n_rows, 1)
        return float(torch.diagonal(C).sum() ** 2 / torch.clamp(C.pow(2).sum(), min=1e-30))

    @torch.no_grad()
    def eig(self, m_out: int):
        w, U = torch.linalg.eigh(self.M / max(self.n_rows, 1))
        w, U = w.flip(0), U.flip(1)
        k = min(m_out, w.shape[0])
        return torch.clamp(w[:k], min=0.0).cpu(), U[:, :k].T.cpu()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class CausalLMBackend:
    """Text decoder. Rows are (batch x seq) tokens; padding must be removed."""

    default_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    def __init__(self, args):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
        rows = [json.loads(l) for l in open(args.data) if l.strip()][:args.max_seqs]
        self.texts = [r[args.text_key] for r in rows]
        if args.chat_template:
            kw = ({"enable_thinking": False}
                  if "enable_thinking" in (getattr(self.tok, "chat_template", None) or "") else {})
            self.texts = [self.tok.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False,
                add_generation_prompt=True, **kw) for t in self.texts]

    def __len__(self):
        return len(self.texts)

    def batches(self):
        a = self.args
        for i in range(0, len(self.texts), a.batch):
            enc = self.tok(self.texts[i:i + a.batch], return_tensors="pt", padding=True,
                           truncation=True, max_length=a.max_len,
                           add_special_tokens=not a.chat_template).to(a.device)
            yield enc, enc["attention_mask"].reshape(-1).bool(), min(i + a.batch, len(self.texts))

    def provenance(self):
        return dict(kind="causal_lm", data=self.args.data,
                    data_sha256=sha256(Path(self.args.data)), text_key=self.args.text_key,
                    chat_template=bool(self.args.chat_template), n_items=len(self.texts),
                    max_len=self.args.max_len)


class CLIPVisionBackend:
    """CLIP vision encoder. Every patch token is real, so there is NO padding mask.

    The pool is the standard task-arithmetic eight: tanganke/clip-vit-base-patch32_{cars,dtd,
    eurosat,gtsrb,mnist,resisc45,sun397,svhn} (PoC.md:857), on base openai/clip-vit-base-patch32.
    Twelve layers x six linear modules = 72, widths 768 and 3072, which is why --exact_cov is
    affordable here and nowhere else."""

    default_modules = "q_proj,k_proj,v_proj,out_proj,fc1,fc2"

    def __init__(self, args):
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.args = args
        self.model = CLIPVisionModel.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
        self.proc = CLIPImageProcessor.from_pretrained(args.image_processor or args.model)
        if args.synthetic:
            self.items = list(range(args.synthetic))
            self.synthetic = True
        else:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            found = sorted(q for q in Path(args.images).rglob("*") if q.suffix.lower() in exts)
            if not found:
                raise SystemExit(f"[refuse] no images under {args.images}")
            # ⛔ NEVER TRUNCATE ALPHABETICALLY. sorted(...)[:max_seqs] orders by path string, so
            # with 200 images from each of four datasets and max_seqs=512 the selection is
            # 200+200+112+0 -- C_H becomes the base geometry on two and a half of the four
            # sources, which is exactly the task-conditioning this file exists to avoid, and
            # nothing downstream can tell. Sample round-robin across immediate parents instead.
            if len(found) > args.max_seqs:
                import collections
                by = collections.OrderedDict()
                for q in found:
                    by.setdefault(q.parent.name, []).append(q)
                picked, i = [], 0
                while len(picked) < args.max_seqs and any(by.values()):
                    for kk in list(by):
                        if by[kk] and len(picked) < args.max_seqs:
                            picked.append(by[kk].pop(0))
                    i += 1
                found = sorted(picked)
            self.items = found
            self.synthetic = False

    def __len__(self):
        return len(self.items)

    def batches(self):
        a = self.args
        size = getattr(self.proc, "size", {"shortest_edge": 224})
        px = size.get("shortest_edge", 224) if isinstance(size, dict) else 224
        for i in range(0, len(self.items), a.batch):
            chunk = self.items[i:i + a.batch]
            if self.synthetic:
                g = torch.Generator().manual_seed(1000 + i)
                pv = torch.randn(len(chunk), 3, px, px, generator=g)
            else:
                from PIL import Image
                imgs = [Image.open(q).convert("RGB") for q in chunk]
                pv = self.proc(images=imgs, return_tensors="pt")["pixel_values"]
            pv = pv.to(a.device, dtype=getattr(torch, a.dtype))
            # No mask: every patch token is a real observation.
            yield {"pixel_values": pv}, None, min(i + a.batch, len(self.items))

    def provenance(self):
        if self.synthetic:
            return dict(kind="clip_vision", n_items=len(self.items),
                        images="SYNTHETIC NOISE -- plumbing only, NOT a valid calibration set")
        import collections, hashlib
        h = hashlib.sha256()
        for q in self.items:
            h.update(str(q.name).encode()); h.update(str(q.stat().st_size).encode())
        return dict(kind="clip_vision", images=str(self.args.images),
                    n_items=len(self.items), max_seqs=self.args.max_seqs,
                    per_source=dict(collections.Counter(q.parent.name for q in self.items)),
                    image_list_sha256=h.hexdigest())


BACKENDS = {"causal_lm": CausalLMBackend, "clip_vision": CLIPVisionBackend}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="causal_lm", choices=list(BACKENDS))
    ap.add_argument("--model", required=True, help="the FROZEN BASE checkpoint, never a specialist")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sketch", type=int, default=128, help="m; sketch holds 2m rows")
    ap.add_argument("--exact_cov", action="store_true",
                    help="accumulate the full d x d second moment instead of sketching. Only "
                         "viable for small d (CLIP ViT-B/32 is 594 MB across all 72 modules; "
                         "Qwen2.5-7B would be ~280 GB). Use it to validate the sketch path.")
    ap.add_argument("--m_out", type=int, default=128, help="eigenpairs written per module")
    ap.add_argument("--max_seqs", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--include", default=None,
                    help="comma-separated submodule names; defaults to the backend's set")
    # causal_lm only
    ap.add_argument("--data", help="calibration set, jsonl (causal_lm)")
    ap.add_argument("--text_key", default="prompt")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--chat_template", action="store_true",
                    help="apply the chat template, matching eval/run_prism_gen.py:362. Set this "
                         "whenever the specialists were evaluated through it, or C_H describes a "
                         "different input distribution than the one the task vectors saw.")
    # clip_vision only
    ap.add_argument("--images", help="directory of images (clip_vision)")
    ap.add_argument("--image_processor", default=None)
    ap.add_argument("--synthetic", type=int, default=0,
                    help="clip_vision: use N synthetic noise images. ⛔ PLUMBING TEST ONLY. "
                         "The resulting C_H describes noise, not any real input distribution, "
                         "and must never reach a table.")
    args = ap.parse_args()

    if args.arch == "causal_lm" and not args.data:
        raise SystemExit("[refuse] --arch causal_lm needs --data")
    if args.arch == "clip_vision" and not (args.images or args.synthetic):
        raise SystemExit("[refuse] --arch clip_vision needs --images or --synthetic")
    if args.exact_cov and args.arch == "causal_lm":
        print("[cov] ⚠️  --exact_cov on a 7B model needs ~280 GB. This will almost certainly "
              "fail; it is here for small models.", flush=True)
    if not args.exact_cov and args.sketch < 4 * args.m_out:
        print(f"[cov] ⚠️  --sketch {args.sketch} is small next to --m_out {args.m_out}. "
              f"At sketch=64 / m_out=32 the subspace overlap was 0.505 in the synthetic "
              f"check, against 0.998 at sketch=128. Use sketch >= 4 x m_out.", flush=True)

    backend = BACKENDS[args.arch](args)
    model = backend.model
    for prm in model.parameters():
        prm.requires_grad_(False)

    keep = tuple(x.strip() for x in (args.include or backend.default_modules).split(",") if x.strip())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    targets = {n: m for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.split(".")[-1] in keep}
    if not targets:
        raise SystemExit(f"[refuse] no nn.Linear matched {keep}. Check --include against the "
                         f"architecture; CLIP uses out_proj/fc1/fc2, Qwen uses o_proj/up_proj.")
    print(f"[cov] arch={args.arch}  {len(targets)} linear modules match {keep}", flush=True)

    accum: dict = {}
    cur_mask: dict = {}

    def rows_of(inp):
        x = inp[0]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        mask = cur_mask.get("m")
        if mask is None:
            # Backends with no padding (vision) pass None deliberately. Text must not.
            if cur_mask.get("requires_mask"):
                raise RuntimeError("a padded backend produced no mask; refusing to accumulate")
            return x
        if mask.numel() != x.shape[0]:
            # ⛔ Padding rows would bias C_H toward the pad embedding, which no real input
            # visits. Silently skipping the filter would corrupt the covariance with no signal
            # that anything went wrong, so it is fatal instead.
            raise RuntimeError(
                f"attention mask has {mask.numel()} entries but the module saw {x.shape[0]} "
                f"rows. The token axis does not line up, so padding cannot be removed.")
        return x[mask]

    def hook1(name):
        def fn(_mod, inp, _out):
            x = rows_of(inp)
            if x.shape[0] == 0:
                return
            if name not in accum:
                accum[name] = (ExactFull(x.shape[-1], x.device) if args.exact_cov
                               else FrequentDirections(x.shape[-1], args.sketch, x.device))
            accum[name].add(x.detach().float())
        return fn

    def sweep(tag):
        t0 = time.time()
        with torch.no_grad():
            for kwargs, mask, done in backend.batches():
                cur_mask["m"] = mask
                cur_mask["requires_mask"] = (args.arch == "causal_lm")
                model(**kwargs)
                if done % (args.batch * 10) == 0 or done == len(backend):
                    el = time.time() - t0
                    print(f"[{tag}] {done}/{len(backend)}  {el:.0f}s  "
                          f"eta {el/done*(len(backend)-done):.0f}s", flush=True)

    handles = [m.register_forward_hook(hook1(n)) for n, m in targets.items()]
    sweep("pass1/exact" if args.exact_cov else "pass1/FD")
    for h in handles:
        h.remove()

    # ---- pass 2: exact spectrum inside the sketched subspace (sketch path only) ------------
    exact, sketch_w = {}, {}
    if not args.exact_cov:
        for name, fd in accum.items():
            w_b, V = fd.eig(args.m_out)
            sketch_w[name] = w_b
            exact[name] = ExactInSubspace(V)

        def hook2(name):
            def fn(_mod, inp, _out):
                x = rows_of(inp)
                if x.shape[0] and name in exact:
                    exact[name].add(x.detach())
            return fn

        handles = [m.register_forward_hook(hook2(n)) for n, m in targets.items() if n in exact]
        sweep("pass2/exact")
        for h in handles:
            h.remove()

    manifest = dict(model=args.model, arch=args.arch, sketch_m=args.sketch,
                    exact_cov=bool(args.exact_cov), m_out=args.m_out, dtype=args.dtype,
                    modules={}, **backend.provenance())
    for name, acc in sorted(accum.items()):
        if args.exact_cov:
            w, V = acc.eig(args.m_out)
            bias = None
        else:
            w, V = exact[name].eig()
            w_b = sketch_w[name]
            bias = float((w[:len(w_b)] - w_b).abs().max() / max(float(w[0]), 1e-12))
        # ⛔ trace_full = E[||h||^2] = tr(C_H), NOT the sum of the stored eigenvalues. The
        # difference is the energy in the (d - m_out) directions that were never stored, and
        # a null model that omits it is confined to the top-m subspace: on CLIP fc1, two such
        # nulls overlapped at 0.649 where the isotropic expectation is k/d = 0.010, because
        # both were trapped in the same 64 of 768 dimensions. analysis/act_test.py needs this
        # number to give the null an isotropic tail with the right residual variance.
        trace_full = acc.fro2 / max(acc.n_rows, 1)
        torch.save({"eigvals": w, "eigvecs": V, "d": acc.d, "n_rows": acc.n_rows,
                    "fro2": acc.fro2, "trace_full": trace_full,
                    "sketch_m": None if args.exact_cov else acc.m,
                    "sketch_eigvals": None if args.exact_cov else sketch_w[name],
                    "source": ("exact full covariance" if args.exact_cov
                               else "FD subspace + exact in-subspace spectrum (2 passes)")},
                   out / f"{name.replace('.', '__')}.pt")
        # Participation ratio (sum w)^2 / sum w^2: how many directions the data actually
        # determines. ⛔ If --m_out or the test's --m exceeds this, the extra directions are
        # not estimated, they are arbitrary, and two runs on the same data will disagree
        # about them. Measured on CLIP ViT-B/32 with synthetic noise inputs, the median
        # participation ratio is 1.99, and there the top-8 subspace agrees between the exact
        # and sketched covariance to a minimum of 0.978 while the top-32 falls to 0.611 --
        # not a sketch failure, a request for more directions than the inputs excite.
        pr = acc.pr_full() if args.exact_cov else float(
            (w.sum() ** 2 / torch.clamp((w ** 2).sum(), min=1e-30)))
        manifest["modules"][name] = dict(
            d=acc.d, n_rows=acc.n_rows, participation_ratio=pr,
            participation_ratio_is_lower_bound=(not args.exact_cov),
            fd_spectral_bound=(None if args.exact_cov
                               else acc.fro2 / acc.m / max(acc.n_rows, 1)),
            sketch_eigval_bias_rel_top1=bias,
            trace_full=trace_full, trace_stored=float(w.sum()),
            tail_energy_fraction=float(1.0 - w.sum() / max(trace_full, 1e-30)),
            top1=float(w[0]), median=float(w[len(w) // 2]), tail=float(w[-1]),
            cond_in_subspace=float(w[0] / max(float(w[-1]), 1e-12)),
            trace_captured=float(w.sum()))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[cov] wrote {len(accum)} covariances + manifest.json to {out}")

    pr = [v["participation_ratio"] for v in manifest["modules"].values()]
    # On the sketch path the stored PR is a lower bound capped at m_out, so this gate carries
    # no information there and says so rather than firing on every module.
    thin = ([] if not args.exact_cov else
            [n for n, v in manifest["modules"].items() if v["participation_ratio"] < args.m_out])
    print(f"[cov] participation ratio (directions the data determines): median "
          f"{sorted(pr)[len(pr)//2]:.1f}, min {min(pr):.1f}, max {max(pr):.1f}")
    if thin:
        print(f"[cov] ⛔ {len(thin)}/{len(pr)} modules have a participation ratio BELOW "
              f"--m_out={args.m_out}. Directions past it are undetermined, not estimated. "
              f"Pick the test's --m below the MINIMUM participation ratio here, or widen the "
              f"calibration set until the spectrum supports the m you need. Worst: "
              f"{sorted(thin, key=lambda n: manifest['modules'][n]['participation_ratio'])[:3]}")
    tf = [v["tail_energy_fraction"] for v in manifest["modules"].values()]
    print(f"[cov] energy outside the stored {args.m_out} directions: median "
          f"{sorted(tf)[len(tf)//2]:.3f}, max {max(tf):.3f}. act_test.py spreads this over the "
          f"complement so the null is not confined to the stored subspace.")
    c = [v["cond_in_subspace"] for v in manifest["modules"].values()]
    print(f"[cov] in-subspace condition number: median {sorted(c)[len(c)//2]:.1e}, max {max(c):.1e}. "
          f"Feed these to analysis/act_whitening_calibration.py -- the type-I inflation scales "
          f"with this number and sets the truncation m.")
    if not args.exact_cov:
        b = [v["sketch_eigval_bias_rel_top1"] for v in manifest["modules"].values()]
        print(f"[cov] FD eigenvalue bias corrected by pass 2: max {max(b):.3f}, median "
              f"{sorted(b)[len(b)//2]:.3f} (relative to the top eigenvalue). Large values here "
              f"are expected and are exactly why pass 2 exists.")
        print("[cov] CHECK BEFORE USING: fd_spectral_bound must be small next to the "
              "eigenvalues you intend to keep. If it is not, raise --sketch or --max_seqs.")
    if getattr(backend, "synthetic", False):
        print("[cov] ⛔ SYNTHETIC INPUTS. This C_H describes noise. Plumbing check only.")


if __name__ == "__main__":
    main()
