"""결함별 ablation: 수정 1(여집합 전체 회전)과 수정 2(싱글턴 부호)의 기여를 분리한다.

R1 의 지적: 44%->91% 가 네 결함 "전체"의 효과로만 제시되고, 실제로는 결함 1 하나가 거의
전부일 수 있다. 네 줄이면 답이 나온다.
"""
import sys, os, json, torch, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actnull.null import top_right, overlap, resolvable_blocks, LazyCheckpoint, load_delta
SP = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(SP))
from power import plant                      # 같은 심기 구성을 재사용

def null_variant(dW, V, gen, blocks, full_complement, flip_singletons, tail_rank=128):
    """coupling_destroying_null 의 두 수정을 각각 켜고 끌 수 있는 판본."""
    m, d_in = V.shape
    Vd = V.to(dW)
    C = dW @ Vd.T
    R = dW - C @ Vd
    for blk in blocks:
        n = len(blk)
        idx = torch.tensor(blk, device=dW.device)
        if n < 2:
            if flip_singletons:
                sgn = 1.0 - 2.0 * float(torch.randint(0, 2, (1,), generator=gen))
                C[:, idx] = C[:, idx] * sgn
            continue
        Q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float32))
        C[:, idx] = C[:, idx] @ Q.to(dW)
    if full_complement:
        rank = int(min(R.shape[0], max(d_in - m, 0)))
        if rank >= 2 and float(R.norm()) > 0:
            # act_test.py 와 같은 대체 경로. gesdd 는 이 잔차에서 종종 수렴에 실패한다.
            try:
                Ur, Sr, _ = torch.linalg.svd(R.double(), full_matrices=False)
                M = Ur[:, :rank] * Sr[:rank]
            except torch._C._LinAlgError:
                Rd = R.double()
                ev, U = torch.linalg.eigh(Rd @ Rd.T)
                ev, U = ev.flip(0)[:rank].clamp(min=0.0), U.flip(1)[:, :rank]
                M = U * ev.sqrt()
            G = torch.randn(d_in, rank, generator=gen, dtype=torch.float32).double()
            G = G - Vd.T.double() @ (Vd.double() @ G)
            G, _ = torch.linalg.qr(G)
            R = (M @ G.T).to(dW)
    else:
        j = min(tail_rank, max(d_in - m, 0))
        if j >= 2:
            B = torch.randn(d_in, j, generator=gen, dtype=torch.float32)
            B = B - Vd.T.float() @ (Vd.float() @ B)
            B, _ = torch.linalg.qr(B); B = B.to(dW)
            Q, _ = torch.linalg.qr(torch.randn(j, j, generator=gen, dtype=torch.float32))
            P = R @ B
            R = R - P @ B.T + (P @ Q.to(dW)) @ B.T
    return C @ Vd + R

def study(arch, mods, experts, k=8, draws=10, seed=0):
    cov = SP / f"k1000_cov_{arch}"
    man = json.loads((cov / "manifest.json").read_text())
    base = LazyCheckpoint(f"openai/{arch}", "vision_model.")
    E = {n: LazyCheckpoint(p, "vision_model.") for n, p in experts.items()}
    names = sorted(E)
    VAR = [("shipped (둘 다 꺼짐)", False, False),
           ("싱글턴 부호만", False, True),
           ("여집합 전체만", True, False),
           ("corrected (둘 다)", True, True)]
    acc = {v[0]: {"raw": [], "null": []} for v in VAR}
    for mod in mods:
        blob = torch.load(cov / f"{mod.replace('.','__')}.pt", weights_only=False)
        V, w = blob["eigvecs"], blob["eigvals"]
        fw = [torch.load(SP / f"k1000_fold_{arch}" / f"f{i}" / f"{mod.replace('.','__')}.pt",
                         weights_only=False)["eigvals"].double() for i in range(8)]
        L = min(len(x) for x in fw)
        se = (torch.stack([x[:L] for x in fw]).std(0) / 8 ** 0.5).float()
        blocks = resolvable_blocks(w[:L], se, 2.0)
        key = mod + ".weight"
        for lab, fc, fs in VAR:
            g = torch.Generator().manual_seed(seed)
            gp = torch.Generator().manual_seed(seed + 10_000)
            for i in range(draws):
                a, b = names[i % len(names)], names[(i + 1 + i // len(names)) % len(names)]
                if a == b: continue
                dWa = load_delta(base, E[a], key, "cpu"); dWb = load_delta(base, E[b], key, "cpu")
                if float(dWa.norm()) == 0 or float(dWb.norm()) == 0: continue
                # 파트너는 항상 교정 널 (arm 무관)
                N = null_variant(dWb, V, gp, blocks, True, True)
                P = plant(dWa, N, 2, k)                     # s=2 고정
                acc[lab]["raw"].append(overlap(top_right(dWa, k), top_right(P, k)))
                acc[lab]["null"].append(overlap(
                    top_right(null_variant(dWa, V, g, blocks, fc, fs), k),
                    top_right(null_variant(P, V, g, blocks, fc, fs), k)))
    return acc

if __name__ == "__main__":
    arch = "clip-vit-base-patch16"
    T = ["dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
    exp = {t: f"tanganke/{arch}_{t}" for t in T}
    mods = ["encoder.layers.4.self_attn.q_proj", "encoder.layers.4.mlp.fc1",
            "encoder.layers.8.self_attn.v_proj", "encoder.layers.8.mlp.fc2"]
    acc = study(arch, mods, exp)
    base_raw = st.mean(acc["corrected (둘 다)"]["raw"])
    print("  심은 공유 2/8, 목표 (2/8)(1-O_0)")
    print(f"  {'변형':>20s} {'raw':>8s} {'null':>8s} {'초과':>9s} {'회수율':>7s}")
    out = []
    for lab in acc:
        raw, nl = st.mean(acc[lab]["raw"]), st.mean(acc[lab]["null"])
        print(f"  {lab:>20s} {raw:8.4f} {nl:8.4f} {raw-nl:+9.4f}")
        out.append(dict(variant=lab, raw=raw, null=nl, excess=raw-nl))
    json.dump(out, open(os.path.join(os.environ.get("ACTNULL_RESULTS", "results"), "ablation.json"),"w"),
              ensure_ascii=False, indent=1)
