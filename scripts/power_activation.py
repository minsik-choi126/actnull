"""검정력(power) 연구: 알려진 양의 진짜 공유 구조를 심고 검정이 그것을 회수하는지 잰다.

⛔ 왜 필요한가. H0 검사는 널이 '너무 많이 부수는지'만 본다. 아무것도 하지 않는 항등 널도
H0 = 0 을 받는다. 그런데 과소파괴는 explained 를 부풀리는 방향이고, explained 는 검정력에
관한 진술이다. 수준(level)만 검증하고 검정력을 재지 않으면 헤드라인을 뒷받침할 수 없다.

설계. 실제 C_H, 실제 블록 구조, 실제 task vector 를 쓴다.
  - 파트너 N = coupling_destroying_null(dW_b)  -> dW_a 와 H0 관계 (진짜 공유 0)
  - N 의 상위 s 개 read 방향을 dW_a 의 것으로 강제 -> 정확히 s/k 만큼 진짜로 공유
  - 진실: excess 는 대략 s/k 여야 하고, explained 는 심은 몫에 대해 0 이어야 한다
"""
import sys, json, torch, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util as _il, os
from actnull.null import top_right, overlap, resolvable_blocks, LazyCheckpoint, load_delta
import actnull.null as _cur
_sp = _il.spec_from_file_location("act_head", os.path.join(os.path.dirname(__file__), "act_test_HEAD.py"))
_head = _il.module_from_spec(_sp); _sp.loader.exec_module(_head)
_WHICH = os.environ.get("NULL_VERSION", "current")
# ⛔ THE PARTNER MUST NOT COME FROM THE NULL UNDER TEST. It did in the first version, so the
# two arms were scored on different test data: raw(s=0) was 0.0675 for the shipped null and
# 0.0310 for the corrected one, a factor of 2.2, and the headline 44% against 91% compared
# recoveries measured on pairs that were not the same pairs. The partner is now always drawn
# with the corrected null, for both arms, so only the measurement changes between them.
coupling_destroying_null = (_head.coupling_destroying_null if _WHICH == "shipped"
                            else _cur.coupling_destroying_null)
partner_null = _cur.coupling_destroying_null

SP = Path(os.environ.get("ACTNULL_WORK", "work"))

def plant(dWa, N, s, k):
    """N 의 상위 s 개 read 방향을 dWa 의 것으로 교체. 스펙트럼은 N 의 것을 유지."""
    if s == 0:
        return N
    Ua, Sa, Wah = torch.linalg.svd(dWa.float(), full_matrices=False)
    Un, Sn, Wnh = torch.linalg.svd(N.float(), full_matrices=False)
    Wa, Wn = Wah.T, Wnh.T                              # (d_in, r)
    keep = Wa[:, :s]
    rest = Wn - keep @ (keep.T @ Wn)                   # 나머지를 심은 방향에 직교화
    rest, _ = torch.linalg.qr(rest[:, :Wn.shape[1] - s])
    W = torch.cat([keep, rest], 1)
    r = min(W.shape[1], len(Sn))
    return ((Un[:, :r] * Sn[:r]) @ W[:, :r].T).to(dWa)

def study(arch, mods, experts, k=8, draws=10, seed=0):
    cov = SP / f"k1000_cov_{arch}"
    man = json.loads((cov / "manifest.json").read_text())
    base = LazyCheckpoint(f"openai/{arch}", "vision_model.")
    E = {n: LazyCheckpoint(p, "vision_model.") for n, p in experts.items()}
    names = sorted(E)
    out = {}
    for s in (0, 1, 2, 4):
        rows = []
        for mod in mods:
            blob = torch.load(cov / f"{mod.replace('.','__')}.pt", weights_only=False)
            V0, w = blob["eigvecs"], blob["eigvals"]
            fw = [torch.load(SP / f"k1000_fold_{arch}" / f"f{i}" / f"{mod.replace('.','__')}.pt",
                             weights_only=False)["eigvals"].double() for i in range(8)]
            L = min(len(x) for x in fw)
            se = (torch.stack([x[:L] for x in fw]).std(0) / 8 ** 0.5).float()
            blocks = resolvable_blocks(w[:L], se, 2.0)
            # act_test 는 널에 저장된 고유방향 전체를 넘긴다. participation ratio 는
            # 화이트닝(cond) 경로에만 쓰이므로 여기서 자르면 블록 인덱스와 어긋난다.
            V = V0
            key = mod + ".weight"
            g = torch.Generator().manual_seed(seed)
            gp = torch.Generator().manual_seed(seed + 10_000)   # partner stream, arm-independent
            for i in range(draws):
                a, b = names[i % len(names)], names[(i + 1 + i // len(names)) % len(names)]
                if a == b: continue
                dWa = load_delta(base, E[a], key, "cpu")
                dWb = load_delta(base, E[b], key, "cpu")
                if float(dWa.norm()) == 0 or float(dWb.norm()) == 0: continue
                N = partner_null(dWb, V, 0, gp, 128, blocks=blocks)
                P = plant(dWa, N, s, k)                       # s/k 만큼 진짜 공유
                raw = overlap(top_right(dWa, k), top_right(P, k))
                nl = st.mean(overlap(top_right(coupling_destroying_null(dWa, V, 0, g, 128, blocks=blocks), k),
                                     top_right(coupling_destroying_null(P, V, 0, g, 128, blocks=blocks), k))
                             for _ in range(4))
                rows.append((raw, nl, blob["d"]))
        raw = st.mean(r for r, _, _ in rows); nl = st.mean(n for _, n, _ in rows)
        iso = st.mean(k / d for _, _, d in rows)
        out[s] = (raw, nl, iso, raw - nl, s / k)
    return out

if __name__ == "__main__":
    arch = "clip-vit-base-patch16"
    T = ["dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
    exp = {t: f"tanganke/{arch}_{t}" for t in T}
    mods = ["encoder.layers.4.self_attn.q_proj", "encoder.layers.4.mlp.fc1",
            "encoder.layers.8.self_attn.v_proj", "encoder.layers.8.mlp.fc2"]
    r = study(arch, mods, exp)
    print(f"  {'심은 공유':>9s} {'진실 초과':>9s} {'raw':>8s} {'null':>8s} {'측정 초과':>9s} "
          f"{'회수율':>7s} {'전달률':>7s}")
    rows = []
    base_raw = r[0][0]
    base_null = r[0][1]
    for s, (raw, nl, iso, ex, truth) in r.items():
        rec = "  -" if truth == 0 else f"{100*ex/truth:6.0f}%"
        # 전달률: 심기가 실제로 전달한 overlap / 명목 s/k. 널과 무관한 양성통제 자체의 충실도.
        deliv = None if truth == 0 else 100 * (raw - base_raw) / truth
        dv = "  -" if deliv is None else f"{deliv:6.1f}%"
        print(f"  {s:5d}/8   {truth:9.4f} {raw:8.4f} {nl:8.4f} {ex:+9.4f} {rec:>7s} {dv:>7s}")
        rows.append(dict(planted=s, k=8, truth=truth, raw=raw, null=nl, iso=iso, excess=ex,
                         recovery=None if truth == 0 else 100 * ex / truth,
                         delivered=deliv,
                         # 누출률: 심은 구조가 널을 통과해 살아남은 비율. 널만 고립시킨다.
                         leak=None if truth == 0 else 100 * (nl - base_null) / truth))
    import json, os
    out = (os.environ.get("ACTNULL_RESULTS", "results") + "/"
           f"power_{os.environ.get('NULL_VERSION', 'current')}.json")
    json.dump(dict(arch=arch, modules=mods, experts=sorted(exp), rows=rows),
              open(out, "w"), indent=1)
    print(f"  wrote {out}")
