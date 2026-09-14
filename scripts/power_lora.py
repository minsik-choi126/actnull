"""LoRA 널의 검정력: 공유 초기화 풀에 알려진 양의 진짜 공유 방향을 심고 회수율을 잰다.

⛔ 목표값이 s/k 가 아니다. LoRA 풀은 row(dW) ⊆ row(A) 이고 A 를 공유하므로 기준 겹침 O_0 이
0.7 근처다. s 개를 정확히 공유시키면 나머지 k-s 개는 여전히 O_0 로 겹치므로 늘어나는 양은
(s/k)(1-O_0) 이지 s/k 가 아니다. 활성 널에서는 O_0=0.026 이라 2.6% 차이였지만 여기서는 70% 다.
"""
import sys, os, json, torch, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actnull.null import (AdapterCheckpoint, LazyCheckpoint, load_delta, top_right, overlap,
                      lora_shared_init_null)
import actnull.null as _cur
import importlib.util as _il
_sp = _il.spec_from_file_location("ah", os.path.join(os.path.dirname(__file__), "act_test_HEAD.py"))
_head = _il.module_from_spec(_sp); _sp.loader.exec_module(_head)
WHICH = os.environ.get("NULL_VERSION", "current")
null_fn = _head.lora_shared_init_null if WHICH == "shipped" else _cur.lora_shared_init_null

def plant(ref, N, s, k):
    if s == 0: return N
    Wa = torch.linalg.svd(ref.float(), full_matrices=False)[2].T
    Un, Sn, Wnh = torch.linalg.svd(N.float(), full_matrices=False)
    Wn = Wnh.T
    keep = Wa[:, :s]
    rest = Wn - keep @ (keep.T @ Wn)
    rest, _ = torch.linalg.qr(rest[:, :max(Wn.shape[1] - s, 1)])
    W = torch.cat([keep, rest], 1)
    r = min(W.shape[1], len(Sn))
    return ((Un[:, :r] * Sn[:r]) @ W[:, :r].T).to(ref)

def run(arch, mods, tasks, k=8, draws=12, seed=0):
    base = LazyCheckpoint(f"openai/{arch}", "vision_model.")
    E = {t: AdapterCheckpoint(f"tanganke/{arch}_{t}_lora-16") for t in tasks}
    names = sorted(E)
    out = {}
    for s in (0, 1, 2, 4):
        raws, nulls = [], []
        for mod in mods:
            key = mod + ".weight"
            g = torch.Generator().manual_seed(seed)
            gp = torch.Generator().manual_seed(seed + 10_000)
            for i in range(draws):
                a, b = names[i % len(names)], names[(i + 1 + i // len(names)) % len(names)]
                if a == b:
                    continue
                dWa = load_delta(base, E[a], key, "cpu")
                if dWa is None or float(dWa.norm()) == 0:
                    continue
                Ab = E[b].mods[mod]["lora_A"].float()
                # 파트너는 항상 교정된 널로 뽑는다 (arm 무관)
                N = _cur.lora_shared_init_null(Ab, E[b].mods[mod]["lora_B"].float(),
                                               E[b].scale, gp)
                P = plant(dWa, N, s, k)
                raws.append(overlap(top_right(dWa, k), top_right(P, k)))
                # ⛔ 널은 심긴 객체 자체에서 유도해야 한다. 원래 어댑터에서 다시 뽑으면
                # 널이 P 를 보지 않으므로 심은 구조를 파괴했는지 잴 수 없다.
                # row(P) ⊆ row(A) 이므로 B_P = P A^+ 가 P 의 B 인자다.
                Bp = P @ torch.linalg.pinv(Ab) / E[b].scale
                na = null_fn(E[a].mods[mod]["lora_A"].float(),
                             E[a].mods[mod]["lora_B"].float(), E[a].scale, g)
                nb = null_fn(Ab, Bp, E[b].scale, g)
                nulls.append(overlap(top_right(na, k), top_right(nb, k)))
        out[s] = (st.mean(raws), st.mean(nulls))
    return out


if __name__ == "__main__":
    arch = "clip-vit-base-patch16"
    T = ["dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
    mods = ["encoder.layers.4.self_attn.q_proj", "encoder.layers.4.self_attn.v_proj",
            "encoder.layers.8.self_attn.q_proj", "encoder.layers.8.self_attn.v_proj"]
    r = run(arch, mods, T)
    o0 = r[0][0]
    print(f"  기준 겹침 O_0 = {o0:.4f}")
    print(f"  {'심은':>6s} {'목표':>8s} {'raw':>8s} {'null':>8s} {'초과':>9s} {'회수율':>7s}")
    rows = []
    for s, (raw, nl) in r.items():
        truth = (s / 8) * (1 - o0)
        ex = raw - nl
        rec = "  -" if s == 0 else f"{100*(raw-r[0][0]-(nl-r[0][1]))/truth:6.0f}%"
        print(f"  {s:4d}/8 {truth:8.4f} {raw:8.4f} {nl:8.4f} {ex:+9.4f} {rec:>7s}")
        rows.append(dict(planted=s, k=8, O0=o0, truth=truth, raw=raw, null=nl, excess=ex))
    json.dump(dict(arch=arch, modules=mods, tasks=T, rows=rows),
              open(fos.path.join(os.environ.get("ACTNULL_RESULTS", "results"), "power_lora_{WHICH}.json"), "w"), indent=1)
