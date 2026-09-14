"""이 문헌이 쓰는 기준선들이 진실이 0인 데이터에서 무엇을 보고하는가.

⛔ 왜 이 실험이 필요한가. 심기 프로토콜을 우리 널에만 적용하면 자기참조다.
같은 참 널 쌍을 문헌의 기준선들에 통과시키면, 그 기준선들이 아무 공유 구조도 없는
데이터에서 얼마를 '공유 구조'로 보고하는지 나온다.

참 널 쌍 (act_test.h0_pair): 활성 기하와 스펙트럼은 공유하되 방향 합의는 없는 두 업데이트.
공통 베이스에서 나왔지만 서로 무관하게 학습된 두 전문가가 만족해야 할 조건. 진실 = 0.
"""
import os
import sys, json, torch, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import actnull.null as at
from actnull.null import (top_right, overlap, resolvable_blocks, LazyCheckpoint, load_delta,
                      truncated_whiten)
SP = Path(__file__).resolve().parent

def run(arch, mods, experts, k=8, draws=20, seed=0):
    cov = SP / f"k1000_cov_{arch}"
    man = json.loads((cov / "manifest.json").read_text())
    base = LazyCheckpoint(f"openai/{arch}", "vision_model.")
    E = {n: LazyCheckpoint(p, "vision_model.") for n, p in experts.items()}
    names = sorted(E)
    KEYS = ["orthogonality", "isotropic", "regmean", "tikhonov", "generative", "permutation", "ours"]
    acc = {kk: [] for kk in KEYS}
    isos = []
    for mod in mods:
        blob = torch.load(cov / f"{mod.replace('.','__')}.pt", weights_only=False)
        V, w, d_in, tr = blob["eigvecs"], blob["eigvals"], blob["d"], blob["trace_full"]
        fw = [torch.load(SP/f"k1000_fold_{arch}"/f"f{i}"/f"{mod.replace('.','__')}.pt",
                         weights_only=False)["eigvals"].double() for i in range(8)]
        L = min(len(x) for x in fw)
        se = (torch.stack([x[:L] for x in fw]).std(0)/8**0.5).float()
        blocks = resolvable_blocks(w[:L], se, 2.0)
        m_pr = max(1, min(int(man["modules"][mod]["participation_ratio"]), V.shape[0]))
        key = mod + ".weight"
        g = torch.Generator().manual_seed(seed)
        for i in range(draws):
            dW = load_delta(base, E[names[i % len(names)]], key, "cpu")
            if dW is None or float(dW.norm()) == 0: continue
            # ⛔ h0_pair 는 쓰지 않는다. 그 객체는 top-8 read mass 의 0.996 을 span(V) 안에
            # 두는데 실제 task vector 는 0.40 이다. 기각된 생성적 널의 프로파일이므로
            # 그 위에서 잰 기준선 수준은 현실적 데이터에 대한 진술이 아니다.
            #
            # 대신 실제 task vector 두 개를 각각 교정 널에 독립적으로 통과시킨다. 스펙트럼과
            # 읽기 질량 프로파일은 실제 그대로, 방향 합의만 없다. 진실 = 0.
            dW2 = load_delta(base, E[names[(i + 1) % len(names)]], key, "cpu")
            if dW2 is None or float(dW2.norm()) == 0: continue
            A0 = at.coupling_destroying_null(dW, V, 0, g, 128, blocks=blocks)
            B0 = at.coupling_destroying_null(dW2, V, 0, g, 128, blocks=blocks)
            o = overlap(top_right(A0, k), top_right(B0, k))
            iso = k / d_in; isos.append(iso)
            # (a) 직교성 기준: 대조점이 0 (gargiulo 계열)
            acc["orthogonality"].append(o - 0.0)
            # (b) 등방 k/d: 문헌 표준
            acc["isotropic"].append(o - iso)
            # (c,d) 화이트닝 후 등방 기준
            for tag, mm in (("regmean", m_pr), ("tikhonov", V.shape[0])):
                ca, cb = (top_right(truncated_whiten(x, w, V, mm), k) for x in (A0, B0))
                acc[tag].append(overlap(ca, cb) - iso)
            # (e) 생성적 널 (§3.2 에서 기각)
            ga = at.act_matched_null(A0, w, V, tr, g); gb = at.act_matched_null(B0, w, V, tr, g)
            acc["generative"].append(o - overlap(top_right(ga, k), top_right(gb, k)))
            # (f) 좌표 순열 (§3.2 에서 기각)
            perm = torch.randperm(d_in, generator=g)
            acc["permutation"].append(o - overlap(top_right(A0[:, perm], k), top_right(B0[:, perm], k)))
            # (g) 우리 널
            na = at.coupling_destroying_null(A0, V, 0, g, 128, blocks=blocks)
            nb = at.coupling_destroying_null(B0, V, 0, g, 128, blocks=blocks)
            acc["ours"].append(o - overlap(top_right(na, k), top_right(nb, k)))
    return acc, st.mean(isos)

if __name__ == "__main__":
    arch = "clip-vit-base-patch16"
    T = ["dtd","eurosat","gtsrb","mnist","resisc45","stanford-cars","sun397","svhn"]
    exp = {t: f"tanganke/{arch}_{t}" for t in T}
    man = json.loads((SP/f"k1000_cov_{arch}"/"manifest.json").read_text())
    mods = sorted(man["modules"])[::9]
    acc, iso = run(arch, mods, exp)
    LAB = [("orthogonality", "exact orthogonality"), ("isotropic", "isotropic $k/d$"),
           ("regmean", "whitened at participation ratio"), ("tikhonov", "whitened, all stored"),
           ("generative", "generative null"), ("permutation", "coordinate permutation"),
           ("ours", "activation-conditioned (ours, circular here)")]
    print(f"\n  참 널 쌍에서 보고되는 초과. 진실 = 0.  등방 수준 k/d = {iso:.5f}")
    print(f"  모듈 {len(mods)}, 표본 {len(acc['ours'])}\n")
    print("  %-34s %11s %10s" % ("기준선", "보고 초과", "우연 대비"))
    out = {}
    for kk, lab in LAB:
        v = st.mean(acc[kk]); out[kk] = v
        print("  %-34s %+11.5f %9.1fx" % (lab, v, (iso + v) / iso))
    json.dump(dict(arch=arch, modules=mods, k=8, iso=iso, n=len(acc["ours"]), level=out),
              open(os.path.join(os.environ.get("ACTNULL_RESULTS", "results"), "baseline_levels.json"),"w"),
              indent=1)
    print("\n  wrote results/act/baseline_levels.json")
