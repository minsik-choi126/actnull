"""랭크 예산을 고정하고 tau 기저와 교차항 기저의 머징 정확도를 비교한다.

머징 업데이트를 랭크 r 부분공간으로 압축한다:
    theta = theta_0 + lambda * sum_t (dW_t V V^T)
V 를 tau^T tau 의 상위 r (문헌이 쓰는 것) 또는 G_cross 의 상위 r (이 논문의 처방)로 둔다.
주장: 같은 정확도를 교차항 기저가 훨씬 작은 r 로 낸다.

평가는 CLIP 제로샷. 텍스트 타워는 얼려두고 비전 타워만 머징한다.
"""
import argparse, json, sys, torch, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actnull.null import LazyCheckpoint, load_delta
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from datasets import load_dataset

TASKS = ["dtd", "eurosat", "gtsrb", "mnist", "resisc45", "stanford-cars", "sun397", "svhn"]
DSET = {"dtd": ("tanganke/dtd", None), "eurosat": ("tanganke/eurosat", None),
        "gtsrb": ("tanganke/gtsrb", None), "mnist": ("ylecun/mnist", None),
        "resisc45": ("tanganke/resisc45", None), "stanford-cars": ("tanganke/stanford_cars", None),
        "sun397": ("tanganke/sun397", None), "svhn": ("ufldl-stanford/svhn", "cropped_digits")}
TEMPL = ["a photo of a {}.", "a photo of the {}.", "a blurry photo of a {}."]

def text_emb(model, tok, prompts, device):
    """transformers 5.x 의 get_text_features 는 텐서가 아니라 출력 객체를 돌려주고, 그 안의
    pooler_output 이 사영 전인지 후인지가 버전에 따라 모호하다. 명시 경로로 고정한다."""
    t = tok(prompts, padding=True, return_tensors="pt")
    o = model.text_model(**{k: v.to(device) for k, v in t.items()})
    return model.text_projection(o.pooler_output)

def img_emb(model, px, device):
    o = model.vision_model(pixel_values=px.to(device))
    return model.visual_projection(o.pooler_output)

def heads(model, tok, names, device):
    with torch.no_grad():
        embs = []
        for c in names:
            e = text_emb(model, tok, [s.format(c.replace("_", " ")) for s in TEMPL], device)
            e = e / e.norm(dim=-1, keepdim=True)
            m = e.mean(0)
            embs.append(m / m.norm())
    return torch.stack(embs)

def testset(task, n, seed=0):
    name, cfg = DSET[task]
    ds = load_dataset(name, cfg, split="test") if cfg else load_dataset(name, split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    lab = "label" if "label" in ds.features else next(k for k in ds.features if "label" in k)
    img = "image" if "image" in ds.features else next(k for k in ds.features if "im" in k)
    names = ds.features[lab].names
    return ds, img, lab, names

def evaluate(model, proc, tok, sets, device, batch=32):
    accs = {}
    for task, (ds, img, lab, names) in sets.items():
        H = heads(model, tok, names, device)
        ok = tot = 0
        with torch.no_grad():
            for i in range(0, len(ds), batch):
                ch = ds[i:i + batch]
                px = proc(images=[im.convert("RGB") for im in ch[img]], return_tensors="pt")
                f = img_emb(model, px["pixel_values"], device)
                f = f / f.norm(dim=-1, keepdim=True)
                pred = (f @ H.T).argmax(-1).cpu()
                ok += int((pred == torch.tensor(ch[lab])).sum()); tot += len(pred)
        accs[task] = 100.0 * ok / tot
        print(f"    {task:14s} {accs[task]:5.2f}%  (n={tot})", flush=True)
    return accs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="clip-vit-base-patch16")
    ap.add_argument("--rp", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    model = CLIPModel.from_pretrained(f"openai/{a.arch}").to(dev).eval()
    proc = CLIPProcessor.from_pretrained(f"openai/{a.arch}")
    tok = CLIPTokenizer.from_pretrained(f"openai/{a.arch}")
    print("[eval] 테스트셋 적재", flush=True)
    sets = {t: testset(t, a.n_test) for t in TASKS}

    base = LazyCheckpoint(f"openai/{a.arch}", "vision_model.")
    E = {t: LazyCheckpoint(f"tanganke/{a.arch}_{t}", "vision_model.") for t in TASKS}
    lins = [n for n, m in model.vision_model.named_modules() if isinstance(m, torch.nn.Linear)]
    print(f"[eval] 태스크 벡터 적재 ({len(lins)} 모듈)", flush=True)
    deltas, orig = {}, {}
    sd = model.vision_model.state_dict()
    for mod in lins:
        key = mod + ".weight"
        if key not in base or any(key not in e for e in E.values()): continue
        D = [load_delta(base, E[t], key, "cpu") for t in TASKS]
        if any(float(x.norm()) == 0 for x in D): continue
        deltas[mod] = D; orig[mod] = sd[key].clone()

    def apply(kind, r):
        for mod, D in deltas.items():
            tau = sum(D)
            G = (tau.T @ tau).double()
            if kind == "cross": G = G - sum((x.T @ x).double() for x in D)
            if kind == "full":
                upd = tau
            else:
                rr = min(r, G.shape[0] - 1)
                ev, U = torch.linalg.eigh(G)
                V = U.flip(1)[:, :rr].float()
                upd = sum(x @ V @ V.T for x in D)
            sd[mod + ".weight"].copy_(orig[mod] + a.lam * upd)
    def restore():
        for mod in deltas: sd[mod + ".weight"].copy_(orig[mod])

    res = {}
    print("\n[eval] zero-shot (머징 없음)", flush=True); restore()
    res["zeroshot"] = evaluate(model, proc, tok, sets, dev)
    print("\n[eval] task arithmetic (압축 없음)", flush=True); apply("full", 0)
    res["full"] = evaluate(model, proc, tok, sets, dev)
    for kind in ("tau", "cross"):
        for r in a.rp:
            print(f"\n[eval] {kind} r={r}", flush=True); apply(kind, r)
            res[f"{kind}_{r}"] = evaluate(model, proc, tok, sets, dev)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n[eval] wrote {a.out}")
    print(f"  {'설정':>12s} {'평균 정확도':>10s}")
    for k, v in res.items():
        print(f"  {k:>12s} {st.mean(v.values()):9.2f}%")

if __name__ == "__main__":
    main()
