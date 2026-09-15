#!/usr/bin/env python3
"""Evaluate rank-constrained B/32 task-arithmetic projector bases.

For each linear vision module, the script forms the cumulative task update and
compares two rank-r projection bases: the leading eigenspace of ``tau.T @ tau``
(``tau``) and the leading eigenspace after removing per-task self-Gram terms
(``cross``). It then reports zero-shot accuracy on the same deterministic
500-example test subset for each of eight tasks. This is an empirical comparison
at fixed ranks; the script makes no claim that either basis dominates in general.

Every model and dataset is resolved through an immutable revision in
``upstream_revisions.json``. The output contains per-task accuracies only and
refuses to replace an existing file.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import CLIPConfig, CLIPModel, CLIPProcessor, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actnull.null import LazyCheckpoint, load_delta


TASKS = (
    "dtd", "eurosat", "gtsrb", "mnist", "resisc45",
    "stanford-cars", "sun397", "svhn",
)
PROMPT_TEMPLATES = (
    "a photo of a {}.", "a photo of the {}.", "a blurry photo of a {}.",
)

from prompt_templates import BY_TASK as TASK_TEMPLATES  # noqa: E402


def pinned_weight_snapshot(repository: str, revision: str) -> Path:
    """Materialize only checkpoint files at one immutable repository revision."""
    return Path(
        snapshot_download(
            repo_id=repository,
            revision=revision,
            allow_patterns=("*.safetensors*", "*.bin", "*.json"),
        )
    )


def dataset_specifications(pins: dict[str, object]) -> dict[str, dict[str, object]]:
    """Resolve the eight evaluation datasets from the shared revision record."""
    primary = pins["calibration_datasets"]
    alternative = pins["calibration_composition_datasets"]
    result: dict[str, dict[str, object]] = {}
    for task in TASKS:
        source = task.replace("-", "_")
        if source in alternative:
            record = alternative[source]
            result[task] = {
                "repository": record["repository"],
                "config": record["config"],
                "split": record["split"],
                "revision": record["revision"],
            }
        else:
            if task not in primary["revision_by_source"]:
                raise ValueError(f"no pinned evaluation dataset revision for {task!r}")
            result[task] = {
                "repository": primary["repository_template"].format(source=task),
                "config": None,
                "split": "test",
                "revision": primary["revision_by_source"][task],
            }
    return result


def text_embeddings(model, tokenizer, prompts: list[str], device: str) -> torch.Tensor:
    """Use an explicit projected-text path, stable across Transformers wrappers."""
    tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    output = model.text_model(**{key: value.to(device) for key, value in tokens.items()})
    return model.text_projection(output.pooler_output)


def image_embeddings(model, pixels: torch.Tensor, device: str) -> torch.Tensor:
    output = model.vision_model(pixel_values=pixels.to(device))
    return model.visual_projection(output.pooler_output)


def classifier_heads(model, tokenizer, names: list[str], device: str,
                     task: str | None = None) -> torch.Tensor:
    templates = TASK_TEMPLATES.get(task, PROMPT_TEMPLATES)
    with torch.no_grad():
        embeddings = []
        for name in names:
            prompts = [template.format(name.replace("_", " "))
                       for template in templates]
            values = text_embeddings(model, tokenizer, prompts, device)
            values = values / values.norm(dim=-1, keepdim=True)
            average = values.mean(0)
            embeddings.append(average / average.norm())
    return torch.stack(embeddings)


def test_set(specification: dict[str, object], count: int, seed: int):
    from datasets import load_dataset

    arguments = {
        "path": specification["repository"],
        "split": specification["split"],
        "revision": specification["revision"],
    }
    if specification["config"]:
        arguments["name"] = specification["config"]
    dataset = load_dataset(**arguments)
    dataset = dataset.shuffle(seed=seed).select(range(min(count, len(dataset))))
    label = "label" if "label" in dataset.features else next(
        key for key in dataset.features if "label" in key
    )
    image = "image" if "image" in dataset.features else next(
        key for key in dataset.features if "im" in key
    )
    return dataset, image, label, dataset.features[label].names


def evaluate(model, processor, tokenizer, test_sets, device: str, batch: int):
    accuracies = {}
    for task, (dataset, image, label, names) in test_sets.items():
        heads = classifier_heads(model, tokenizer, names, device, task)
        correct = total = 0
        with torch.no_grad():
            for start in range(0, len(dataset), batch):
                chunk = dataset[start:start + batch]
                inputs = processor(
                    images=[item.convert("RGB") for item in chunk[image]],
                    return_tensors="pt",
                )
                features = image_embeddings(model, inputs["pixel_values"], device)
                features = features / features.norm(dim=-1, keepdim=True)
                prediction = (features @ heads.T).argmax(-1).cpu()
                correct += int((prediction == torch.tensor(chunk[label])).sum())
                total += len(prediction)
        accuracy = 100.0 * correct / total
        stderr = 100.0 * math.sqrt(max(accuracy / 100.0 * (1 - accuracy / 100.0), 0.0) / total)
        accuracies[task] = accuracy
        accuracies[task + "__stderr"] = stderr
        accuracies[task + "__n"] = total
        print(f"    {task:14s} {accuracy:5.2f}% +- {stderr:4.2f} (n={total})", flush=True)
    return accuracies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-record", type=Path, required=True)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--lambda-scale", type=float, default=0.3)
    parser.add_argument("--test-examples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pins = json.loads(args.revision_record.read_text(encoding="utf-8"))
    if args.model_id not in pins["base_models"]:
        parser.error(f"no pinned base-model record for {args.model_id!r}")
    architecture = args.model_id.removeprefix("openai/")
    specialist = pins["full_finetuning_specialists"].get(architecture)
    if specialist is None:
        parser.error(f"no pinned specialist family for {architecture!r}")
    if set(TASKS) - set(specialist["revision_by_task"]):
        parser.error("specialist revision record does not cover all evaluation tasks")

    base_record = pins["base_models"][args.model_id]
    weight_revision = base_record["task_vector_weights_revision"]
    configuration_revision = base_record[
        "activation_covariance_config_preprocessor_weights_revision"
    ]
    configuration = CLIPConfig.from_pretrained(
        args.model_id, revision=configuration_revision
    )
    model = CLIPModel.from_pretrained(
        args.model_id, revision=weight_revision, config=configuration
    ).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(
        args.model_id, revision=configuration_revision
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id, revision=configuration_revision
    )

    dataset_specs = dataset_specifications(pins)
    print("[eval] loading pinned test subsets", flush=True)
    test_sets = {
        task: test_set(dataset_specs[task], args.test_examples, args.seed)
        for task in TASKS
    }

    base_path = pinned_weight_snapshot(args.model_id, weight_revision)
    base = LazyCheckpoint(str(base_path), "vision_model.")
    experts = {}
    for task in TASKS:
        repository = specialist["repository_template"].format(task=task)
        revision = specialist["revision_by_task"][task]
        experts[task] = LazyCheckpoint(
            str(pinned_weight_snapshot(repository, revision)), "vision_model."
        )

    linear_modules = [
        name for name, module in model.vision_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    print(f"[eval] loading task vectors for {len(linear_modules)} modules", flush=True)
    deltas, original = {}, {}
    state = model.vision_model.state_dict()
    for module in linear_modules:
        key = module + ".weight"
        if key not in base or any(key not in expert for expert in experts.values()):
            continue
        values = [load_delta(base, experts[task], key, "cpu") for task in TASKS]
        if any(float(value.norm()) == 0 for value in values):
            continue
        deltas[module] = values
        original[module] = state[key].clone()

    def apply_projection(kind: str, rank: int) -> None:
        for module, values in deltas.items():
            cumulative = sum(values)
            gram = (cumulative.T @ cumulative).double()
            if kind == "cross":
                gram = gram - sum((value.T @ value).double() for value in values)
            if kind == "full":
                update = cumulative
            else:
                effective_rank = min(rank, gram.shape[0] - 1)
                _, vectors = torch.linalg.eigh(gram)
                basis = vectors.flip(1)[:, :effective_rank].float()
                update = sum(value @ basis @ basis.T for value in values)
            target = state[module + ".weight"]
            target.copy_(
                original[module] + args.lambda_scale * update.to(target.device)
            )

    def restore() -> None:
        for module in deltas:
            state[module + ".weight"].copy_(original[module])

    result = {}
    print("\n[eval] zero-shot", flush=True)
    restore()
    result["zeroshot"] = evaluate(
        model, processor, tokenizer, test_sets, args.device, args.batch_size
    )
    print("\n[eval] uncompressed task arithmetic", flush=True)
    apply_projection("full", 0)
    result["full"] = evaluate(
        model, processor, tokenizer, test_sets, args.device, args.batch_size
    )
    def checkpoint() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    checkpoint()
    for kind in ("tau", "cross"):
        for rank in args.ranks:
            print(f"\n[eval] {kind} r={rank}", flush=True)
            apply_projection(kind, rank)
            result[f"{kind}_{rank}"] = evaluate(
                model, processor, tokenizer, test_sets, args.device, args.batch_size
            )
            checkpoint()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")
    except FileExistsError:
        parser.error(f"refusing to overwrite existing result: {args.out}")
    print(f"\n[eval] wrote {args.out}")
    for setting, values in result.items():
        print(f"  {setting:>12s} {stats.fmean(values.values()):9.3f}%")


if __name__ == "__main__":
    main()
