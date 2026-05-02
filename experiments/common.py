"""Shared helpers for dataset-backed CQRCD experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from config import CONCENTRATION_THRESHOLD, DEFAULT_RETRIEVER, N_NEIGHBORS
from modules.cqrcd_filter import CQRCDFilter
from modules.neighbor_generator import NeighborGenerator
from modules.retriever import DenseRetriever


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
DATA_DIR = Path("data")


def ensure_output_dirs(output_dir: str | Path = RESULTS_DIR) -> Dict[str, Path]:
    root = Path(output_dir)
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return {"root": root, "figures": figures, "tables": tables}


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_bundle(data_dir: str | Path = DATA_DIR) -> Dict[str, List[Dict]]:
    data_dir = Path(data_dir)
    bundle = {
        "tasks": load_json(data_dir / "asb_tasks.json"),
        "legitimate": load_json(data_dir / "legitimate_kb.json"),
        "blackbox": load_json(data_dir / "adversarial_bb.json"),
        "whitebox": load_json(data_dir / "adversarial_wb.json"),
    }
    return bundle


def init_retriever(name: str = DEFAULT_RETRIEVER):
    return DenseRetriever(name)


def build_neighbor_generator(retriever, smoke: bool = False):
    model_name = None if smoke else "Vamsi/T5_Paraphrase_Paws"
    return NeighborGenerator(model_name=model_name, retriever=retriever)


def build_cqrcd_filter(
    retriever,
    smoke: bool = False,
    threshold: float = CONCENTRATION_THRESHOLD,
    n_neighbors: int = N_NEIGHBORS,
):
    ng = build_neighbor_generator(retriever, smoke=smoke)
    return CQRCDFilter(
        retriever=retriever,
        neighbor_generator=ng,
        threshold=threshold,
        n_neighbors=n_neighbors,
    )


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_json(payload, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def save_figure(fig, path_base: str | Path) -> None:
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def representative_tasks(tasks: Sequence[Dict]) -> List[Dict]:
    return [task for task in tasks if task.get("is_representative")]


def build_scenarios(
    tasks: Sequence[Dict],
    legitimate_docs: Sequence[Dict],
    attack_docs: Sequence[Dict],
) -> List[Dict]:
    task_by_id = {task["task_id"]: task for task in tasks}
    legit_by_domain: Dict[str, List[Dict]] = {}
    for doc in legitimate_docs:
        legit_by_domain.setdefault(doc["domain"], []).append(doc)

    scenarios = []
    for attack_doc in attack_docs:
        task = task_by_id.get(attack_doc["task_id"])
        if task is None:
            continue
        scenarios.append(
            {
                "scenario_id": attack_doc["scenario_id"],
                "query": task["query"],
                "task_id": task["task_id"],
                "domain": task["domain"],
                "available_tools": list(task.get("available_tools", [])),
                "attack_tool": attack_doc["attack_tool"],
                "attack_doc": dict(attack_doc),
                "legitimate_docs": [dict(doc) for doc in legit_by_domain.get(task["domain"], [])],
            }
        )
    return scenarios


def sample_for_smoke(rows: Sequence[Dict], limit: int = 40) -> List[Dict]:
    return list(rows[:limit])


def summarize_group(values: Iterable[float], label: str) -> Dict[str, float | str]:
    series = pd.Series(list(values), dtype="float64")
    return {
        "group": label,
        "count": int(series.shape[0]),
        "mean": float(series.mean()) if not series.empty else 0.0,
        "std": float(series.std(ddof=0)) if not series.empty else 0.0,
        "min": float(series.min()) if not series.empty else 0.0,
        "max": float(series.max()) if not series.empty else 0.0,
        "median": float(series.median()) if not series.empty else 0.0,
    }
