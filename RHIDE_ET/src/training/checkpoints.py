#src/training/checkpoints.py
from __future__ import annotations

import os
import json
import pickle
from typing import Any, Dict, Optional
from datetime import datetime

# Expecting your project already has these utilities:
# - save_model(model, path) that writes a TF SavedModel or H5 under a dir.
# - load_model(path) if you later add restore flows.
from src.models.serialization import save_model


def _meta_path(run_dir: str) -> str:
    return os.path.join(run_dir, "meta.json")


def _round_ckpt_dir(run_dir: str, round_idx: int) -> str:
    return os.path.join(run_dir, f"ckpt_{round_idx:05d}")


def checkpoint_init_state(run_dir: str, meta: Dict[str, Any]) -> None:
    """
    Initialize checkpoint folder and meta if absent.
    meta can include arbitrary run descriptors (configs, git sha, etc.).
    """
    os.makedirs(run_dir, exist_ok=True)
    mp = _meta_path(run_dir)
    if not os.path.exists(mp):
        meta = {
            **meta,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "latest_round": -1,
        }
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2)


def checkpoint_save_round(
    run_dir: str,
    round_idx: int,
    model,
    edge_states: Dict[str, Any],
    solver_warm: Any,
    logs: Dict[str, Any],
    rl_agents: Optional[Dict[str, Any]] = None,  # map edge_id -> agent (must have .save(dir))
) -> str:
    """
    Save a full snapshot for this round:
      - Model weights (SavedModel) under ckpt_<round>/model/
      - Edge state dict (pickle)  -> edge_states.pkl
      - Solver warm start (pickle)-> solver_warm.pkl
      - Round logs/metrics (json) -> logs.json
      - RL agent checkpoints per edge under ckpt_<round>/rl/<edge_id>/
    Returns the checkpoint directory path.
    """
    ckpt_dir = _round_ckpt_dir(run_dir, round_idx)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1) Model
    model_path = os.path.join(ckpt_dir, "model")
    save_model(model, model_path)

    # 2) Edge states
    with open(os.path.join(ckpt_dir, "edge_states.pkl"), "wb") as f:
        pickle.dump(edge_states, f)

    # 3) Solver warm-start blob (e.g., dict with z/beta/k)
    with open(os.path.join(ckpt_dir, "solver_warm.pkl"), "wb") as f:
        pickle.dump(solver_warm, f)

    # 4) Logs/metrics
    with open(os.path.join(ckpt_dir, "logs.json"), "w") as f:
        json.dump(logs, f, indent=2)

    # 5) RL agents (per edge)
    if rl_agents:
        rl_root = os.path.join(ckpt_dir, "rl")
        os.makedirs(rl_root, exist_ok=True)
        for edge_id, agent in rl_agents.items():
            ed = os.path.join(rl_root, str(edge_id))
            os.makedirs(ed, exist_ok=True)
            # Agent is expected to expose save(dir) per src/rl/agents.py
            try:
                agent.save(ed)
            except Exception as ex:
                # Don't fail the whole checkpoint; record the error
                with open(os.path.join(ed, "ERROR.txt"), "w") as ef:
                    ef.write(f"Failed to save RL agent for edge {edge_id}: {ex}\n")

    # Update meta latest pointer
    mp = _meta_path(run_dir)
    if os.path.exists(mp):
        with open(mp, "r") as f:
            meta = json.load(f)
    else:
        meta = {"created_at": datetime.utcnow().isoformat() + "Z"}
    meta["latest_round"] = int(round_idx)
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2)

    return ckpt_dir


def checkpoint_load_latest(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Returns {"round": int, "ckpt_dir": str} for the latest checkpoint, or None.
    """
    mp = _meta_path(run_dir)
    if not os.path.exists(mp):
        return None
    with open(mp, "r") as f:
        meta = json.load(f)
    latest = int(meta.get("latest_round", -1))
    if latest < 0:
        return None
    return {"round": latest, "ckpt_dir": _round_ckpt_dir(run_dir, latest)}

